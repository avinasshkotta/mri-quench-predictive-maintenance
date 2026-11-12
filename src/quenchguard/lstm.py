"""PyTorch LSTM detector, training loop and gradient-based attribution.

Architecture
------------
    (batch, 48 windows, 36 features)
        -> LSTM(36, 64, 2 layers, dropout 0.3)   # reads the 12 h history
        -> last hidden state
        -> Dropout(0.3) -> Linear(64, 1)
        -> logit  (trained with BCE-with-logits + capped class weight)

The prediction is attached to the **last** window of the sequence: "will a quench
happen within the next 96 h, given the last 12 hours of all four sensors?"
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from .config import Config, default_config


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_backend(cfg: Config) -> None:
    """Apply backend-level settings (see ``TrainConfig.disable_cudnn``)."""
    if cfg.train.disable_cudnn and torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = False


def resolve_device(cfg: Config) -> torch.device:
    if cfg.train.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.train.device)


def device_summary(cfg: Config | None = None) -> str:
    cfg = cfg or default_config()
    dev = resolve_device(cfg)
    if dev.type == "cuda":
        return f"{dev} ({torch.cuda.get_device_name(0)})"
    return f"{dev} (PyTorch {torch.__version__})"


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class LSTMDetector(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)      # logits, shape (batch,)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def _forward_scores(model: nn.Module, X: torch.Tensor, batch: int = 1024) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            chunks.append(torch.sigmoid(model(X[i:i + batch])).cpu())
    return torch.cat(chunks) if chunks else torch.empty(0)


def train_lstm(cfg: Config | None = None, splits: dict | None = None, verbose: bool = True
               ) -> tuple[LSTMDetector, list[dict[str, Any]], torch.device]:
    """Train with early stopping on validation PR-AUC; returns (best model, history, device)."""
    cfg = cfg or default_config()
    set_seed(cfg.train.seed)
    configure_backend(cfg)
    device = resolve_device(cfg)

    tr, va = splits["train"], splits["val"]
    n_features = tr.X.shape[2]

    Xtr = torch.from_numpy(tr.X).to(device)
    ytr = torch.from_numpy(tr.y.astype(np.float32)).to(device)
    Xva = torch.from_numpy(va.X).to(device)
    yva = va.y.astype(np.float32)

    prevalence = float(tr.y.mean())
    pos_weight = float(np.clip((1.0 - prevalence) / max(prevalence, 1e-6),
                               1.0, cfg.train.max_pos_weight))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    model = LSTMDetector(n_features, cfg.train.hidden_size, cfg.train.num_layers,
                         cfg.train.dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate,
                              weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.train.epochs)

    if verbose:
        print(f"device={device_summary(cfg)} | params={count_parameters(model):,} | "
              f"pos_weight={pos_weight:.1f} | train={len(tr):,} val={len(va):,}")

    best = {"ap": -1.0, "state": None, "epoch": -1}
    history: list[dict[str, Any]] = []
    stale = 0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        total, n_batches = 0.0, 0
        for i in range(0, len(perm), cfg.train.batch_size):
            idx = perm[i:i + cfg.train.batch_size]
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(Xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += float(loss.detach())
            n_batches += 1
        sched.step()

        val_scores = _forward_scores(model, Xva).numpy()
        val_ap = average_precision_score(yva, val_scores) if len(yva) else 0.0
        val_roc = roc_auc_score(yva, val_scores) if len(yva) and len(np.unique(yva)) > 1 else 0.0
        train_loss = total / max(n_batches, 1)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_ap": float(val_ap), "val_roc": float(val_roc),
                        "lr": sched.get_last_lr()[0]})

        improved = val_ap > best["ap"] + 1e-4
        if improved:
            best = {"ap": float(val_ap), "state": {k: v.detach().cpu().clone()
                                                    for k, v in model.state_dict().items()},
                    "epoch": epoch}
            stale = 0
        else:
            stale += 1

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == cfg.train.epochs):
            print(f"  epoch {epoch:>2}/{cfg.train.epochs} | train_loss {train_loss:.4f} | "
                  f"val PR-AUC {val_ap:.4f} | val ROC-AUC {val_roc:.4f} | "
                  f"{'best' if improved else f'stale {stale}'}")

        if stale >= cfg.train.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best epoch {best['epoch']}, "
                      f"val PR-AUC {best['ap']:.4f})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.to(device)
    return model, history, device


def predict_scores(model: nn.Module, X: np.ndarray, device: torch.device,
                   batch: int = 1024) -> np.ndarray:
    """Sigmoid probabilities for an array of sequences."""
    if len(X) == 0:
        return np.empty(0, dtype=np.float32)
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    return _forward_scores(model, Xt, batch=batch).numpy()


# --------------------------------------------------------------------------------------
# explainability
# --------------------------------------------------------------------------------------
def gradient_importance(model: nn.Module, X: np.ndarray, feature_names: list[str],
                        device: torch.device, max_samples: int = 2000,
                        batch: int = 256, seed: int = 0) -> "Any":
    """Mean |gradient x input| per feature (inputs are standardised, so comparable)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    if len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    Xs = X[idx]

    model.eval()
    total = np.zeros(len(feature_names), dtype=np.float64)
    seen = 0
    # cuDNN refuses to run an RNN backward outside training mode -> use the native
    # backend for attribution (dropout stays off, so results are deterministic).
    with torch.backends.cudnn.flags(enabled=False):
        for i in range(0, len(Xs), batch):
            xb = torch.from_numpy(Xs[i:i + batch]).to(device)
            xb.requires_grad_(True)
            logits = model(xb)
            grad = torch.autograd.grad(logits.sum(), xb, retain_graph=False)[0]
            contrib = (grad * xb).abs().mean(dim=(0, 1)).detach().cpu().numpy()
            total += contrib * xb.shape[0]
            seen += xb.shape[0]

    imp = total / max(seen, 1)
    return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
        "importance", ascending=False).reset_index(drop=True)


def explain_sequence(model: nn.Module, x: np.ndarray, feature_names: list[str],
                     device: torch.device, top_k: int = 8) -> "Any":
    """Per-window, per-feature attribution for a single alert sequence."""
    import pandas as pd

    model.eval()
    xb = torch.from_numpy(np.ascontiguousarray(x[None, ...])).to(device)
    xb.requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        logits = model(xb)
        grad = torch.autograd.grad(logits.sum(), xb)[0]
    contrib = (grad * xb)[0].detach().cpu().numpy()          # (seq_len, n_features)
    per_feature = contrib.mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "contribution": per_feature})
    df["abs"] = df["contribution"].abs()
    df = df.sort_values("abs", ascending=False).head(top_k).drop(columns="abs")
    return df.reset_index(drop=True)


def save_model(model: nn.Module, path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "meta": meta or {},
                "arch": {"n_features": model.lstm.input_size,
                         "hidden_size": model.lstm.hidden_size,
                         "num_layers": model.lstm.num_layers}}, path)


def load_model(path, device: torch.device | None = None) -> LSTMDetector:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arch = payload["arch"]
    model = LSTMDetector(**arch)
    model.load_state_dict(payload["state_dict"])
    return model.to(device or torch.device("cpu")).eval()
