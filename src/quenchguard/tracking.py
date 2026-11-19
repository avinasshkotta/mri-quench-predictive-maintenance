"""MLflow experiment tracking + model registry.

One run per pipeline execution:

* **params**  — the reviewer-facing config (window, threshold, cost model)
* **metrics** — every number quoted in the README, per model
* **artifacts** — metrics.json, sweeps, figures, model card
* **model**   — the trained LSTM logged as a PyTorch flavour and registered
  as ``quenchguard_lstm`` with the ``champion`` alias (the modern replacement
  for the deprecated Staging/Production stages)

Tracking is a *file* store inside the repo (``mlruns/``, git-ignored), so the
whole experiment history travels with the checkout and needs no server.

Design note: logging failures never fail the pipeline — a tracking outage
must not cost a 3-minute training run. Failures are printed as warnings and
``log_run`` returns ``None``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from .config import Config, MODELS_DIR, REPO_ROOT, REPORTS_DIR, default_config

# SQLite backend: MLflow 3.x put the plain filesystem store into maintenance
# mode, so experiment + registry metadata live in a small database file.
# Artifacts still land in the repo-local ``mlartifacts/`` (both git-ignored).
MLFLOW_DB = REPO_ROOT / "mlflow.db"
MLRUNS_DIR = REPO_ROOT / "mlruns"
DEFAULT_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
EXPERIMENT_NAME = "quenchguard"
MODEL_NAME = "quenchguard_lstm"


def _flattened_metrics(report: dict[str, Any]) -> dict[str, float]:
    """{model}_{metric} for every scalar the README quotes."""
    metrics: dict[str, float] = {}
    for model_name, res in report.items():
        if model_name.startswith("_"):
            continue
        for section in ("classification", "events", "alerts", "cost"):
            block = res.get(section) or {}
            for key, value in block.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[f"{model_name}_{key}"] = float(value)
    return metrics


def log_run(cfg: Config | None = None, report: dict[str, Any] | None = None,
            thresholds: dict[str, float] | None = None,
            history: list[dict] | None = None, model: Any = None,
            extra_artifacts: list[Path] | None = None,
            tags: dict[str, str] | None = None,
            tracking_uri: str | None = None) -> dict[str, Any] | None:
    """Log one pipeline run to MLflow and register the LSTM as ``champion``.

    ``tracking_uri`` defaults to the repo-local file store; tests pass a
    temporary directory so they never touch real experiment history.

    Returns ``{"run_id", "model_uri", "registered_version"}`` or ``None`` on
    any tracking failure (logged as a warning).
    """
    cfg = cfg or default_config()
    report = report or {}
    thresholds = thresholds or {}

    try:
        mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="pipeline") as run:
            # ---- params: the config a reviewer would challenge ------------------
            mlflow.log_params({
                "seq_len": cfg.window.seq_len,
                "step_minutes": cfg.window.step_minutes,
                "horizon_hours": cfg.window.horizon_hours,
                "min_lead_hours": cfg.window.min_lead_hours,
                "learning_rate": cfg.train.learning_rate,
                "hidden_size": cfg.train.hidden_size,
                "num_layers": cfg.train.num_layers,
                "seed": cfg.train.seed,
                "false_alert_budget": cfg.threshold.false_alert_budget_per_magnet_month,
                "emergency_total_usd": cfg.cost.emergency_total,
                "planned_total_usd": cfg.cost.planned_total,
                **{f"threshold_{k}": v for k, v in thresholds.items()},
            })

            # ---- metrics: every README number ------------------------------------
            metrics = _flattened_metrics(report)
            if history:
                metrics["training_best_val_pr_auc"] = float(
                    max(h["val_ap"] for h in history))
                metrics["training_epochs_run"] = float(len(history))
            if metrics:
                mlflow.log_metrics(metrics)

            # ---- artifacts -------------------------------------------------------
            for name in ("metrics.json", "threshold_sweep.csv",
                         "training_history.csv"):
                path = REPORTS_DIR / name
                if path.exists():
                    mlflow.log_artifact(str(path))
            figures = REPORTS_DIR / "figures"
            if figures.exists():
                mlflow.log_artifacts(str(figures))
            for path in extra_artifacts or []:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
            card = REPO_ROOT / "docs" / "model_card.md"
            if card.exists():
                mlflow.log_artifact(str(card))

            # ---- serving triple: registry-backed serving reads these ------------
            serving_files = ("lstm_detector.pt", "scaler.joblib", "hist_gbdt.joblib")
            if all((MODELS_DIR / name).exists() for name in serving_files):
                for name in serving_files:
                    mlflow.log_artifact(str(MODELS_DIR / name), artifact_path="serving")

            # ---- model: log + register with the champion alias -------------------
            model_uri, version = None, None
            if model is not None:
                cpu_model = _to_cpu_copy(model)
                # `pickle` avoids MLflow 3.x's pt2 tracing requirements while
                # still producing a fully loadable pytorch flavour.
                mlflow.pytorch.log_model(cpu_model, name="model",
                                         serialization_format="pickle")
                model_uri = f"runs:/{run.info.run_id}/model"
                client = MlflowClient()
                mv = mlflow.register_model(model_uri, MODEL_NAME)
                # registry writes are eventually consistent; wait for the version
                client.get_registered_model(MODEL_NAME)
                client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
                version = int(mv.version)
                model_uri = f"models:/{MODEL_NAME}@champion"

            if tags:
                mlflow.set_tags(tags)

            result = {"run_id": run.info.run_id, "model_uri": model_uri,
                      "registered_version": version}
        return result

    except Exception as exc:  # tracking must never kill a training run
        warnings.warn(f"MLflow logging failed ({type(exc).__name__}: {exc}); "
                      "pipeline continues without experiment tracking.",
                      RuntimeWarning, stacklevel=2)
        return None


def _to_cpu_copy(model: Any) -> Any:
    """Deep-copy the model onto CPU so MLflow never touches CUDA state."""
    import copy

    return copy.deepcopy(model).cpu().eval()


def load_champion(tracking_uri: str | None = None) -> Any:
    """Load the registered champion LSTM (used by the serving layer, Phase C)."""
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    return mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@champion")


__all__ = ["log_run", "load_champion", "EXPERIMENT_NAME", "MODEL_NAME",
           "DEFAULT_TRACKING_URI"]
