"""Time-ordered splits, scaling and LSTM sequence construction.

Rules
-----
1. Split boundaries are **global timestamps** (60 % / 20 % / 20 % of the fleet
   timeline). Every magnet is cut at the same instant, so no magnet's future can
   appear in training.
2. A sequence is kept only if all 48 windows fall inside one split *and* are
   contiguous (no gaps), so the model never learns from an interrupted history.
3. ``StandardScaler`` is fit on **training rows only** and then applied to val/test.
4. The sequence label is the label of the **last** window (the prediction instant);
   everything before it is context the model is allowed to see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from .config import Config, default_config
from .features import FEATURE_COLUMNS, ensure_features
from .labeling import add_labels


@dataclass
class SplitArrays:
    """Model-ready arrays for one split."""

    X: np.ndarray                 # (n_seq, seq_len, n_features) float32
    y: np.ndarray                 # (n_seq,) int64
    times: np.ndarray             # (n_seq,) datetime64 of the last window
    magnets: np.ndarray           # (n_seq,) magnet id strings
    features: pd.DataFrame        # rows for the last window of each sequence

    def __len__(self) -> int:
        return len(self.y)

    @property
    def prevalence(self) -> float:
        return float(self.y.mean()) if len(self.y) else 0.0


def split_boundaries(cfg: Config, df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Global cut timestamps: (train_end, val_end], computed from absolute time."""
    t0, t1 = df["window_end"].min(), df["window_end"].max()
    span = t1 - t0
    train_end = t0 + span * cfg.split.train_frac
    val_end = train_end + span * cfg.split.val_frac
    return train_end, val_end


def add_split(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Tag every row with its split name using global time cut-points."""
    cfg = cfg or default_config()
    train_end, val_end = split_boundaries(cfg, df)
    out = df.copy()
    out["split"] = np.where(
        out["window_end"] <= train_end, "train",
        np.where(out["window_end"] <= val_end, "val", "test"),
    )
    return out


def prepare_data(cfg: Config | None = None, verbose: bool = True) -> tuple[dict, StandardScaler, pd.DataFrame]:
    """features -> labels -> split -> scale -> sequences.

    Returns
    -------
    splits   : {"train": SplitArrays, "val": SplitArrays, "test": SplitArrays}
    scaler   : StandardScaler fit on training rows only
    labelled : the full labelled feature frame (with split column) for analysis
    """
    cfg = cfg or default_config()
    feat = ensure_features(cfg, verbose=verbose)
    labelled = add_labels(feat, cfg, verbose=verbose)
    labelled = add_split(labelled, cfg)

    labelled = labelled[labelled.eligible].reset_index(drop=True)
    scaler = StandardScaler().fit(labelled.loc[labelled.split == "train", FEATURE_COLUMNS])

    splits: dict[str, SplitArrays] = {}
    for name in ("train", "val", "test"):
        part = labelled[labelled.split == name].copy()
        arr = scaler.transform(part[FEATURE_COLUMNS]).astype(np.float32)
        part[[f"s__{c}" for c in FEATURE_COLUMNS]] = arr
        splits[name] = _build_sequences(part, cfg)
        if verbose:
            s = splits[name]
            print(f"  {name:>5}: {len(s):>6,} sequences | {s.X.shape[1]} steps x "
                  f"{s.X.shape[2]} features | prevalence {s.prevalence:.1%}")
    return splits, scaler, labelled


def _build_sequences(part: pd.DataFrame, cfg: Config) -> SplitArrays:
    seq_len, step = cfg.window.seq_len, pd.Timedelta(minutes=cfg.window.step_minutes)
    scaled_cols = [f"s__{c}" for c in FEATURE_COLUMNS]

    xs, ys, ts, ms, last_rows = [], [], [], [], []
    code = {"train": 0, "val": 1, "test": 2}
    part = part.copy()
    part["_code"] = part["split"].map(code)

    for magnet, g in part.groupby("magnet_id", sort=True):
        g = g.sort_values("window_end").reset_index(drop=True)
        if len(g) < seq_len:
            continue
        arr = g[scaled_cols].to_numpy(np.float32)
        times = g["window_end"].to_numpy()
        codes = g["_code"].to_numpy()

        wins = sliding_window_view(arr, seq_len, axis=0).transpose(0, 2, 1)
        t_wins = sliding_window_view(times, seq_len)
        c_wins = sliding_window_view(codes, seq_len)

        contiguous = (t_wins[:, -1] - t_wins[:, 0]) == np.timedelta64(
            (seq_len - 1) * cfg.window.step_minutes, "m")
        same_split = c_wins.min(axis=1) == c_wins.max(axis=1)
        keep = contiguous & same_split

        xs.append(wins[keep])
        ys.append(g["label"].to_numpy(np.int64)[seq_len - 1:][keep])
        ts.append(t_wins[:, -1][keep])
        ms.append(np.repeat(magnet, int(keep.sum())))
        last_rows.append(g.iloc[np.arange(seq_len - 1, len(g))[keep]])

    if not xs:
        return SplitArrays(np.empty((0, seq_len, len(FEATURE_COLUMNS)), np.float32),
                           np.empty(0, np.int64), np.empty(0, "datetime64[ns]"),
                           np.empty(0, object), part.iloc[0:0])

    return SplitArrays(
        X=np.concatenate(xs),
        y=np.concatenate(ys),
        times=np.concatenate(ts),
        magnets=np.concatenate(ms),
        features=pd.concat(last_rows, ignore_index=True)
        .drop(columns=[c for c in scaled_cols if c in last_rows[0].columns]),
    )
