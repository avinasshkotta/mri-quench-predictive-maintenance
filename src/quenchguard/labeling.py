"""Horizon-based labelling — the single most common source of leakage in this task.

Target definition
-----------------
For a window closing at time ``t``:

    label(t) = 1  iff  a quench occurs in  (t, t + horizon_hours]
                 with horizon_hours = 96 h

while the *business* acceptance test is that the first alert for an event lands
at least ``min_lead_hours = 72 h`` before it. The 24 h difference is intentional:
labelling must reach far enough back that the model can act on the early trend,
otherwise every positive sample sits inside the last 72 h where only the cold-head
threshold (27 h before failure) is informative.

Additional guards
-----------------
* Windows inside the post-event repair window are dropped entirely
  (``offline_days_after_event``), so the model never trains on magnet-off data.
* ``ttf_hours`` is stored for analysis but is **never** a model input.
* Split membership is decided by absolute global time, not per-magnet shuffling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, default_config
from .data_gen import load_quench_events


def add_labels(features: pd.DataFrame, cfg: Config | None = None,
               events: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    """Attach ``label``, ``event_time``, ``ttf_hours`` and ``is_offline`` columns."""
    cfg = cfg or default_config()
    events = events if events is not None else load_quench_events(cfg)

    df = features.sort_values(["magnet_id", "window_end"]).reset_index(drop=True).copy()
    label = np.zeros(len(df), dtype=np.int64)
    ttf = np.full(len(df), np.nan)
    ev_time = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    offline = np.zeros(len(df), dtype=bool)

    horizon = pd.Timedelta(hours=cfg.window.horizon_hours)
    repair = pd.Timedelta(days=cfg.window.offline_days_after_event)

    for ev in events.itertuples():
        in_magnet = (df["magnet_id"] == ev.magnet_id).to_numpy()
        dt = ev.event_time - df["window_end"]              # >0 => still in the future
        pos = (dt > pd.Timedelta(0)) & (dt <= horizon) & in_magnet
        label |= pos.astype(np.int64)
        ttf = np.where(pos, dt.dt.total_seconds() / 3600.0, ttf)
        ev_time = ev_time.mask(pos, ev.event_time)

        off = (df["window_end"] >= ev.event_time) & (df["window_end"] < ev.event_time + repair)
        offline |= (off & in_magnet).to_numpy()

    df["label"] = label
    df["event_time"] = ev_time.values
    df["ttf_hours"] = ttf
    df["is_offline"] = offline
    df["eligible"] = ~df["is_offline"]           # rows the model may train/score on

    if verbose:
        n_pos = int((df["label"] == 1).sum())
        n_elig = int(df["eligible"].sum())
        print(f"labels: {n_pos:,} positive windows "
              f"({n_pos / max(n_elig, 1):.1%} of {n_elig:,} eligible) | "
              f"{int(offline.sum()):,} windows dropped as post-event/offline")
    return df


def label_report(labelled: pd.DataFrame) -> pd.DataFrame:
    """Per-magnet class balance and event counts — quick sanity table."""
    rows = []
    for magnet, g in labelled[labelled.eligible].groupby("magnet_id"):
        rows.append({
            "magnet_id": magnet,
            "windows": len(g),
            "positive": int(g.label.sum()),
            "positive_pct": 100.0 * g.label.mean(),
            "quench_events": int(g.loc[g.label == 1, "event_time"].nunique()),
            "first_window": g.window_end.min(),
            "last_window": g.window_end.max(),
        })
    return pd.DataFrame(rows)
