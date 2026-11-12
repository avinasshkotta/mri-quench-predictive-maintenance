"""Tests for horizon labelling: no future leakage, correct offline embargo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.config import default_config
from quenchguard.labeling import add_labels, label_report

CFG = default_config()


def _frame(magnet: str, start: str, n: int = 60 * 48) -> pd.DataFrame:
    """15-min windows over `n` steps starting at `start`."""
    idx = pd.date_range(start, periods=n, freq="15min")
    return pd.DataFrame({"magnet_id": magnet, "window_end": idx})


def test_label_appears_only_inside_the_horizon():
    features = _frame("MRI-01", "2025-03-01")
    events = pd.DataFrame({
        "magnet_id": ["MRI-01"],
        "event_time": [pd.Timestamp("2025-03-03 00:00")],
        "event_type": ["quench"],
    })
    out = add_labels(features, CFG, events, verbose=False)

    horizon = pd.Timedelta(hours=CFG.window.horizon_hours)
    ev = events.event_time.iloc[0]
    pos = out[out.label == 1]
    assert len(pos) > 0
    assert (pos.window_end > ev - horizon).all()      # not before the horizon
    assert (pos.window_end <= ev).all()               # never at/after the failure
    neg = out[out.label == 0]
    before = neg[neg.window_end < ev]
    assert (before.window_end <= ev - horizon).all()  # everything earlier is negative


def test_windows_after_the_failure_are_excluded_not_negative():
    features = _frame("MRI-01", "2025-03-01")
    events = pd.DataFrame({
        "magnet_id": ["MRI-01"],
        "event_time": [pd.Timestamp("2025-03-02 00:00")],
        "event_type": ["quench"],
    })
    out = add_labels(features, CFG, events, verbose=False)
    repair_end = events.event_time.iloc[0] + pd.Timedelta(
        days=CFG.window.offline_days_after_event)
    inside_repair = out[(out.window_end >= events.event_time.iloc[0])
                        & (out.window_end < repair_end)]
    assert inside_repair.is_offline.all()
    assert (~inside_repair.eligible).all()
    after = out[out.window_end >= repair_end]
    assert after.eligible.all()


def test_labels_are_invariant_to_the_future():
    """Truncating the frame must not change any label that remains."""
    events = pd.DataFrame({
        "magnet_id": ["MRI-01"],
        "event_time": [pd.Timestamp("2025-03-03 12:00")],
        "event_type": ["quench"],
    })
    full = _frame("MRI-01", "2025-03-01", n=60 * 60)
    cut = full.iloc[: len(full) // 2]
    a = add_labels(full, CFG, events, verbose=False).iloc[: len(cut)]
    b = add_labels(cut, CFG, events, verbose=False)
    pd.testing.assert_series_equal(a.label.reset_index(drop=True),
                                   b.label.reset_index(drop=True))


def test_other_magnets_do_not_leak_labels():
    features = pd.concat([
        _frame("MRI-01", "2025-03-01"),
        _frame("MRI-02", "2025-03-01"),
    ], ignore_index=True)
    events = pd.DataFrame({
        "magnet_id": ["MRI-01"],
        "event_time": [pd.Timestamp("2025-03-02 00:00")],
        "event_type": ["quench"],
    })
    out = add_labels(features, CFG, events, verbose=False)
    assert out.loc[out.magnet_id == "MRI-02", "label"].sum() == 0


def test_label_report_shape():
    features = _frame("MRI-01", "2025-03-01", n=60 * 24 * 10)
    events = pd.DataFrame({"magnet_id": ["MRI-01"],
                           "event_time": [pd.Timestamp("2025-03-05")],
                           "event_type": ["quench"]})
    out = add_labels(features, CFG, events, verbose=False)
    rep = label_report(out)
    assert list(rep.columns) == ["magnet_id", "windows", "positive", "positive_pct",
                                 "quench_events", "first_window", "last_window"]
    assert rep.positive.iloc[0] > 0
