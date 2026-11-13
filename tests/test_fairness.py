"""Tests for the fleet fairness audit (fairness.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.fairness import (EVENT_RECALL_REQUIREMENT,
                                  MAX_SELECTION_RATE_DIFFERENCE,
                                  MAX_SELECTION_RATE_RATIO, audit_verdict,
                                  group_metrics, per_magnet_pr_auc,
                                  run_fairness_audit)


def _frame(seed: int = 0, recall_pattern: str = "equal",
           n_per_magnet: int = 200) -> pd.DataFrame:
    """Synthetic test-score frame for 3 magnets.

    recall_pattern:
      equal   -> every magnet detects its positives
      blind   -> MRI-03 misses all of its positives (under-protected)
    """
    rng = np.random.default_rng(seed)
    rows = []
    for magnet in ("MRI-01", "MRI-02", "MRI-03"):
        y = np.zeros(n_per_magnet, dtype=int)
        y[rng.choice(n_per_magnet, size=10, replace=False)] = 1
        score = rng.uniform(0.0, 0.7, size=n_per_magnet)
        score[y == 1] = rng.uniform(0.9, 1.0, size=10)   # detections
        if recall_pattern == "blind" and magnet == "MRI-03":
            score[y == 1] = rng.uniform(0.0, 0.4, size=10)   # misses
        # unequal alert burden: MRI-02 pages twice as often
        base = {"MRI-01": 0.72, "MRI-02": 0.86, "MRI-03": 0.72}[magnet]
        score[y == 0] = np.clip(score[y == 0] * base * 1.4, 0, 1)
        rows.append(pd.DataFrame({"magnet_id": magnet, "y": y, "score": score}))
    return pd.concat(rows, ignore_index=True)


def test_group_metrics_contract():
    mf = group_metrics(_frame(), threshold=0.8)
    assert set(mf["by_group"]) == {"MRI-01", "MRI-02", "MRI-03"}
    for magnet, row in mf["by_group"].items():
        assert {"selection_rate", "true_positive_rate", "false_positive_rate",
                "count"} <= set(row)
        assert row["count"] == 200
        assert 0.0 <= row["selection_rate"] <= 1.0
    # overall is an aggregate, not a dict of groups
    assert isinstance(mf["overall"]["selection_rate"], float)


def test_verdict_fails_when_magnet_misses_events():
    frame = _frame(recall_pattern="blind")
    v = audit_verdict(group_metrics(frame, 0.8)["by_group"], frame, 0.8)
    assert v["verdict"] == "fail"
    assert v["event_recall_by_magnet"]["MRI-03"] == 0.0
    assert any("systematically missing" in f for f in v["findings"])


def test_verdict_flags_alert_burden_disparity():
    frame = _frame()
    v = audit_verdict(group_metrics(frame, 0.8)["by_group"], frame, 0.8)
    assert v["selection_rate_ratio"] is not None
    assert v["selection_rate_difference"] is not None
    assert v["max_selection_rate_ratio"] == MAX_SELECTION_RATE_RATIO
    assert v["max_selection_rate_difference"] == MAX_SELECTION_RATE_DIFFERENCE
    # the constructed frame makes MRI-02 page ~2x -> disparity is visible
    assert v["selection_rate_ratio"] > 1.0
    if (v["selection_rate_difference"] > MAX_SELECTION_RATE_DIFFERENCE
            or v["selection_rate_ratio"] > MAX_SELECTION_RATE_RATIO):
        assert v["verdict"] in {"warn", "fail"}


def test_clean_magnets_never_alert_do_not_break_parity():
    """A magnet that never pages is correct — parity ignores zero rates."""
    rng = np.random.default_rng(7)
    rows = []
    for magnet, has_events in (("MRI-01", True), ("MRI-02", True), ("MRI-03", False)):
        n = 400
        y = np.zeros(n, dtype=int)
        score = rng.uniform(0.0, 0.2, size=n)
        if has_events:
            y[rng.choice(n, 12, replace=False)] = 1
            score[y == 1] = rng.uniform(0.9, 1.0, size=12)
        rows.append(pd.DataFrame({"magnet_id": magnet, "y": y, "score": score}))
    frame = pd.concat(rows, ignore_index=True)
    v = audit_verdict(group_metrics(frame, 0.8)["by_group"], frame, 0.8)
    assert v["n_alerting_magnets"] == 2            # MRI-03 correctly silent
    assert v["selection_rate_ratio"] == 1.0
    assert v["selection_rate_difference"] == 0.0
    assert v["verdict"] == "pass"


def test_verdict_passes_on_equal_fleet():
    rng = np.random.default_rng(5)
    rows = []
    for magnet in ("MRI-01", "MRI-02", "MRI-03"):
        n = 300
        y = np.zeros(n, dtype=int)
        y[rng.choice(n, 8, replace=False)] = 1
        score = np.where(y == 1, rng.uniform(0.9, 1.0, n), rng.uniform(0, 0.3, n))
        rows.append(pd.DataFrame({"magnet_id": magnet, "y": y, "score": score}))
    frame = pd.concat(rows, ignore_index=True)
    v = audit_verdict(group_metrics(frame, 0.8)["by_group"], frame, 0.8)
    assert v["verdict"] == "pass"
    assert all(v["event_recall_by_magnet"][m] == 1.0
               for m in v["event_recall_by_magnet"])
    assert v["findings"] == ["no fairness policy breach detected"]


def test_pr_auc_skips_magnets_without_positives():
    frame = _frame()
    frame.loc[frame.magnet_id == "MRI-02", "y"] = 0
    aps = per_magnet_pr_auc(frame)
    assert aps["MRI-02"] is None
    assert aps["MRI-01"] is not None and 0.0 <= aps["MRI-01"] <= 1.0


def test_run_fairness_audit_writes_report(tmp_path):
    from quenchguard import fairness as F
    from quenchguard.config import REPORTS_DIR

    F.REPORTS_DIR = tmp_path
    try:
        audit = run_fairness_audit(frame=_frame(), threshold=0.8, verbose=False)
        assert audit["verdict"] in {"pass", "warn", "fail"}
        assert audit["n_magnets"] == 3
        saved = json.loads((tmp_path / "fairness.json").read_text(encoding="utf-8"))
        assert saved["threshold"] == 0.8
        assert any("magnet_id" in n for n in saved["notes"])
    finally:
        F.REPORTS_DIR = REPORTS_DIR


def test_run_fairness_audit_real_model():
    """Full audit with the champion LSTM on the real test split (if present)."""
    from quenchguard.config import default_config

    cfg = default_config()
    if not (cfg.model_path.exists() and cfg.features_path.exists()):
        pytest.skip("model/gold artifacts not built")
    audit = run_fairness_audit(cfg, verbose=False)
    assert audit["n_magnets"] == 6
    assert 0.0 <= audit["threshold"] <= 1.0
    # safety equity: every test event must be detected, per magnet
    recalls = [v for v in audit["event_recall_by_magnet"].values() if v is not None]
    assert recalls and min(recalls) >= EVENT_RECALL_REQUIREMENT
