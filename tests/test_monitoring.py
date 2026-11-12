"""Tests for drift monitoring (monitoring.py).

Core cases run on synthetic frames (hermetic, CI-safe). A real-data
integration test runs when gold features exist locally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.monitoring import (PSI_ALERT, SHARE_ALERT, drift_verdict,
                                    key_features, run_drift_check)

FEATURES = ["f_a", "f_b", "f_c", "f_d"]


def _frames(seed: int = 0, shift: dict[str, float] | None = None,
            n: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ref = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    cur = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    for col, offset in (shift or {}).items():
        cur[col] = cur[col] + offset
    return ref, cur


@pytest.fixture(scope="module")
def psi_stable() -> dict[str, float]:
    ref, cur = _frames(seed=1)
    from quenchguard.monitoring import _drift_snapshot
    psi, count, share = _drift_snapshot(ref, cur, FEATURES)
    assert 0.0 <= share <= 1.0
    return psi


@pytest.fixture(scope="module")
def psi_shifted() -> dict[str, float]:
    ref, cur = _frames(seed=1, shift={"f_a": 1.5})
    from quenchguard.monitoring import _drift_snapshot
    psi, count, share = _drift_snapshot(ref, cur, FEATURES)
    return psi


def test_psi_flags_the_shifted_feature(psi_shifted):
    assert psi_shifted["f_a"] > PSI_ALERT, psi_shifted
    # shifted feature must dominate every stable one
    for col in ("f_b", "f_c", "f_d"):
        assert psi_shifted["f_a"] > psi_shifted[col] * 3


def test_stable_features_have_low_psi(psi_stable):
    assert max(psi_stable.values()) < PSI_ALERT, psi_stable


def test_verdict_retrain_on_key_feature_drift(psi_shifted):
    v = drift_verdict(psi_shifted, keys=["f_a"], drifted_share=0.1)
    assert v["retrain_recommended"]
    assert v["key_drifted"] == ["f_a"]
    assert "f_a" in v["reason"]
    # psi table sorted by severity
    names = list(v["psi"])
    mags = [v["psi"][n] for n in names]
    assert mags == sorted(mags, reverse=True)


def test_verdict_stable_when_nothing_drifts(psi_stable):
    v = drift_verdict(psi_stable, keys=["f_a"], drifted_share=0.0)
    assert not v["retrain_recommended"]
    assert v["reason"] == "within thresholds"


def test_verdict_fleet_wide_share_breach(psi_stable):
    # even without key-feature drift, >25 % share trips the trigger
    v = drift_verdict(psi_stable, keys=["f_x"], drifted_share=SHARE_ALERT + 0.01)
    assert v["retrain_recommended"]
    assert "drifted share" in v["reason"]


def test_drift_snapshot_rejects_bad_input():
    from quenchguard.monitoring import _drift_snapshot

    ref, cur = _frames()
    with pytest.raises(ValueError, match="missing"):
        _drift_snapshot(ref, cur, FEATURES + ["ghost"])
    ref2, cur2 = _frames()
    ref2["f_a"] = ref2["f_a"].astype(str)
    with pytest.raises(ValueError, match="numeric"):
        _drift_snapshot(ref2, cur2, FEATURES)


def test_key_features_reads_metrics_json(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "shap": {"global_importance": [
            {"feature": f"f{i}", "mean_abs_shap": v}
            for i, v in enumerate([0.5, 0.9, 0.1, 0.7])]}}), encoding="utf-8")
    keys = key_features(path)
    assert keys == ["f1", "f3", "f0", "f2"]          # sorted by importance
    assert key_features(tmp_path / "nope.json") == []


def test_run_drift_check_writes_reports(tmp_path):
    ref, cur = _frames(seed=2, shift={"f_c": 2.0})
    verdict = run_drift_check(reference=ref, current=cur, write_reports=False,
                              verbose=False)
    assert isinstance(verdict["retrain_recommended"], bool)
    assert set(verdict["psi"]) == set(FEATURES)

    from quenchguard import monitoring as M

    M.REPORTS_DIR = tmp_path
    try:
        run_drift_check(reference=ref, current=cur, write_reports=True,
                        verbose=False)
        assert (tmp_path / "drift.json").exists()
        html = tmp_path / "drift_report.html"
        assert html.exists() and html.stat().st_size > 1000
        saved = json.loads((tmp_path / "drift.json").read_text(encoding="utf-8"))
        assert saved["psi"]["f_c"] > PSI_ALERT
    finally:
        from quenchguard.config import REPORTS_DIR

        M.REPORTS_DIR = REPORTS_DIR


def test_run_drift_check_full_pipeline_gold():
    """Integration: train vs test slice of the real gold frame (if present)."""
    from quenchguard.config import default_config
    from quenchguard.features import FEATURE_COLUMNS

    gold = default_config().features_path
    if not gold.exists():
        pytest.skip("gold features not built (run scripts/run_pipeline.py)")
    labelled = pd.read_parquet(gold)
    if "split" not in labelled.columns:
        pytest.skip("gold frame has no split column")
    ref = labelled.loc[labelled.split == "train", FEATURE_COLUMNS]
    cur = labelled.loc[labelled.split == "test", FEATURE_COLUMNS]
    verdict = run_drift_check(reference=ref, current=cur, verbose=False)
    assert set(verdict["psi"]) == set(FEATURE_COLUMNS)
    # train and test come from the same distribution -> stable verdict
    assert len(verdict["key_drifted"]) == 0 or not verdict["retrain_recommended"]
