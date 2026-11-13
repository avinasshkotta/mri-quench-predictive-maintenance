"""Tests for C-MAPSS external validation (external_validation.py).

Hermetic cases use tiny synthetic C-MAPSS-shaped frames/files; the real-data
test runs only when the NASA FD001 files exist locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.external_validation import (CMAPSS_DIR, HORIZON_CYCLES,
                                             INFORMATIVE_SENSORS,
                                             MIN_LEAD_CYCLES, _COLUMNS,
                                             add_rul, alert_episodes_cycles,
                                             build_features, evaluate_cycles,
                                             false_alerts_cycles,
                                             feature_columns, load_cmapss,
                                             match_events_cycles,
                                             run_external_validation)

SENSORS = _COLUMNS[5:]


def _raw_frame(units: int = 3, cycles: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic C-MAPSS frame: a clear late-life degradation trend."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, units + 1):
        for t in range(1, cycles + 1):
            wear = max(0.0, (t - (cycles - 25)) / 25.0)      # ramps up at the end
            row = {"unit": unit, "cycle": t, "set1": 0.0, "set2": 0.0, "set3": 100.0}
            for i, s in enumerate(SENSORS):
                base = 500.0 + i
                row[s] = base + wear * (2.0 + 0.1 * i) + rng.normal(0, 0.02)
            rows.append(row)
    return pd.DataFrame(rows, columns=_COLUMNS)


def _events(frame: pd.DataFrame, horizon: int = HORIZON_CYCLES) -> pd.DataFrame:
    last = frame.groupby("unit")["cycle"].max()
    return pd.DataFrame({
        "unit": last.index.astype(int),
        "event_cycle": last.to_numpy(int),
        "last_observed_cycle": last.to_numpy(int),
        "rul_at_truncation": 0,
        "observable": True,
        "evaluable_in_time": True,
    })


# --------------------------------------------------------------------------------------
# data plumbing
# --------------------------------------------------------------------------------------
def test_add_rul_train_ends_at_zero():
    df = add_rul(_raw_frame(units=2, cycles=30))
    assert df.groupby("unit")["rul"].min().eq(0).all()
    assert df.groupby("unit")["rul"].max().eq(29).all()


def test_add_rul_test_reconstruction_is_exact():
    df = _raw_frame(units=2, cycles=10)[["unit", "cycle"] + SENSORS].copy()
    out = add_rul(df, pd.Series([7.0, 20.0]))
    last = out.groupby("unit").cycle.max()
    at_end = out[out.cycle == out.unit.map(last)].set_index("unit")["rul"]
    assert at_end.loc[1] == 7.0 and at_end.loc[2] == 20.0
    first = out[out.cycle == 1].set_index("unit")["rul"]
    assert first.loc[1] == 16.0 and first.loc[2] == 29.0


def test_missing_files_raise_with_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match="validate_cmapss.py --download"):
        load_cmapss(base=tmp_path)


# --------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------
def test_features_are_causal():
    """Truncating the future must not change any past feature value."""
    full = build_features(_raw_frame(units=2, cycles=40))
    cut_raw = _raw_frame(units=2, cycles=40)
    cut = build_features(cut_raw[cut_raw.cycle <= 20])
    cols = feature_columns(full)
    a = full[full.cycle <= 20].reset_index(drop=True)
    b = cut[cut.cycle <= 20].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["unit", "cycle"] + cols],
                                  b[["unit", "cycle"] + cols])


def test_feature_columns_exclude_identity_and_target():
    feat = build_features(_raw_frame())
    cols = feature_columns(feat)
    assert len(cols) == 4 * len(INFORMATIVE_SENSORS)
    for forbidden in ("unit", "cycle", "rul", "y", "set1", "set2", "set3"):
        assert forbidden not in cols
    assert not any(c in cols for c in ("s1", "s5", "s10", "s16", "s18", "s19"))


# --------------------------------------------------------------------------------------
# cycle-domain evaluation (mirrors evaluate.match_events)
# --------------------------------------------------------------------------------------
def test_alert_episodes_collapse_consecutive_cycles():
    frame = pd.DataFrame({"unit": [1] * 6, "cycle": [1, 2, 3, 4, 5, 6],
                          "score": [0.0, 0.9, 0.9, 0.1, 0.9, 0.9]})
    ep = alert_episodes_cycles(frame, 0.5)
    assert list(zip(ep.alert_cycle, ep.last_cycle)) == [(2, 3), (5, 6)]


def test_episodes_do_not_leak_across_units():
    frame = pd.DataFrame({"unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2],
                          "score": [0.9, 0.9, 0.9, 0.9]})
    ep = alert_episodes_cycles(frame, 0.5)
    assert len(ep) == 2 and set(ep.unit) == {1, 2}


def test_match_events_clamps_lead_to_horizon():
    episodes = pd.DataFrame({"unit": [1], "alert_cycle": [10], "last_cycle": [80]})
    events = pd.DataFrame({"unit": [1], "event_cycle": [100],
                           "last_observed_cycle": [100], "rul_at_truncation": [0],
                           "observable": [True], "evaluable_in_time": [True]})
    out = match_events_cycles(events, episodes, horizon=30, min_lead=15).iloc[0]
    assert out.detected and out.detected_in_time
    assert out.lead_cycles == 30               # clamped, never > horizon
    assert out.lead_cycles_unclamped == 90


def test_episode_outside_horizon_is_not_detected():
    episodes = pd.DataFrame({"unit": [1], "alert_cycle": [10], "last_cycle": [60]})
    events = pd.DataFrame({"unit": [1], "event_cycle": [100],
                           "last_observed_cycle": [100], "rul_at_truncation": [0],
                           "observable": [True], "evaluable_in_time": [True]})
    out = match_events_cycles(events, episodes, horizon=30, min_lead=15).iloc[0]
    assert not out.detected and not out.detected_in_time


def test_censored_unit_alert_is_false():
    """RUL at truncation > horizon: failure is beyond the observed data."""
    episodes = pd.DataFrame({"unit": [1], "alert_cycle": [10], "last_cycle": [20]})
    events = pd.DataFrame({"unit": [1], "event_cycle": [140],
                           "last_observed_cycle": [100], "rul_at_truncation": [40],
                           "observable": [False], "evaluable_in_time": [False]})
    false = false_alerts_cycles(episodes, events, horizon=30)
    assert bool(false.is_false.iloc[0])


def test_evaluate_cycles_censoring_denominators():
    frame = pd.DataFrame({"unit": [1] * 10 + [2] * 10,
                          "cycle": list(range(1, 11)) * 2,
                          "y": [0] * 20,
                          "score": [0.9] * 20})
    events = pd.DataFrame({
        "unit": [1, 2], "event_cycle": [40, 120],
        "last_observed_cycle": [10, 10], "rul_at_truncation": [30, 110],
        "observable": [True, False], "evaluable_in_time": [True, False]})
    out = evaluate_cycles(frame, events, threshold=0.5)
    assert out["n_observable_events"] == 1
    assert out["n_evaluable_in_time"] == 1
    assert out["events"]["event_recall"] == 1.0     # unit 1 alerted inside horizon
    assert out["alerts"]["false_alerts"] == 1       # unit 2 is censored -> false


# --------------------------------------------------------------------------------------
# end-to-end on synthetic files
# --------------------------------------------------------------------------------------
def _write_synthetic(base: Path, n_units: int = 12, cycles: int = 150) -> None:
    base.mkdir(parents=True, exist_ok=True)
    full = _raw_frame(units=n_units, cycles=cycles, seed=1)
    full.to_csv(base / "train_FD001.txt", sep=" ", header=False, index=False)
    # test: truncate each unit so its true RUL at truncation is the listed value
    ruls = [10, 12, 18, 22, 28, 35, 40, 55, 70, 90, 110, 130]
    parts = []
    for unit in range(1, n_units + 1):
        rul = ruls[(unit - 1) % len(ruls)]
        keep = cycles - rul
        parts.append(full[(full.unit == unit) & (full.cycle <= keep)])
    test = pd.concat(parts, ignore_index=True)
    test.to_csv(base / "test_FD001.txt", sep=" ", header=False, index=False)
    pd.Series(ruls[:n_units]).to_csv(base / "RUL_FD001.txt", sep=" ",
                                     header=False, index=False)


def test_run_external_validation_end_to_end_synthetic(tmp_path):
    base = tmp_path / "cmapss"
    _write_synthetic(base)
    report = run_external_validation(base=base, write_reports=False,
                                     n_train_units=8, verbose=False)
    assert report["dataset"]["train_units"] == 12
    assert report["dataset"]["test_units"] == 12
    assert report["protocol"]["horizon_cycles"] == HORIZON_CYCLES
    assert report["champion"] in report["results"]
    for name, r in report["results"].items():
        assert 0.0 <= r["threshold"] <= 1.0
        assert "pr_auc" in r["test"]["window"]
        assert "event_recall_in_time" in r["test"]["events"]


def test_split_rejects_too_many_train_units(tmp_path):
    base = tmp_path / "cmapss"
    _write_synthetic(base, n_units=6)
    with pytest.raises(ValueError, match="n_train_units"):
        run_external_validation(base=base, write_reports=False,
                                n_train_units=6, verbose=False)


# --------------------------------------------------------------------------------------
# real NASA data (integration; skipped when files are absent)
# --------------------------------------------------------------------------------------
def test_real_cmapss_shapes():
    if not (CMAPSS_DIR / "train_FD001.txt").exists():
        pytest.skip("C-MAPSS FD001 not downloaded")
    train, test, rul = load_cmapss()
    assert train.shape == (20631, 26)
    assert test.shape[1] == 26
    assert train.unit.nunique() == 100 and test.unit.nunique() == 100
    assert len(rul) == 100 and rul.min() == 7 and rul.max() == 145


def test_real_validation_reproduces_headline():
    if not (CMAPSS_DIR / "train_FD001.txt").exists():
        pytest.skip("C-MAPSS FD001 not downloaded")
    report = run_external_validation(write_reports=False, verbose=False)
    champ = report["results"][report["champion"]]["test"]
    assert champ["window"]["pr_auc"] > 0.80
    assert champ["events"]["event_recall"] >= 0.60
    assert champ["events"]["event_recall_in_time"] >= 0.50
    assert champ["alerts"]["false_per_unit"] <= 1.0
