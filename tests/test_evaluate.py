"""Tests for alert episodes, event matching, the operating-point policy and costs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.config import default_config
from quenchguard.evaluate import (alert_episodes, choose_threshold, classification_metrics,
                                  cost_summary, false_alerts, magnet_months,
                                  match_events, sweep_thresholds)

CFG = default_config()
H = pd.Timedelta(hours=CFG.window.horizon_hours)


def _scores(magnet: str, start: str, values: list[float]) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq="15min")
    return pd.DataFrame({"magnet_id": magnet, "t": idx,
                         "y": [int(v >= 0.9) for v in values],
                         "score": values, "split": "test"})


def test_episodes_collapse_runs_and_start_at_the_first_window():
    frame = _scores("MRI-01", "2025-03-01",
                    [0.1, 0.95, 0.99, 0.2, 0.97, 0.98, 0.99, 0.1])
    ep = alert_episodes(frame, 0.9)
    assert len(ep) == 2
    assert ep.alert_time.iloc[0] == pd.Timestamp("2025-03-01 00:15")
    assert ep.windows.tolist() == [2, 3]
    assert ep.last_time.iloc[1] == pd.Timestamp("2025-03-01 01:30")


def test_episodes_do_not_merge_across_magnets():
    a = _scores("MRI-01", "2025-03-01", [0.99] * 5)
    b = _scores("MRI-02", "2025-03-01", [0.99] * 5)
    frame = pd.concat([a, b], ignore_index=True)
    ep = alert_episodes(frame, 0.9)
    assert len(ep) == 2
    assert set(ep.magnet_id) == {"MRI-01", "MRI-02"}


def test_alert_before_the_horizon_is_clamped_not_dropped():
    """An episode starting 120 h early but still active at the horizon is credited."""
    ev = pd.DataFrame({"magnet_id": ["MRI-01"],
                       "event_time": [pd.Timestamp("2025-03-10 00:00")],
                       "event_type": ["quench"]})
    n = 96 * 5                       # continuous alarm covering 03-05 .. 03-10
    frame = _scores("MRI-01", "2025-03-05", [0.99] * n)
    ep = alert_episodes(frame, 0.9)
    out = match_events(ev, ep, CFG)
    assert bool(out.detected.iloc[0]) is True
    assert out.lead_hours.iloc[0] == pytest.approx(CFG.window.horizon_hours)
    assert out.lead_hours_unclamped.iloc[0] == pytest.approx(120.0, abs=0.3)


def test_an_alert_that_ends_before_the_horizon_is_not_credited():
    ev = pd.DataFrame({"magnet_id": ["MRI-01"],
                       "event_time": [pd.Timestamp("2025-03-10 00:00")],
                       "event_type": ["quench"]})
    frame = _scores("MRI-01", "2025-03-05", [0.99] * 10)   # stops 5 days early
    out = match_events(ev, alert_episodes(frame, 0.9), CFG)
    assert bool(out.detected.iloc[0]) is False


def test_no_alert_means_not_detected():
    ev = pd.DataFrame({"magnet_id": ["MRI-01"],
                       "event_time": [pd.Timestamp("2025-03-10")],
                       "event_type": ["quench"]})
    frame = _scores("MRI-01", "2025-03-05", [0.1] * 10)
    out = match_events(ev, alert_episodes(frame, 0.9), CFG)
    assert bool(out.detected.iloc[0]) is False
    assert bool(out.detected_in_time.iloc[0]) is False
    assert np.isnan(out.lead_hours.iloc[0])


def test_false_alert_classification():
    ev = pd.DataFrame({"magnet_id": ["MRI-01"],
                       "event_time": [pd.Timestamp("2025-03-10 00:00")],
                       "event_type": ["quench"]})
    true_ep = alert_episodes(_scores("MRI-01", "2025-03-09 12:00", [0.99] * 4), 0.9)
    false_ep = alert_episodes(_scores("MRI-01", "2025-03-01", [0.99] * 4), 0.9)
    eps = pd.concat([true_ep, false_ep], ignore_index=True)
    flags = false_alerts(eps, ev, CFG)
    assert flags.is_false.tolist() == [False, True]


def test_magnet_months_uses_observation_span():
    frame = _scores("MRI-01", "2025-03-01", [0.1] * (96 * 30))   # 30 days
    assert magnet_months(frame) == pytest.approx(30 / 30.4375, rel=1e-3)


def test_threshold_choice_respects_the_tightened_budget():
    """Synthesize a validation sweep where only high thresholds meet the budget."""
    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame({
        "magnet_id": rng.choice(["MRI-01", "MRI-02"], n),
        "t": pd.to_datetime("2025-03-01")
        + pd.to_timedelta(np.arange(n) // 2 * 15, unit="min"),
        "y": rng.binomial(1, 0.1, n),
        "score": rng.random(n),
        "split": "val",
    })
    sweep = sweep_thresholds(frame, pd.DataFrame(columns=["magnet_id", "event_time",
                                                          "event_type"]), CFG)
    choice = choose_threshold(sweep, CFG)
    assert 0.0 < choice["threshold"] <= 1.0
    feasible = sweep[sweep.false_per_magnet_month <= choice["budget"]]
    assert len(feasible) > 0, "the sweep must contain at least one affordable point"
    chosen = sweep[sweep.threshold == choice["threshold"]].iloc[0]
    assert chosen.false_per_magnet_month <= choice["budget"] + 1e-9
    assert chosen.false_per_magnet_month <= sweep.false_per_magnet_month.max()


def test_cost_summary_arithmetic():
    events = pd.DataFrame({
        "magnet_id": ["MRI-01", "MRI-02", "MRI-03"],
        "event_time": pd.to_datetime(["2025-03-10", "2025-03-11", "2025-03-12"]),
        "lead_hours": [90.0, 90.0, np.nan],
        "detected_in_time": [True, True, False],
    })
    out = cost_summary(events, n_false=2, cfg=CFG)
    c = CFG.cost
    assert out["reactive_total_usd"] == 3 * c.emergency_total
    assert out["predictive_total_usd"] == (1 * c.emergency_total
                                           + 2 * c.planned_total
                                           + 2 * c.false_alarm_usd)
    assert out["net_savings_usd"] == out["reactive_total_usd"] - out["predictive_total_usd"]
    assert out["net_savings_usd"] > 0


def test_classification_metrics_contains_the_business_numbers():
    y = np.array([0, 0, 1, 1, 1, 0])
    s = np.array([0.1, 0.8, 0.9, 0.95, 0.2, 0.05])
    m = classification_metrics(y, s, threshold=0.85)
    assert m["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 1, "tp": 2}
    assert 0 <= m["pr_auc"] <= 1 and 0 <= m["roc_auc"] <= 1
