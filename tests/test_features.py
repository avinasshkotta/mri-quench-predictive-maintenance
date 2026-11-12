"""Tests for the causal feature-extraction helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.features import (FEATURE_COLUMNS, FEATURE_DOCS, FEATURE_GROUPS,
                                  _causal_daily_slope, _slope7, window_end_of)


def test_window_end_maps_minutes_into_their_interval():
    ts = pd.Series(pd.to_datetime([
        "2025-01-01 00:00", "2025-01-01 00:14", "2025-01-01 00:15",
        "2025-01-01 00:29", "2025-01-01 00:31",
    ]))
    ends = window_end_of(ts, 15)
    assert list(ends.dt.strftime("%H:%M")) == ["00:15", "00:15", "00:30", "00:30", "00:45"]


def test_window_end_never_precedes_the_minute_it_encloses():
    ts = pd.Series(pd.date_range("2025-01-01", periods=200, freq="1min"))
    ends = window_end_of(ts, 15)
    assert (ends > ts).all()
    assert ((ends - ts) <= pd.Timedelta(minutes=15)).all()


def test_slope7_recovers_a_known_line():
    x = np.arange(7, dtype=float)
    assert _slope7(2.0 * x + 5.0) == pytest.approx(2.0)
    assert np.isnan(_slope7(np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])))


def test_daily_slope_uses_only_complete_days():
    """The published slope must be the rolling slope shifted by one whole day."""
    idx = pd.date_range("2025-01-01", periods=30, freq="1D")
    noise = np.random.default_rng(0).normal(0, 0.01, 30)
    s = pd.Series(np.linspace(2.4, 2.0, 30) + noise, index=idx)

    raw = s.rolling(7, min_periods=3).apply(_slope7, raw=True)
    published = _causal_daily_slope(s)
    pd.testing.assert_series_equal(published, raw.shift(1), check_names=False)

    # invariance: extending the series must not change what was already published
    short = _causal_daily_slope(s.iloc[:20])
    pd.testing.assert_series_equal(published.iloc[3:20], short.iloc[3:20])


def test_feature_registry_is_consistent():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)), "duplicate feature names"
    grouped = [c for cols in FEATURE_GROUPS.values() for c in cols]
    assert grouped == FEATURE_COLUMNS
    documented = {row[0] for row in FEATURE_DOCS}
    assert documented == set(FEATURE_COLUMNS)
