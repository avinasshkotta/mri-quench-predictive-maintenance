"""Tests for time-ordered splits and sequence construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.config import default_config
from quenchguard.features import FEATURE_COLUMNS
from quenchguard.sequences import _build_sequences, add_split, split_boundaries

CFG = default_config()
SEQ = CFG.window.seq_len


def _frame(n_per_magnet: int = 120, magnets=("MRI-01", "MRI-02")) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2025-01-01")
    for m in magnets:
        idx = pd.date_range(start, periods=n_per_magnet, freq="15min")
        rows.append(pd.DataFrame({
            "magnet_id": m,
            "window_end": idx,
            "label": np.zeros(n_per_magnet, dtype=int),
        }))
    df = pd.concat(rows, ignore_index=True)
    for i, c in enumerate(FEATURE_COLUMNS):
        df[c] = (np.arange(len(df)) + i) % 7
    return df


def test_split_boundaries_are_global_timestamps():
    df = _frame()
    train_end, val_end = split_boundaries(CFG, df)
    assert df.window_end.min() < train_end < val_end < df.window_end.max()
    tagged = add_split(df, CFG)
    counts = tagged.split.value_counts().to_dict()
    assert set(counts) == {"train", "val", "test"}
    # splits must be contiguous in time
    assert tagged[tagged.split == "train"].window_end.max() < \
        tagged[tagged.split == "val"].window_end.min()
    assert tagged[tagged.split == "val"].window_end.max() < \
        tagged[tagged.split == "test"].window_end.min()


def test_sequences_have_the_right_shape_and_labels():
    df = _frame(n_per_magnet=100)
    df["split"] = "train"
    df.loc[90:, "label"] = 1
    df[[f"s__{c}" for c in FEATURE_COLUMNS]] = df[FEATURE_COLUMNS].to_numpy(np.float32)

    out = _build_sequences(df, CFG)
    assert out.X.shape == (2 * (100 - SEQ + 1), SEQ, len(FEATURE_COLUMNS))
    assert out.y.shape == (len(out.X),)
    # the sequence label belongs to the LAST window of the sequence
    assert out.y[-1] == 1 and out.y[0] == 0
    assert set(np.unique(out.magnets)) == {"MRI-01", "MRI-02"}


def test_sequences_never_cross_a_split_boundary():
    df = _frame(n_per_magnet=200)
    cut = df.loc[100, "window_end"]                    # a global time cut
    df["split"] = np.where(df.window_end <= cut, "train", "test")
    df[[f"s__{c}" for c in FEATURE_COLUMNS]] = df[FEATURE_COLUMNS].to_numpy(np.float32)

    train = _build_sequences(df[df.split == "train"], CFG)
    test = _build_sequences(df[df.split == "test"], CFG)
    assert train.times.max() <= cut
    assert test.times.min() > cut
    assert len(train) > 0 and len(test) > 0


def test_gaps_are_dropped():
    df = _frame(n_per_magnet=100)
    df = df[~((df.magnet_id == "MRI-01") & df.window_end.dt.hour.eq(5))]  # punch a hole
    df["split"] = "train"
    df[[f"s__{c}" for c in FEATURE_COLUMNS]] = df[FEATURE_COLUMNS].to_numpy(np.float32)
    out = _build_sequences(df, CFG)

    # independent expectation: count windows whose 48 stamps are contiguous
    times = np.sort(df.loc[df.magnet_id == "MRI-01", "window_end"].to_numpy())
    step = np.timedelta64(CFG.window.step_minutes, "m")
    valid = sum(
        1 for i in range(len(times) - SEQ + 1)
        if times[i + SEQ - 1] - times[i] == (SEQ - 1) * step
    )
    assert (out.magnets == "MRI-01").sum() == valid
    assert valid < len(times) - SEQ + 1, "the gap must have removed some windows"


def test_too_short_a_history_yields_no_sequences():
    df = _frame(n_per_magnet=SEQ - 5)
    df["split"] = "train"
    df[[f"s__{c}" for c in FEATURE_COLUMNS]] = df[FEATURE_COLUMNS].to_numpy(np.float32)
    out = _build_sequences(df, CFG)
    assert len(out) == 0
    assert out.X.shape == (0, SEQ, len(FEATURE_COLUMNS))
