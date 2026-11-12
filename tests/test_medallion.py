"""Tests for the Bronze -> Silver -> Gold medallion pipeline and its quality gates."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.config import (GOLD_DIR, LINEAGE_DIR, RAW_DIR, SILVER_DIR,
                                default_config)
from quenchguard.medallion import (QualityGateError, _clean_pressure,
                                   build_bronze, build_gold, build_silver)
from quenchguard.quality import run_silver_suite

CFG = default_config()


# --------------------------------------------------------------------------------------
# silver cleaning rules
# --------------------------------------------------------------------------------------
def test_clean_pressure_deduplicates_and_drops_out_of_range():
    idx = pd.date_range("2025-01-01", periods=5, freq="1min")
    df = pd.DataFrame({
        "timestamp": list(idx) + [idx[2]],          # duplicate row
        "magnet_id": ["MRI-01"] * 6,
        "pressure_mpa": [2.3, 2.31, None, 2.29, 9.99, 2.30],
    })
    # 9.99 MPa is physically impossible; the duplicate at t2 resolves to its
    # *last* occurrence (2.30), which also discards the None row.
    out = _clean_pressure(df)

    assert len(out) == 4, "duplicate + out-of-range row must be removed"
    assert out["pressure_mpa"].notna().all()
    assert out["pressure_mpa"].between(0.5, 3.5).all()
    assert out["timestamp"].is_monotonic_increasing
    assert out["pressure_mpa"].iloc[2] == pytest.approx(2.30)  # keep="last" wins


def test_clean_pressure_forward_fills_internal_nulls():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="1min"),
        "magnet_id": ["MRI-01"] * 3,
        "pressure_mpa": [2.3, None, 2.31],
    })
    out = _clean_pressure(df)
    assert len(out) == 3
    assert out["pressure_mpa"].iloc[1] == pytest.approx(2.3)   # borrowed from the past
    assert out["pressure_mpa"].iloc[2] == pytest.approx(2.31)  # never altered


def test_clean_pressure_never_backfills():
    """The first sample of a magnet has no past: it must be dropped, not backfilled."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="1min"),
        "magnet_id": ["MRI-01"] * 3,
        "pressure_mpa": [None, 2.3, 2.31],
    })
    out = _clean_pressure(df)
    assert len(out) == 2
    assert out["pressure_mpa"].iloc[0] == pytest.approx(2.3)


# --------------------------------------------------------------------------------------
# quality gates
# --------------------------------------------------------------------------------------
def test_quality_gate_raises_on_corrupt_pressure():
    """Nulls + impossible values must fail the suite and raise QualityGateError."""
    bad = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=4, freq="1min"),
        "magnet_id": ["MRI-01", "MRI-99", "MRI-01", "MRI-01"],
        "pressure_mpa": [2.3, 2.31, None, 9.99],   # null + unknown magnet + 9.99 MPa
    })
    result = run_silver_suite("pressure", bad, CFG)
    assert result["success"] is False
    assert result["statistics"]["unsuccessful_expectations"] >= 3

    err = QualityGateError("pressure", result)
    assert "pressure" in str(err)
    assert "expectations failed" in str(err)


def test_quality_gate_passes_on_clean_pressure():
    good = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=2000, freq="1min"),
        "magnet_id": ["MRI-01"] * 2000,
        "pressure_mpa": [2.3] * 2000,
    })
    result = run_silver_suite("pressure", good, CFG)
    # row-count expectation is fleet-sized, so only the value expectations pass
    failed = [r for r in result["results"] if not r["success"]]
    assert all("row_count" in r["expectation"] for r in failed), \
        f"unexpected failures: {[r['expectation'] for r in failed]}"


def test_unknown_table_raises_keyerror():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(KeyError):
        run_silver_suite("nonexistent_table", df, CFG)


# --------------------------------------------------------------------------------------
# real-data integration (bronze must be present)
# --------------------------------------------------------------------------------------
def test_silver_cleaning_is_lossless_on_fleet_data():
    """The simulator's clean data must survive cleaning unchanged in row count.

    This is the invariant that makes gold reproducible: if cleaning ever drops
    a legitimate row, the feature frame (and every model metric) shifts.
    """
    pytest.importorskip("pyarrow")
    if not CFG.pressure_path.exists():
        pytest.skip("bronze data not generated")

    result = build_silver(cfg=CFG, verbose=False, validate=True, write=False)
    for table, stats in result["quality"].items():
        assert stats["unsuccessful_expectations"] == 0, f"{table}: {stats}"

    from quenchguard.medallion import _SILVER_TABLES, _clean_events
    frames = {
        "pressure": pd.read_parquet(CFG.pressure_path),
        "cold_head": pd.read_parquet(CFG.cold_head_path),
        "vibration": pd.read_parquet(CFG.vibration_path),
        "rf": pd.read_parquet(CFG.rf_path),
        "events": pd.read_csv(CFG.events_path, parse_dates=["event_time"]),
    }
    cleaners = {
        "pressure": _clean_pressure,
        "cold_head": _SILVER_TABLES["cold_head"]["cleaner"],
        "vibration": _SILVER_TABLES["vibration"]["cleaner"],
        "rf": _SILVER_TABLES["rf"]["cleaner"],
        "events": _clean_events,
    }
    for name, raw in frames.items():
        cleaned = cleaners[name](raw)
        assert len(cleaned) == len(raw), (
            f"{name}: cleaning changed row count {len(raw)} -> {len(cleaned)}"
        )


# --------------------------------------------------------------------------------------
# lineage
# --------------------------------------------------------------------------------------
def test_bronze_lineage_records_checksums(tmp_path):
    if not CFG.pressure_path.exists():
        pytest.skip("bronze data not generated")

    build_bronze(cfg=CFG, verbose=False)
    manifest_path = LINEAGE_DIR / "bronze.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["layer"] == "bronze"
    names = {f["path"] for f in payload["files"]}
    assert "data/raw/pressure_1min.parquet" in names

    for record in payload["files"]:
        path = Path(__file__).resolve().parents[1] / record["path"]
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["sha256"] == expected, f"checksum mismatch for {record['path']}"


def test_gold_layer_has_model_ready_schema():
    if not (GOLD_DIR / "features.parquet").exists():
        pytest.skip("gold features not built")

    from quenchguard.features import FEATURE_COLUMNS
    gold = pd.read_parquet(GOLD_DIR / "features.parquet")
    assert set(FEATURE_COLUMNS).issubset(gold.columns)
    assert gold["magnet_id"].nunique() == CFG.fleet.n_magnets
    # time ordering must hold within each magnet (sequences depend on it)
    ordered = gold.groupby("magnet_id")["window_end"].apply(
        lambda s: s.is_monotonic_increasing
    )
    assert ordered.all(), f"unsorted windows in: {list(ordered[~ordered].index)}"
    assert not gold[FEATURE_COLUMNS].isna().any().any(), "gold must be NaN-free"
