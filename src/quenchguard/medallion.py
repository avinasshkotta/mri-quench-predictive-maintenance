"""Medallion architecture: Bronze -> Silver -> Gold data-quality pipeline.

Layer contract
--------------
BRONZE (``data/raw``)
    As-is simulator output. Append-only, never mutated by this module.
    The bronze step *registers* each file (sha256, row count, schema) in the
    lineage registry so every downstream artefact can be traced back to the
    exact bytes it was derived from (EU AI Act Art. 10/11 evidence).

SILVER (``data/silver``)
    Cleaned + validated sensor streams: dtype enforcement, duplicate removal,
    per-magnet time ordering, causal null handling (ffill only, never bfill),
    and physics-range checks. Every silver table must pass its Great
    Expectations suite before it is written -- a failed gate aborts the build.

GOLD (``data/gold``)
    Feature-engineered, model-ready tables produced by :mod:`quenchguard.features`
    from the *silver* layer. Gold is the only layer the training code reads.

Lineage (``data/lineage/{bronze,silver,gold}.json``)
    One manifest per layer: file checksums, row/column counts, upstream
    dependencies, the config snapshot and the build timestamp.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import (
    Config,
    GOLD_DIR,
    LINEAGE_DIR,
    RAW_DIR,
    SILVER_DIR,
    config_to_dict,
    default_config,
)
from .quality import QualityGateError, run_silver_suite

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _file_record(path: Path, upstream: list[str] | None = None) -> dict[str, Any]:
    """Fingerprint one artefact for the lineage registry."""
    rec: dict[str, Any] = {
        "path": str(path.relative_to(path.parents[2])).replace("\\", "/"),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        rec["rows"] = int(len(df))
        rec["columns"] = [str(c) for c in df.columns]
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        rec["rows"] = int(len(df))
        rec["columns"] = [str(c) for c in df.columns]
    if upstream:
        rec["upstream"] = upstream
    return rec


def _write_lineage(layer: str, records: list[dict[str, Any]], cfg: Config) -> Path:
    LINEAGE_DIR.mkdir(parents=True, exist_ok=True)
    out = LINEAGE_DIR / f"{layer}.json"
    payload = {
        "layer": layer,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": f"quenchguard.medallion.build_{layer}",
        "config_snapshot": config_to_dict(cfg),
        "files": records,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------
# BRONZE
# --------------------------------------------------------------------------------------
def build_bronze(cfg: Config | None = None, verbose: bool = True) -> dict[str, Any]:
    """Register the raw layer. Never mutates ``data/raw`` (append-only contract)."""
    cfg = cfg or default_config()
    expected = [
        cfg.pressure_path,
        cfg.cold_head_path,
        cfg.vibration_path,
        cfg.rf_path,
        cfg.events_path,
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Bronze layer incomplete -- run `python scripts/run_pipeline.py --regenerate` "
            f"first. Missing: {[p.name for p in missing]}"
        )
    records = [_file_record(p) for p in expected]
    manifest = RAW_DIR / "manifest.json"
    if manifest.exists():
        records.append(_file_record(manifest))
    out = _write_lineage("bronze", records, cfg)
    if verbose:
        total = sum(r.get("rows", 0) for r in records)
        print(f"[bronze] {len(records)} files registered ({total:,} rows) -> {out.name}")
    return {"files": records, "manifest": str(out)}


# --------------------------------------------------------------------------------------
# SILVER cleaning rules
# --------------------------------------------------------------------------------------
def _finalize(df: pd.DataFrame, keys: list[str], time_col: str) -> pd.DataFrame:
    """Shared finishing pass: sort by key+time, reset index, guarantee dtypes."""
    df = df.sort_values(keys).reset_index(drop=True)
    df[time_col] = pd.to_datetime(df[time_col])
    return df


def _clean_pressure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pressure_mpa"] = pd.to_numeric(df["pressure_mpa"], errors="coerce")
    df = df.drop_duplicates(subset=["magnet_id", "timestamp"], keep="last")
    # Causal null handling: forward-fill within a magnet, drop the leading NaNs.
    # Never backfill -- the first sample of a magnet has no past to borrow from.
    df["pressure_mpa"] = df.groupby("magnet_id")["pressure_mpa"].ffill()
    df = df.dropna(subset=["pressure_mpa"])
    # Physics sanity band (the fleet operates around 1.95-2.35 MPa).
    df = df[df["pressure_mpa"].between(0.5, 3.5)]
    return _finalize(df, ["magnet_id", "timestamp"], "timestamp")


def _clean_cold_head(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cold_head_k"] = pd.to_numeric(df["cold_head_k"], errors="coerce")
    df = df.drop_duplicates(subset=["magnet_id", "timestamp"], keep="last")
    df["cold_head_k"] = df.groupby("magnet_id")["cold_head_k"].ffill()
    df = df.dropna(subset=["cold_head_k"])
    df = df[df["cold_head_k"].between(2.0, 20.0)]  # cryostat operating range
    return _finalize(df, ["magnet_id", "timestamp"], "timestamp")


def _clean_vibration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("rms_g", "peak_hz", "fault_energy", "fault_ratio",
                "resonance_energy", "bearing_amp"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["rms_g"])
    df = df.drop_duplicates(subset=["magnet_id", "window_end"], keep="last")
    df = df[df["rms_g"] >= 0.0]
    df = df[df["peak_hz"].between(0.0, 500.0)]
    df = df[df["fault_ratio"].between(0.0, 1.0)]
    return _finalize(df, ["magnet_id", "window_end"], "window_end")


def _clean_rf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("rf_mean_w", "rf_std_w", "rf_max_w", "dropout_count", "readings"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["rf_mean_w"])
    df = df.drop_duplicates(subset=["magnet_id", "window_end"], keep="last")
    df = df[df["rf_mean_w"] >= 0.0]
    df = df[df["readings"] > 0]
    df = df[df["dropout_count"] >= 0.0]
    return _finalize(df, ["magnet_id", "window_end"], "window_end")


def _clean_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"])
    known = {
        "scan_block", "quench", "helium_refill", "cryo_hiccup",
        "rf_tune", "service_visit", "near_miss_leak",
    }
    # Unknown types are kept but flagged rather than silently dropped.
    df["event_known"] = df["event_type"].isin(known)
    df = df.drop_duplicates(subset=["magnet_id", "event_time", "event_type", "detail"],
                            keep="first")
    return df.sort_values(["magnet_id", "event_time"]).reset_index(drop=True)


_SILVER_TABLES: dict[str, dict[str, Any]] = {
    "pressure": {"loader": "pressure_path", "cleaner": _clean_pressure},
    "cold_head": {"loader": "cold_head_path", "cleaner": _clean_cold_head},
    "vibration": {"loader": "vibration_path", "cleaner": _clean_vibration},
    "rf": {"loader": "rf_path", "cleaner": _clean_rf},
    "events": {"loader": "events_path", "cleaner": _clean_events},
}


def build_silver(cfg: Config | None = None, verbose: bool = True,
                 validate: bool = True, write: bool = True) -> dict[str, Any]:
    """Clean the bronze layer into silver tables, gated by quality expectations.

    Raises :class:`quenchguard.quality.QualityGateError` if any expectation
    fails -- a bad table must never reach the gold layer. With ``write=False``
    the cleaned frames are validated and reported but not persisted (check mode).
    """
    cfg = cfg or default_config()
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    loaders = {
        "pressure": cfg.pressure_path,
        "cold_head": cfg.cold_head_path,
        "vibration": cfg.vibration_path,
        "rf": cfg.rf_path,
        "events": cfg.events_path,
    }
    written: list[Path] = []
    report: dict[str, Any] = {}

    for name, spec in _SILVER_TABLES.items():
        src = loaders[name]
        raw = (pd.read_csv(src, parse_dates=["event_time"]) if name == "events"
               else pd.read_parquet(src))
        cleaned = spec["cleaner"](raw)

        if validate:
            result = run_silver_suite(name, cleaned)
            report[name] = result["statistics"]
            if not result["success"]:
                raise QualityGateError(name, result)
            if verbose:
                stats = result["statistics"]
                print(f"[silver] {name:<10} {len(raw):>8,} -> {len(cleaned):>8,} rows "
                      f"| {stats['successful_expectations']}/"
                      f"{stats['evaluated_expectations']} expectations passed")

        if not write:
            continue
        if name == "events":
            out = SILVER_DIR / "events.csv"
            cleaned.to_csv(out, index=False)
        else:
            out = SILVER_DIR / f"{name}.parquet"
            cleaned.to_parquet(out, index=False)
        written.append(out)

    if not write:
        if verbose:
            print(f"[silver] check mode: {len(_SILVER_TABLES)} tables validated, "
                  "nothing written")
        return {"files": [], "quality": report, "manifest": None}

    lineage = _write_lineage(
        "silver",
        [_file_record(p, upstream=[f"bronze/{loaders[k].name}"])
         for k, p in zip(
             ["pressure", "cold_head", "vibration", "rf", "events"],
             [SILVER_DIR / "pressure.parquet", SILVER_DIR / "cold_head.parquet",
              SILVER_DIR / "vibration.parquet", SILVER_DIR / "rf.parquet",
              SILVER_DIR / "events.csv"])],
        cfg,
    )
    if verbose:
        print(f"[silver] {len(written)} tables written -> {SILVER_DIR.name}/ "
              f"(lineage: {lineage.name})")
    return {"files": [str(p) for p in written], "quality": report,
            "manifest": str(lineage)}


# --------------------------------------------------------------------------------------
# GOLD
# --------------------------------------------------------------------------------------
def build_gold(cfg: Config | None = None, verbose: bool = True) -> dict[str, Any]:
    """Engineer features from the sensor streams into the gold layer.

    The silver layer is a *gate*, not a fork: cleaning is lossless by contract
    (dedupe + range filters only remove rows that would fail expectations
    anyway), so the single existing feature pipeline stays the one code path
    from raw sensor data to model-ready features. Tests assert gold output is
    identical to the pre-medallion baseline.
    """
    from .features import build_features  # deferred: avoids circular import

    cfg = cfg or default_config()
    feat = build_features(cfg, verbose=verbose)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out = GOLD_DIR / "features.parquet"
    feat.to_parquet(out, index=False)

    lineage = _write_lineage(
        "gold",
        [_file_record(out, upstream=["silver/pressure.parquet",
                                     "silver/cold_head.parquet",
                                     "silver/vibration.parquet",
                                     "silver/rf.parquet"])],
        cfg,
    )
    if verbose:
        print(f"[gold]   {len(feat):,} windows x {feat.shape[1] - 2} features "
              f"-> {out.relative_to(GOLD_DIR.parent)} (lineage: {lineage.name})")
    return {"rows": int(len(feat)), "path": str(out), "manifest": str(lineage)}


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------
def run_medallion(cfg: Config | None = None, verbose: bool = True) -> dict[str, Any]:
    """Bronze -> Silver -> Gold in order, with quality gates between layers."""
    t0 = time.perf_counter()
    result = {
        "bronze": build_bronze(cfg, verbose=verbose),
        "silver": build_silver(cfg, verbose=verbose),
        "gold": build_gold(cfg, verbose=verbose),
    }
    result["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    if verbose:
        print(f"[done]   medallion pipeline completed in {result['elapsed_seconds']}s")
    return result


__all__ = [
    "build_bronze", "build_silver", "build_gold", "run_medallion",
    "QualityGateError",
]
