"""Build the committed demo bundle (data/demo/) for cloud/no-data demos.

    python scripts/build_demo_bundle.py

The full medallion data (data/raw|silver|gold) is git-ignored and regenerated
by ``scripts/run_pipeline.py`` (~3 min on GPU). Streamlit Community Cloud has
no GPU and cannot run the pipeline, so the public demo needs a small,
committed fallback:

* ``features.parquet``  — last 250 gold windows per magnet (scoring needs only
  the trailing 48-step sequence; the bundle keeps a little history so the
  agent can still reason about trend)
* ``pressure.parquet`` / ``cold_head.parquet`` / ``vibration.parquet`` —
  silver telemetry downsampled to 1-hour resolution over the last 30 days

``quenchguard.agent._gold`` and ``sensor_summary`` prefer the real layers and
fall back to this bundle, so local runs use full data and the cloud demo
stays honest about what it serves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEMO = REPO_ROOT / "data" / "demo"
WINDOWS_PER_MAGNET = 250
TELEMETRY_HOURS = 30 * 24


def main() -> int:
    DEMO.mkdir(parents=True, exist_ok=True)

    gold = pd.read_parquet(REPO_ROOT / "data" / "gold" / "features.parquet")
    gold = gold.sort_values(["magnet_id", "window_end"])
    slim = (gold.groupby("magnet_id", group_keys=False)
                .tail(WINDOWS_PER_MAGNET).reset_index(drop=True))
    slim.to_parquet(DEMO / "features.parquet", index=False)
    print(f"features: {len(slim):,} rows "
          f"({slim.magnet_id.nunique()} magnets x {WINDOWS_PER_MAGNET} windows)")

    specs = [("pressure.parquet", "timestamp", "pressure_mpa"),
             ("cold_head.parquet", "timestamp", "cold_head_k"),
             ("vibration.parquet", "window_end", "rms_g")]
    for fname, ts, _val in specs:
        src = REPO_ROOT / "data" / "silver" / fname
        if not src.exists():
            print(f"{fname}: silver missing, skipped")
            continue
        df = pd.read_parquet(src)
        cut = df[ts].max() - pd.Timedelta(hours=TELEMETRY_HOURS)
        recent = df[df[ts] >= cut]
        # 1-hour downsample: mean over each clock hour, per magnet
        parts = []
        for magnet, part in recent.groupby("magnet_id"):
            hourly_part = (part.set_index(ts)
                               .resample("1h").mean(numeric_only=True)
                               .reset_index())
            hourly_part.insert(1, "magnet_id", magnet)
            parts.append(hourly_part)
        hourly = pd.concat(parts, ignore_index=True).sort_values(ts)
        hourly.to_parquet(DEMO / fname, index=False)
        print(f"{fname}: {len(hourly):,} rows (1 h, last {TELEMETRY_HOURS // 24} days)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
