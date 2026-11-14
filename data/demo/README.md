# Demo bundle (committed, ~0.6 MB)

Fallback data for demos that cannot run the pipeline — e.g. Streamlit
Community Cloud, which has no GPU. **Rebuild with:**
`python scripts/build_demo_bundle.py` (requires the full medallion data from
`python scripts/run_pipeline.py`).

| File | Contents | Why |
|---|---|---|
| `features.parquet` | last 250 gold windows per magnet (1,500 rows × 38 cols) | agent tools need only the trailing 48-step sequence per magnet, plus a little history for trend reasoning |
| `pressure.parquet` | silver helium pressure, 1-hour mean, last 30 days | drilldown telemetry chart |
| `cold_head.parquet` | silver cold-head temperature, 1 h, last 30 days | drilldown telemetry chart |
| `vibration.parquet` | silver vibration windows, 1 h, last 30 days | drilldown telemetry chart |

`quenchguard.agent.data_file()` / `_gold()` prefer the real silver/gold
layers and fall back to this bundle, so local runs use full data and the
cloud demo degrades honestly rather than failing. Thresholds, features and
the champion model are identical — only the observation window is shortened.
