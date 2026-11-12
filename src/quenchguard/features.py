"""Causal (leakage-free) feature extraction — the engineering table from the case study.

+--------------------+--------------------------+-----------------------------------------------+---------------------------+
| Raw sensor         | Extraction               | Engineering features                          | Failure signal            |
+--------------------+--------------------------+-----------------------------------------------+---------------------------+
| Helium pressure    | 1-min time-series        | rolling mean (5/30 min), delta, 7-day slope   | drift below 2.0 MPa       |
| Cold head temp     | thermocouple             | deviation from 4.2 K, spike count, rolling std| rise above 5.5 K          |
| Gradient vibration | accelerometer (1 kHz)    | RMS, FFT peak, 100-200 Hz fault-band energy   | 120 Hz bearing resonance  |
| RF power           | power meter (100 ms)     | rolling max, dropout count, stability ratio   | intermittent dropouts     |
+--------------------+--------------------------+-----------------------------------------------+---------------------------+

Causality rules enforced here
-----------------------------
1. Every rolling statistic looks *backwards* only (pandas ``rolling`` default).
2. ``diff`` uses past minutes only.
3. The 7-day trend slope is fit on **complete calendar days** (today's partial mean is
   never used), then joined with a 1-day lag, so no future minutes leak into a feature.
4. 1-minute signals are reduced to 15-minute windows with ``floor(step) + step``,
   i.e. window ``(t - 15 min, t]`` — every minute inside a window precedes its label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, GOLD_DIR, default_config
from .data_gen import load_cold_head, load_pressure, load_rf, load_vibration

# --------------------------------------------------------------------------------------
# feature registry (order = model input order)
# --------------------------------------------------------------------------------------
FEATURE_GROUPS: dict[str, list[str]] = {
    "helium_pressure": [
        "p_mean", "p_last", "p_std", "p_roll5_mean", "p_roll30_mean",
        "p_delta", "p_delta_max", "p_below_crit_count", "p_noise_std", "p_trend_7d",
    ],
    "cold_head_temp": [
        "t_mean", "t_last", "t_max", "t_std", "t_dev", "t_spike_count",
        "t_roll_std", "t_above_crit_count", "t_delta", "t_trend_7d",
    ],
    "vibration": [
        "v_rms", "v_rms_rollmax_1h", "v_rms_rollstd_1h", "v_peak_hz",
        "v_resonance", "v_fault_energy", "v_fault_ratio", "v_fault_rollmean_1h",
    ],
    "rf_power": [
        "rf_mean_w", "rf_std_w", "rf_max_w", "rf_stability",
        "dropout_count", "dropout_rate", "rf_rollmax_1h", "rf_dropouts_roll_1h",
    ],
}

FEATURE_COLUMNS: list[str] = [c for cols in FEATURE_GROUPS.values() for c in cols]

FEATURE_DOCS = [
    ("p_mean", "helium_pressure", "Mean helium pressure over the 15-min window (MPa)."),
    ("p_last", "helium_pressure", "Pressure at window close — the freshest reading."),
    ("p_std", "helium_pressure", "Within-window pressure volatility."),
    ("p_roll5_mean", "helium_pressure", "Mean of the causal 5-min rolling average."),
    ("p_roll30_mean", "helium_pressure", "Mean of the causal 30-min rolling average."),
    ("p_delta", "helium_pressure", "Mean 1-minute pressure change (leak rate proxy)."),
    ("p_delta_max", "helium_pressure", "Largest 1-minute drop inside the window."),
    ("p_below_crit_count", "helium_pressure", "Minutes below the 2.0 MPa failure signal."),
    ("p_noise_std", "helium_pressure", "Causal 60-min rolling std (sensor noise level)."),
    ("p_trend_7d", "helium_pressure", "7-day trend slope in MPa/day (complete days only)."),
    ("t_mean", "cold_head_temp", "Mean cold head temperature over the window (K)."),
    ("t_last", "cold_head_temp", "Temperature at window close."),
    ("t_max", "cold_head_temp", "Peak temperature inside the window."),
    ("t_std", "cold_head_temp", "Within-window temperature spread."),
    ("t_dev", "cold_head_temp", "Deviation from the 4.2 K design baseline."),
    ("t_spike_count", "cold_head_temp", "Minutes deviating more than 0.5 K from baseline."),
    ("t_roll_std", "cold_head_temp", "Mean causal 60-min rolling std (thermal noise)."),
    ("t_above_crit_count", "cold_head_temp", "Minutes above the 5.5 K failure signal."),
    ("t_delta", "cold_head_temp", "Mean 1-minute warming rate (K/min)."),
    ("t_trend_7d", "cold_head_temp", "7-day trend slope in K/day (complete days only)."),
    ("v_rms", "vibration", "RMS acceleration of the 1 kHz capture (g)."),
    ("v_rms_rollmax_1h", "vibration", "Max RMS over the last 4 windows (1 h)."),
    ("v_rms_rollstd_1h", "vibration", "Std of RMS over the last 4 windows."),
    ("v_peak_hz", "vibration", "Dominant FFT peak frequency (Hz)."),
    ("v_resonance", "vibration", "Energy inside 115-125 Hz — the 120 Hz bearing line."),
    ("v_fault_energy", "vibration", "Energy inside the 100-200 Hz fault band."),
    ("v_fault_ratio", "vibration", "Fault-band energy / total energy."),
    ("v_fault_rollmean_1h", "vibration", "1 h causal mean of fault-band energy."),
    ("rf_mean_w", "rf_power", "Mean RF forward power in the window (W)."),
    ("rf_std_w", "rf_power", "Std of RF forward power."),
    ("rf_max_w", "rf_power", "Rolling max power across the window."),
    ("rf_stability", "rf_power", "Power stability ratio = std / mean."),
    ("dropout_count", "rf_power", "Readings below 50 % of nominal inside the window."),
    ("dropout_rate", "rf_power", "dropout_count / readings."),
    ("rf_rollmax_1h", "rf_power", "Max power over the last 4 windows."),
    ("rf_dropouts_roll_1h", "rf_power", "Dropout count summed over the last 4 windows."),
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def window_end_of(timestamps: pd.Series, step_minutes: int) -> pd.Series:
    """Map a minute stamp to the end of the window it belongs to: (t - step, t]."""
    step = pd.Timedelta(minutes=step_minutes)
    return timestamps.dt.floor(step) + step


def _slope7(values: np.ndarray) -> float:
    """Least-squares slope (per day) of up to 7 daily means; NaN if < 3 points."""
    v = values[~np.isnan(values)]
    if v.size < 3:
        return np.nan
    x = np.arange(v.size, dtype=float)
    x -= x.mean()
    v = v - v.mean()
    denom = float((x * x).sum())
    return float((x * v).sum() / denom) if denom else np.nan


def _causal_daily_slope(s: pd.Series) -> pd.Series:
    """7-day trend on daily means, shifted by 1 day so only complete days are used."""
    daily = s.groupby(pd.Grouper(freq="1D")).mean()
    slope = daily.rolling(7, min_periods=3).apply(_slope7, raw=True)
    return slope.shift(1)          # <- strict causality: drop the in-progress day


def _rolling(s: pd.Series, window: int, kind: str = "mean", min_periods: int = 1) -> pd.Series:
    r = s.rolling(window, min_periods=min_periods)
    return {"mean": r.mean, "std": r.std, "max": r.max, "sum": r.sum}[kind]()


# --------------------------------------------------------------------------------------
# per-sensor builders
# --------------------------------------------------------------------------------------
def _pressure_features(cfg: Config) -> pd.DataFrame:
    ph, step = cfg.physics, cfg.window.step_minutes
    df = load_pressure(cfg).sort_values(["magnet_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("magnet_id")["pressure_mpa"]

    df["p_delta"] = g.diff().fillna(0.0)
    df["p_roll5"] = g.transform(lambda s: _rolling(s, 5))
    df["p_roll30"] = g.transform(lambda s: _rolling(s, 30))
    df["p_noise"] = g.transform(lambda s: _rolling(s, 60, "std"))
    df["p_below"] = (df["pressure_mpa"] < ph.pressure_critical_mpa).astype(float)
    df["window_end"] = window_end_of(df["timestamp"], step)

    # 7-day trend per magnet on complete days only (causal), joined with a 1-day lag
    daily = df.set_index("timestamp").groupby("magnet_id")["pressure_mpa"]
    trend = daily.apply(_causal_daily_slope).rename("p_trend_7d").reset_index()
    trend["timestamp"] = trend["timestamp"] + pd.Timedelta(days=1)   # only complete days
    trend = trend.rename(columns={"timestamp": "_day"})
    df["_day"] = df["timestamp"].dt.normalize()
    df = df.merge(trend, on=["magnet_id", "_day"], how="left")
    df = df.drop(columns=["_day"])

    out = (df.groupby(["magnet_id", "window_end"])
             .agg(p_mean=("pressure_mpa", "mean"),
                  p_last=("pressure_mpa", "last"),
                  p_std=("pressure_mpa", "std"),
                  p_roll5_mean=("p_roll5", "mean"),
                  p_roll30_mean=("p_roll30", "mean"),
                  p_delta=("p_delta", "mean"),
                  p_delta_max=("p_delta", "min"),
                  p_below_crit_count=("p_below", "sum"),
                  p_noise_std=("p_noise", "mean"),
                  p_trend_7d=("p_trend_7d", "last"))
             .reset_index())
    out["p_std"] = out["p_std"].fillna(0.0)
    out["p_trend_7d"] = out.groupby("magnet_id")["p_trend_7d"].ffill().fillna(0.0)
    return out


def _cold_head_features(cfg: Config) -> pd.DataFrame:
    ph, step = cfg.physics, cfg.window.step_minutes
    df = load_cold_head(cfg).sort_values(["magnet_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("magnet_id")["cold_head_k"]

    df["t_delta"] = g.diff().fillna(0.0)
    df["t_roll"] = g.transform(lambda s: _rolling(s, 60, "std"))
    dev = df["cold_head_k"] - ph.cold_head_baseline_k
    df["t_spike"] = (dev > 0.5).astype(float)
    df["t_crit"] = (df["cold_head_k"] > ph.cold_head_critical_k).astype(float)
    df["window_end"] = window_end_of(df["timestamp"], step)

    daily = df.set_index("timestamp").groupby("magnet_id")["cold_head_k"]
    trend = daily.apply(_causal_daily_slope).rename("t_trend_7d").reset_index()
    trend["timestamp"] = trend["timestamp"] + pd.Timedelta(days=1)   # only complete days
    trend = trend.rename(columns={"timestamp": "_day"})
    df["_day"] = df["timestamp"].dt.normalize()
    df = df.merge(trend, on=["magnet_id", "_day"], how="left")
    df = df.drop(columns=["_day"])

    out = (df.groupby(["magnet_id", "window_end"])
             .agg(t_mean=("cold_head_k", "mean"),
                  t_last=("cold_head_k", "last"),
                  t_max=("cold_head_k", "max"),
                  t_std=("cold_head_k", "std"),
                  t_spike_count=("t_spike", "sum"),
                  t_roll_std=("t_roll", "mean"),
                  t_above_crit_count=("t_crit", "sum"),
                  t_delta=("t_delta", "mean"),
                  t_trend_7d=("t_trend_7d", "last"))
             .reset_index())
    out["t_dev"] = out["t_mean"] - ph.cold_head_baseline_k
    out["t_std"] = out["t_std"].fillna(0.0)
    out["t_trend_7d"] = out.groupby("magnet_id")["t_trend_7d"].ffill().fillna(0.0)
    return out


def _vibration_features(cfg: Config) -> pd.DataFrame:
    df = load_vibration(cfg).sort_values(["magnet_id", "window_end"]).reset_index(drop=True)
    g = df.groupby("magnet_id")
    df["v_rms_rollmax_1h"] = g["rms_g"].transform(lambda s: _rolling(s, 4, "max"))
    df["v_rms_rollstd_1h"] = g["rms_g"].transform(lambda s: _rolling(s, 4, "std"))
    df["v_fault_rollmean_1h"] = g["fault_energy"].transform(lambda s: _rolling(s, 4, "mean"))
    out = pd.DataFrame({
        "magnet_id": df["magnet_id"],
        "window_end": df["window_end"],
        "v_rms": df["rms_g"],
        "v_peak_hz": df["peak_hz"],
        "v_resonance": df["resonance_energy"],
        "v_fault_energy": df["fault_energy"],
        "v_fault_ratio": df["fault_ratio"],
        "v_rms_rollmax_1h": df["v_rms_rollmax_1h"],
        "v_rms_rollstd_1h": df["v_rms_rollstd_1h"].fillna(0.0),
        "v_fault_rollmean_1h": df["v_fault_rollmean_1h"],
    })
    return out


def _rf_features(cfg: Config) -> pd.DataFrame:
    df = load_rf(cfg).sort_values(["magnet_id", "window_end"]).reset_index(drop=True)
    g = df.groupby("magnet_id")
    df["rf_stability"] = df["rf_std_w"] / df["rf_mean_w"].replace(0, np.nan)
    df["dropout_rate"] = df["dropout_count"] / df["readings"]
    df["rf_rollmax_1h"] = g["rf_max_w"].transform(lambda s: _rolling(s, 4, "max"))
    df["rf_dropouts_roll_1h"] = g["dropout_count"].transform(lambda s: _rolling(s, 4, "sum"))
    return pd.DataFrame({
        "magnet_id": df["magnet_id"],
        "window_end": df["window_end"],
        "rf_mean_w": df["rf_mean_w"],
        "rf_std_w": df["rf_std_w"],
        "rf_max_w": df["rf_max_w"],
        "rf_stability": df["rf_stability"].fillna(0.0),
        "dropout_count": df["dropout_count"].astype("float64"),
        "dropout_rate": df["dropout_rate"],
        "rf_rollmax_1h": df["rf_rollmax_1h"],
        "rf_dropouts_roll_1h": df["rf_dropouts_roll_1h"],
    })


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def build_features(cfg: Config | None = None, verbose: bool = True) -> pd.DataFrame:
    """Assemble the full multi-sensor feature frame (one row per 15-min window)."""
    cfg = cfg or default_config()
    if verbose:
        print("extracting pressure features ...")
    feat = _pressure_features(cfg)
    if verbose:
        print("extracting cold-head features ...")
    feat = feat.merge(_cold_head_features(cfg), on=["magnet_id", "window_end"], how="inner")
    if verbose:
        print("extracting vibration features ...")
    feat = feat.merge(_vibration_features(cfg), on=["magnet_id", "window_end"], how="inner")
    if verbose:
        print("extracting RF power features ...")
    feat = feat.merge(_rf_features(cfg), on=["magnet_id", "window_end"], how="inner")

    feat = feat[["magnet_id", "window_end"] + FEATURE_COLUMNS].sort_values(
        ["magnet_id", "window_end"]).reset_index(drop=True)

    nan_share = feat[FEATURE_COLUMNS].isna().mean()
    bad = nan_share[nan_share > 0.0]
    if len(bad):
        # forward-fill *within each magnet*, then zero: never bfill (that would leak
        # the future into the first rows of a magnet's history).
        feat[FEATURE_COLUMNS] = (
            feat.groupby("magnet_id")[FEATURE_COLUMNS].ffill().fillna(0.0)
        )
        if verbose:
            print(f"  filled NaNs in {len(bad)} columns: {list(bad.index)}")

    if verbose:
        print(f"  feature frame: {len(feat):,} windows x {len(FEATURE_COLUMNS)} features "
              f"({feat.magnet_id.nunique()} magnets)")
    return feat


def ensure_features(cfg: Config | None = None, verbose: bool = True) -> pd.DataFrame:
    """Build (or load from cache) the engineered feature frame.

    The cache is invalidated automatically whenever any raw stream is newer.
    """
    cfg = cfg or default_config()
    raw_paths = [cfg.pressure_path, cfg.cold_head_path, cfg.vibration_path, cfg.rf_path]
    if cfg.features_path.exists() and all(p.exists() for p in raw_paths):
        raw_newer = any(p.stat().st_mtime > cfg.features_path.stat().st_mtime for p in raw_paths)
        feat = pd.read_parquet(cfg.features_path)
        if not raw_newer and set(FEATURE_COLUMNS).issubset(feat.columns):
            if verbose:
                print(f"loaded cached features: {len(feat):,} rows "
                      f"from {cfg.features_path.name}")
            return feat
    feat = build_features(cfg, verbose=verbose)
    save_features(feat, cfg)
    return feat


def save_features(feat: pd.DataFrame, cfg: Config | None = None) -> None:
    cfg = cfg or default_config()
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(cfg.features_path, index=False)


def feature_documentation() -> pd.DataFrame:
    """Human-readable dictionary of every model input."""
    return pd.DataFrame(FEATURE_DOCS, columns=["feature", "sensor_group", "description"])
