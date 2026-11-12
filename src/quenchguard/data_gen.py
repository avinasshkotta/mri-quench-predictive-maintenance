"""Synthetic multi-rate sensor data for the MRI magnet quench case study.

Why synthetic?
--------------
Real quench telemetry is (a) proprietary, (b) rare — a fleet produces a handful of
events per year — and (c) unlabelled. The generator therefore encodes the *physics*
described in the case study so that every sensor has a different failure time-scale:

    sensor            onset before quench   crosses its failure signal at
    ----------------- --------------------- -----------------------------------
    helium pressure   14 days (336 h)       ~79 h   (drift below 2.0 MPa)
    cold head temp     96 h                 ~27 h   (rise above 5.5 K)
    vibration (RMS)   120 h                 band energy rises ~84 h,
                                            120 Hz resonance dominant ~44 h
    RF power           48 h                 dropouts ramp in the last ~24 h

That stagger is the whole point of the exercise: a *static threshold* on any single
sensor is either far too late (cold head, 27 h) or too noisy (RF dropouts), while a
sequence model that fuses the trend of all four channels can warn >= 72 h ahead.

Signals written per magnet:
    pressure_1min.parquet     1-minute helium pressure (MPa)
    cold_head_1min.parquet    1-minute cold head temperature (K)
    vibration_window.parquet  one row per 15-min window: RMS / FFT peak / fault-band
                              energy, computed from an on-the-fly 1 kHz capture
                              (raw 1 kHz is aggregated at the edge and never stored)
    rf_window.parquet         one row per 15-min window: power stats + dropout count
                              from 900 simulated 100 ms power-meter readings
    events.csv                quench events + hard negatives (refill, service, tuning)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, RAW_DIR, config_to_dict, default_config

HOUR = pd.Timedelta(hours=1)
MINUTE = pd.Timedelta(minutes=1)

# Deterministic hard-negative + quench plan: (magnet_id, day_offset, kind)
QUENCH_PLAN: list[tuple[str, int]] = [
    ("MRI-01", 20),   # train period  (day 0-54)
    ("MRI-02", 33),   # train
    ("MRI-03", 47),   # train
    ("MRI-04", 58),   # validation     (day 54-72)
    ("MRI-05", 66),   # validation
    ("MRI-01", 78),   # test           (day 72-90)
    ("MRI-06", 80),   # test
    ("MRI-02", 84),   # test
    ("MRI-05", 87),   # test
]

# Near-misses: single-sensor excursions that look like a failure but never become one.
# They are what stops the model from trusting one channel in isolation.
LEAK_PLAN: list[tuple[str, int]] = [          # pressure drifts < 2.0 MPa, then a refill
    ("MRI-02", 12),   # train
    ("MRI-04", 25),   # train
    ("MRI-05", 45),   # train
    ("MRI-01", 56),   # validation
    ("MRI-03", 62),   # validation
    ("MRI-03", 75),   # test
    ("MRI-04", 84),   # test
]
CRYO_PLAN: list[tuple[str, int]] = [          # cold head spikes > 5.5 K, then recovers
    ("MRI-03", 15),   # train
    ("MRI-01", 40),   # train
    ("MRI-06", 50),   # train
    ("MRI-02", 60),   # validation
    ("MRI-05", 74),   # test
    ("MRI-04", 78),   # test
]
VIB_PLAN: list[tuple[str, int]] = [           # 2 h vibration spike, then normal again
    ("MRI-03", 12),   # train
    ("MRI-06", 40),   # train
    ("MRI-06", 68),   # validation
    ("MRI-04", 70),   # validation
    ("MRI-03", 78),   # test
    ("MRI-02", 76),   # test
    ("MRI-05", 74),   # test
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def magnet_ids(cfg: Config) -> list[str]:
    return [f"MRI-{i:02d}" for i in range(1, cfg.fleet.n_magnets + 1)]


def timeline(cfg: Config) -> pd.DatetimeIndex:
    return pd.date_range(cfg.fleet.start, periods=cfg.fleet.days * 1440, freq="1min")


def window_grid(cfg: Config) -> pd.DatetimeIndex:
    """Window *end* stamps: a window covers (t - step, t]."""
    step = pd.Timedelta(minutes=cfg.window.step_minutes)
    start = pd.Timestamp(cfg.fleet.start) + step
    end = pd.Timestamp(cfg.fleet.start) + pd.Timedelta(days=cfg.fleet.days)
    return pd.date_range(start, end, freq=step, inclusive="right")


def _time_to_failure(times: pd.DatetimeIndex | pd.Series, event_time: pd.Timestamp,
                     onset_hours: float) -> np.ndarray:
    """Hours from each timestamp to `event_time`; +inf outside [0, onset]."""
    ttf = (event_time - pd.DatetimeIndex(times)) / HOUR
    ttf = np.asarray(ttf, dtype=np.float64)
    ttf[(ttf < 0) | (ttf > onset_hours)] = np.inf
    return ttf


def _progress(ttf: np.ndarray, onset_hours: float, exponent: float = 1.0) -> np.ndarray:
    """0 -> 1 ramp of failure progress as `ttf` shrinks from onset to 0."""
    p = 1.0 - ttf / onset_hours
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=0.0), 0.0, 1.0)
    return p ** exponent


def event_schedule(cfg: Config) -> pd.DataFrame:
    """Quench events, placed so train / validation / test each contain events."""
    start = pd.Timestamp(cfg.fleet.start)
    rows = []
    for i, (magnet, day) in enumerate(QUENCH_PLAN):
        if day >= cfg.fleet.days:
            continue
        rows.append(
            {
                "magnet_id": magnet,
                "event_time": start + pd.Timedelta(days=day, hours=3),
                "day_offset": day,
                "event_type": "quench",
                "detail": "cryogen loss -> magnet quench",
                "index": i,
            }
        )
    return pd.DataFrame(rows)


def hard_negative_schedule(cfg: Config) -> pd.DataFrame:
    """Realistic non-failures that look alarming to a naive threshold rule."""
    start = pd.Timestamp(cfg.fleet.start)
    rows: list[dict] = []
    for m_i, magnet in enumerate(magnet_ids(cfg)):
        # helium refill: pressure dip + recovery, every ~30 days
        for day in range(9, cfg.fleet.days, 30):
            rows.append({
                "magnet_id": magnet,
                "event_time": start + pd.Timedelta(days=day + m_i * 2 % 7, hours=6),
                "event_type": "refill",
                "detail": "helium top-up: pressure dips ~0.15 MPa then recovers",
            })
        # heavy scan block: cold head runs hot for 3 h, every 4 days
        for day in range(3, cfg.fleet.days, 4):
            rows.append({
                "magnet_id": magnet,
                "event_time": start + pd.Timedelta(days=day, hours=14),
                "event_type": "scan_block",
                "detail": "high duty-cycle scan block: cold head +0.95 K for 3 h",
            })
        # RF coil tuning: dropout burst with a healthy magnet
        for day in range(6, cfg.fleet.days, 4):
            rows.append({
                "magnet_id": magnet,
                "event_time": start + pd.Timedelta(days=day + m_i % 4, hours=10),
                "event_type": "coil_tuning",
                "detail": "RF coil tuning session: 90 min of legitimate dropouts",
            })
    # one service visit per magnet: vibration doubles for 2 days, then recovers
    for m_i, magnet in enumerate(magnet_ids(cfg)):
        rows.append({
            "magnet_id": magnet,
            "event_time": start + pd.Timedelta(days=31 + m_i * 3),
            "event_type": "service",
            "detail": "routine service: gradient coil re-balanced, vibration elevated 48 h",
        })
    # near-miss: slow pressure leak that a refill fixes before it ever becomes a quench
    for magnet, day in LEAK_PLAN:
        if day >= cfg.fleet.days:
            continue
        rows.append({
            "magnet_id": magnet,
            "event_time": start + pd.Timedelta(days=day, hours=2),
            "event_type": "leak_repair",
            "detail": "pressure fell below 2.0 MPa; cryogen top-up restored it (no quench)",
        })
    # near-miss: cryo-cooler hiccup pushes the cold head above 5.5 K for a few hours
    for magnet, day in CRYO_PLAN:
        if day >= cfg.fleet.days:
            continue
        rows.append({
            "magnet_id": magnet,
            "event_time": start + pd.Timedelta(days=day, hours=13),
            "event_type": "cryo_hiccup",
            "detail": "cryo-cooler fault: cold head > 5.5 K for a few hours, then self-recovered",
        })
    # near-miss: 2 h vibration spike (accessory coupling) with a healthy bearing
    for magnet, day in VIB_PLAN:
        if day >= cfg.fleet.days:
            continue
        rows.append({
            "magnet_id": magnet,
            "event_time": start + pd.Timedelta(days=day, hours=9),
            "event_type": "vibration_transient",
            "detail": "accessory coupling: vibration spike for 2 h, then normal",
        })
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["magnet_id", "event_time", "event_type"])


# --------------------------------------------------------------------------------------
# per-sensor simulation
# --------------------------------------------------------------------------------------
def _simulate_pressure(cfg: Config, times: pd.DatetimeIndex,
                       events: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    ph = cfg.physics
    base = ph.pressure_baseline_mpa * (1.0 + rng.normal(0, 0.045))  # per-magnet baseline
    value = np.full(len(times), base, dtype=np.float64)

    # slow diurnal boil-off
    hours = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    value += 0.015 * np.sin(2 * np.pi * (hours - 5.0) / 24.0)

    # pre-failure drift (slow leak): 2.35 -> 1.95 MPa over 14 days, sqrt profile
    for ev in events.itertuples():
        ttf = _time_to_failure(times, ev.event_time, ph.pressure_ramp_hours)
        p = _progress(ttf, ph.pressure_ramp_hours, exponent=0.5)
        value += (ph.pressure_end_mpa - base) * p

    # helium refill hard negatives: smooth dip + recovery
    negs = hard_negative_schedule(cfg)
    for ev in negs[negs.event_type == "refill"].itertuples():
        dt_h = np.asarray((times - ev.event_time) / pd.Timedelta(hours=1), dtype=float)
        mask = np.abs(dt_h) <= 8.0
        value[mask] += -0.15 * np.exp(-0.5 * (dt_h[mask] / 2.5) ** 2)

    # near-miss leak: 60 h ramp below 2.0 MPa, then a refill clears it (no quench).
    # Only the pressure channel reacts -> a single-sensor rule would page on this.
    for ev in negs[negs.event_type == "leak_repair"].itertuples():
        dt_h = np.asarray((times - ev.event_time) / pd.Timedelta(hours=1), dtype=float)
        pre = (dt_h >= -60) & (dt_h < 0)
        prog = 1.0 + dt_h[pre] / 60.0
        value[pre] += -0.42 * np.sqrt(np.clip(prog, 0.0, 1.0))
        post = (dt_h >= 0) & (dt_h <= 14)
        value[post] += -0.42 * np.exp(-dt_h[post] / 3.5)

    # slow calibration / seasonal drift of the pressure baseline
    t_days = np.arange(len(times)) / 1440.0
    value += 0.05 * np.sin(2 * np.pi * t_days / 17.0 + rng.uniform(0, 2 * np.pi))

    value += rng.normal(0, 0.022, size=len(times))
    return value


def _simulate_cold_head(cfg: Config, times: pd.DatetimeIndex,
                        events: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    ph = cfg.physics
    base = ph.cold_head_baseline_k + rng.normal(0, 0.18)   # per-magnet baseline
    value = np.full(len(times), base, dtype=np.float64)

    hours = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    # duty-cycle heating during scan hours + overnight recovery
    duty = np.clip(np.sin(2 * np.pi * (hours - 8.0) / 24.0), 0.0, None)
    value += 0.30 * duty + 0.10 * np.sin(2 * np.pi * hours / 24.0)

    # pre-failure warming: 4.2 -> 6.0 K over 96 h (linear), crosses 5.5 K at ~27 h
    for ev in events.itertuples():
        ttf = _time_to_failure(times, ev.event_time, ph.cold_head_ramp_hours)
        p = _progress(ttf, ph.cold_head_ramp_hours, exponent=1.0)
        value += (ph.cold_head_end_k - base) * p

    # hard negatives: repeated high duty-cycle scan blocks (+0.95 K, 3 h)
    negs = hard_negative_schedule(cfg)
    for ev in negs[negs.event_type == "scan_block"].itertuples():
        dt_h = np.asarray((times - ev.event_time) / pd.Timedelta(hours=1), dtype=float)
        mask = np.abs(dt_h) <= 1.5
        value[mask] += 0.95 * np.exp(-0.5 * (dt_h[mask] / 0.9) ** 2)

    # near-miss: cryo-cooler hiccup crosses 5.5 K for ~3 h, then self-recovers
    for ev in negs[negs.event_type == "cryo_hiccup"].itertuples():
        dt_h = np.asarray((times - ev.event_time) / pd.Timedelta(hours=1), dtype=float)
        mask = np.abs(dt_h) <= 5.0
        value[mask] += 1.45 * np.exp(-0.5 * (dt_h[mask] / 1.7) ** 2)

    # slow baseline drift of the cold head (ambient + calibration)
    t_days = np.arange(len(times)) / 1440.0
    value += 0.12 * np.sin(2 * np.pi * t_days / 23.0 + rng.uniform(0, 2 * np.pi))

    value += rng.normal(0, 0.085, size=len(times))
    # isolated thermocouple glitches exercise the "spike count" feature
    glitch = rng.random(len(times)) < 0.0004
    value[glitch] += rng.uniform(0.4, 1.0, size=int(glitch.sum()))
    return value


def _simulate_vibration(cfg: Config, ends: pd.DatetimeIndex, events: pd.DataFrame,
                        negs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Aggregate an on-the-fly 1 kHz capture into per-window RMS / FFT / band energy."""
    ph = cfg.physics
    fs = 1000
    n_samples = 1024
    t = np.arange(n_samples) / fs
    hann = np.hanning(n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    band_lo, band_hi = ph.fault_band_hz

    baseline = ph.vibration_baseline_rms_g * (1.0 + rng.normal(0, 0.05))
    ttf_all = np.full(len(ends), np.inf)
    for ev in events.itertuples():
        ttf_all = np.minimum(ttf_all, _time_to_failure(ends, ev.event_time,
                                                       ph.vibration_ramp_hours))

    # service visits elevate RMS for 48 h without any incipient failure
    service_penalty = np.zeros(len(ends))
    for ev in negs[negs.event_type == "service"].itertuples():
        dt_h = (ends - ev.event_time).values.astype("timedelta64[h]").astype(float)
        mask = (dt_h >= 0) & (dt_h <= 48)
        service_penalty[mask] = 1.0

    # 2 h accessory-coupling spikes: loud for a moment, then gone
    transient = np.zeros(len(ends))
    for ev in negs[negs.event_type == "vibration_transient"].itertuples():
        dt_h = np.asarray((ends - ev.event_time) / pd.Timedelta(hours=1), dtype=float)
        mask = (dt_h > -0.1) & (dt_h <= 2.1)
        transient[mask] = 1.0

    p = _progress(ttf_all, ph.vibration_ramp_hours, exponent=1.3)
    rms_target = baseline + (ph.vibration_end_rms_g - baseline) * p
    bearing_amp = 0.30 * _progress(ttf_all, ph.vibration_ramp_hours, exponent=2.0)

    chunk = 4096
    out_rms = np.empty(len(ends))
    out_peak = np.empty(len(ends))
    out_band = np.empty(len(ends))
    out_ratio = np.empty(len(ends))
    out_res = np.empty(len(ends))

    for lo in range(0, len(ends), chunk):
        hi = min(lo + chunk, len(ends))
        n = hi - lo
        # broadband floor + 45 Hz structural tone + optional 120 Hz bearing tone
        y = rng.normal(0.0, 0.055, size=(n, n_samples))
        y += 0.11 * np.sin(2 * np.pi * 45.0 * t[None, :] + rng.uniform(0, 2 * np.pi, n)[:, None])
        y += 0.020 * np.sin(2 * np.pi * 50.0 * t[None, :] + rng.uniform(0, 2 * np.pi, n)[:, None])
        y += (bearing_amp[lo:hi, None]
              * np.sin(2 * np.pi * ph.resonance_hz * t[None, :]
                       + rng.uniform(0, 2 * np.pi, n)[:, None]))
        y += service_penalty[lo:hi, None] * rng.normal(0.0, 0.10, size=(n, n_samples))
        y += transient[lo:hi, None] * (
            0.26 * np.sin(2 * np.pi * 70.0 * t[None, :]
                          + rng.uniform(0, 2 * np.pi, n)[:, None])
            + rng.normal(0.0, 0.12, size=(n, n_samples)))

        spec = np.abs(np.fft.rfft(y * hann[None, :], axis=1)) ** 2
        spec[:, freqs < 10.0] = 0.0                       # drop DC / slow drift
        total = spec.sum(axis=1) + 1e-12
        out_peak[lo:hi] = freqs[np.argmax(spec, axis=1)]
        band = (freqs >= band_lo) & (freqs <= band_hi)
        out_band[lo:hi] = spec[:, band].sum(axis=1)
        out_ratio[lo:hi] = out_band[lo:hi] / total
        res = (freqs >= ph.resonance_hz - 5) & (freqs <= ph.resonance_hz + 5)
        out_res[lo:hi] = spec[:, res].sum(axis=1)
        out_rms[lo:hi] = np.sqrt(np.mean(y ** 2, axis=1))

    # Anchor the RMS level to the physics ramp (bearing wear) while keeping per-window
    # noise, then add the non-failure elevations that must NOT be alerted on.
    out_rms = rms_target * (1.0 + rng.normal(0.0, 0.05, size=len(ends)))
    out_rms += service_penalty * (ph.vibration_baseline_rms_g * 1.1)
    out_rms += transient * 0.30
    out_rms = np.clip(out_rms, 0.02, None)

    return pd.DataFrame({
        "window_end": ends,
        "rms_g": out_rms,
        "peak_hz": out_peak,
        "fault_energy": out_band,
        "fault_ratio": out_ratio,
        "resonance_energy": out_res,
        "bearing_amp": bearing_amp,
    })


def _simulate_rf(cfg: Config, ends: pd.DatetimeIndex, events: pd.DataFrame,
                 negs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """900 simulated 100 ms power-meter readings per 15-min window -> aggregates."""
    ph = cfg.physics
    readings = int(cfg.window.step_minutes * 60 / 0.1)  # 900
    step_s = cfg.window.step_minutes * 60

    ttf_all = np.full(len(ends), np.inf)
    for ev in events.itertuples():
        ttf_all = np.minimum(ttf_all, _time_to_failure(ends, ev.event_time,
                                                       ph.rf_dropout_ramp_hours))
    p_fail = _progress(ttf_all, ph.rf_dropout_ramp_hours, exponent=2.0)
    dropout_p = ph.rf_dropout_p0 + (ph.rf_dropout_p1 - ph.rf_dropout_p0) * p_fail

    # hard negative: recurring coil-tuning session with legitimate dropouts (90 min)
    tuning = np.zeros(len(ends))
    for ev in negs[negs.event_type == "coil_tuning"].itertuples():
        dt_h = np.asarray((ends - ev.event_time) / pd.Timedelta(hours=1), dtype=float)
        mask = (dt_h > -0.1) & (dt_h <= 1.6)
        tuning[mask] = 1.0
    dropout_p = np.clip(dropout_p + tuning * (0.35 - ph.rf_dropout_p0), 0.0, 1.0)

    n = len(ends)
    dropout_count = rng.binomial(readings, dropout_p)
    sag = 0.02 * p_fail + 0.01 * tuning
    power_mean = ph.rf_nominal_w * (1.0 - sag) + rng.normal(0, 6.0, n)
    power_std = 8.0 + 45.0 * p_fail + 20.0 * tuning + rng.normal(0, 1.5, n)
    power_std = np.clip(power_std, 3.0, None)
    power_max = power_mean + 3.0 * power_std
    return pd.DataFrame({
        "window_end": ends,
        "rf_mean_w": power_mean,
        "rf_std_w": power_std,
        "rf_max_w": power_max,
        "dropout_count": dropout_count,
        "readings": readings,
        "dropout_p": dropout_p,
    })


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------
def generate_fleet(cfg: Config | None = None, overwrite: bool = True,
                   verbose: bool = True) -> dict[str, Path]:
    """Generate and persist all raw sensor streams for the synthetic fleet."""
    cfg = cfg or default_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if not overwrite and cfg.pressure_path.exists() and verbose:
        return {k: v for k, v in paths.items() if v.exists()}

    events = event_schedule(cfg)
    negs = hard_negative_schedule(cfg)
    ids = magnet_ids(cfg)
    times = timeline(cfg)
    ends = window_grid(cfg)

    pressure_frames, cold_frames, vib_frames, rf_frames = [], [], [], []

    for i, magnet in enumerate(ids):
        rng = np.random.default_rng(cfg.fleet.seed + 1000 * i)
        mag_events = events[events.magnet_id == magnet]
        mag_negs = negs[negs.magnet_id == magnet]

        pressure_frames.append(pd.DataFrame({
            "timestamp": times,
            "magnet_id": magnet,
            "pressure_mpa": _simulate_pressure(cfg, times, mag_events, rng),
        }))
        cold_frames.append(pd.DataFrame({
            "timestamp": times,
            "magnet_id": magnet,
            "cold_head_k": _simulate_cold_head(cfg, times, mag_events, rng),
        }))

        vib = _simulate_vibration(cfg, ends, mag_events, mag_negs, rng)
        vib.insert(1, "magnet_id", magnet)
        vib_frames.append(vib)

        rf = _simulate_rf(cfg, ends, mag_events, mag_negs, rng)
        rf.insert(1, "magnet_id", magnet)
        rf_frames.append(rf)

        if verbose:
            print(f"  simulated {magnet}: {len(times):,} min-rows, {len(ends):,} windows")

    pressure = pd.concat(pressure_frames, ignore_index=True)
    cold = pd.concat(cold_frames, ignore_index=True)
    vib = pd.concat(vib_frames, ignore_index=True)
    rf = pd.concat(rf_frames, ignore_index=True)

    pressure.to_parquet(cfg.pressure_path, index=False)
    cold.to_parquet(cfg.cold_head_path, index=False)
    vib.to_parquet(cfg.vibration_path, index=False)
    rf.to_parquet(cfg.rf_path, index=False)

    ev_all = pd.concat([events.assign(is_failure=True),
                        negs.assign(is_failure=False, day_offset=np.nan, index=np.nan)],
                       ignore_index=True)
    ev_all.sort_values(["event_time", "magnet_id"]).to_csv(cfg.events_path, index=False)

    manifest = {
        "generated_by": "quenchguard.data_gen.generate_fleet",
        "rows": {
            "pressure": len(pressure),
            "cold_head": len(cold),
            "vibration_windows": len(vib),
            "rf_windows": len(rf),
            "events": int((ev_all.is_failure).sum()),
            "hard_negatives": int((~ev_all.is_failure).sum()),
        },
        "config": config_to_dict(cfg),
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    paths = {
        "pressure": cfg.pressure_path,
        "cold_head": cfg.cold_head_path,
        "vibration": cfg.vibration_path,
        "rf": cfg.rf_path,
        "events": cfg.events_path,
        "manifest": RAW_DIR / "manifest.json",
    }
    if verbose:
        print(f"\n wrote {len(pressure):,} pressure rows | {len(cold):,} cold-head rows | "
              f"{len(vib):,} vibration windows | {len(rf):,} RF windows")
    return paths


# --------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------
def _ensure(cfg: Config, path: Path) -> Path:
    if not path.exists():
        generate_fleet(cfg, verbose=False)
    return path


def load_pressure(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or default_config()
    return pd.read_parquet(_ensure(cfg, cfg.pressure_path))


def load_cold_head(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or default_config()
    return pd.read_parquet(_ensure(cfg, cfg.cold_head_path))


def load_vibration(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or default_config()
    return pd.read_parquet(_ensure(cfg, cfg.vibration_path))


def load_rf(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or default_config()
    return pd.read_parquet(_ensure(cfg, cfg.rf_path))


def load_events(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or default_config()
    _ensure(cfg, cfg.events_path)
    df = pd.read_csv(cfg.events_path, parse_dates=["event_time"])
    return df


def load_quench_events(cfg: Config | None = None) -> pd.DataFrame:
    """Only true quench events, one row per failure."""
    df = load_events(cfg)
    return df[df.event_type == "quench"].sort_values("event_time").reset_index(drop=True)
