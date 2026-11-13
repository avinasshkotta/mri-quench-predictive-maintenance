"""External validation on NASA C-MAPSS FD001 (real turbofan degradation data).

Why this file exists
--------------------
The MRI study is synthetic. A capstone claim of "this methodology works" needs
evidence on data the authors did **not** generate. C-MAPSS FD001 is the standard
public run-to-failure benchmark: 100 engines run to failure, 100 separate engines
truncated before failure, with official RUL ground truth.

What is validated (and what is not)
-----------------------------------
This validates the **methodology**, not the MRI model:

* same task shape: binary "failure within a horizon" (H = 30 cycles, the
  maintenance-planning analogue of the MRI 96 h horizon);
* same causal feature philosophy: trailing rolling statistics per unit, no
  future information, no engine id, no cycle index;
* same evaluation semantics: alert episodes, event recall with a lead-time
  deadline (15 cycles), false alerts outside the horizon, threshold chosen on a
  held-out engine split under a false-alert budget;
* same model family as the MRI baseline (HistGBDT), plus a logistic baseline.

The MRI LSTM is **not** transferred — a different sensor set, sampling rate and
physics make weight transfer meaningless. What should generalise is the
*decision framework*.

C-MAPSS-specific honesty
------------------------
* Test trajectories are truncated at arbitrary points. If an engine's true RUL
  at truncation is **greater than H**, its failure horizon lies beyond the
  observed data: event recall is *not evaluable* for it (right-censored), and
  any alert it produces is a false alert because the failure is > H away.
* If RUL at truncation is **less than the lead deadline**, a perfect model still
  cannot warn in time — the deadline has already passed. Those engines are
  excluded from the in-time recall denominator and reported separately.
* There are no dollar costs in C-MAPSS, so the MRI cost model is deliberately
  not applied (inventing costs would be dishonest).

Data: Saxena & Goebel (2008), NASA Prognostics Data Repository. Files are
downloaded on demand and git-ignored.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, REPO_ROOT

# --------------------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------------------
CMAPSS_DIR = REPO_ROOT / "data" / "external" / "cmapss"
_SOURCE = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/"
_FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")
_COLUMNS = (["unit", "cycle", "set1", "set2", "set3"]
            + [f"s{i}" for i in range(1, 22)])
CONSTANT_SENSORS = ("s1", "s5", "s10", "s16", "s18", "s19")
INFORMATIVE_SENSORS = tuple(c for c in _COLUMNS[5:] if c not in CONSTANT_SENSORS)

# task definition (documented in the model card / external-validation doc)
HORIZON_CYCLES = 30          # failure within 30 cycles == MRI "within 96 h"
MIN_LEAD_CYCLES = 15         # 15-cycle warning deadline (half the horizon)
ROLL_WINDOW = 10             # trailing cycles used for rolling statistics
BASELINE_CYCLES = 5          # first cycles define each engine's healthy baseline
FALSE_ALERT_BUDGET_PER_UNIT = 0.6    # 1.0 page/engine x 0.6 safety, as in the MRI card
N_TRAIN_UNITS = 80           # grouped split: 80 train / 20 validation engines
SEED = 42


def cmapss_paths(base: Path | None = None) -> dict[str, Path]:
    base = base or CMAPSS_DIR
    return {name: base / name for name in _FILES}


def download_cmapss(base: Path | None = None, verbose: bool = True) -> dict[str, Path]:
    """Fetch the three FD001 files if absent; returns their paths."""
    paths = cmapss_paths(base)
    paths[_FILES[0]].parent.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        if path.exists() and path.stat().st_size > 100:
            continue
        with urllib.request.urlopen(_SOURCE + name, timeout=120) as resp:
            path.write_bytes(resp.read())
        if verbose:
            print(f"  downloaded {name} ({path.stat().st_size:,} bytes)")
    return paths


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cmapss(base: Path | None = None,
                download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load train / test / RUL frames with named columns."""
    paths = download_cmapss(base) if download else cmapss_paths(base)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "C-MAPSS FD001 files missing: " + ", ".join(missing)
            + " — run `python scripts/validate_cmapss.py --download`")
    train = pd.read_csv(paths["train_FD001.txt"], sep=r"\s+", header=None,
                        names=_COLUMNS)
    test = pd.read_csv(paths["test_FD001.txt"], sep=r"\s+", header=None,
                       names=_COLUMNS)
    rul = pd.read_csv(paths["RUL_FD001.txt"], sep=r"\s+", header=None,
                      names=["rul"])["rul"].astype(float)
    return train, test, rul


def add_rul(df: pd.DataFrame, rul_final: pd.Series | None = None) -> pd.DataFrame:
    """Attach the *true* remaining useful life per row.

    * run-to-failure frames (``rul_final=None``): RUL = max_cycle - cycle;
    * truncated test frames: the official RUL is given at the **last observed**
      cycle, so RUL(t) = rul_final + (last_cycle - t). This is exact, not an
      approximation — the engine keeps ageing one cycle per cycle.
    """
    out = df.copy().sort_values(["unit", "cycle"]).reset_index(drop=True)
    last = out.groupby("unit")["cycle"].transform("max")
    if rul_final is None:
        out["rul"] = (last - out["cycle"]).astype(float)
    else:
        mapping = {i + 1: float(v) for i, v in enumerate(rul_final.to_numpy())}
        out["rul"] = out["unit"].map(mapping) + (last - out["cycle"]).astype(float)
    return out


# --------------------------------------------------------------------------------------
# causal features
# --------------------------------------------------------------------------------------
def build_features(df: pd.DataFrame, sensors: tuple[str, ...] = INFORMATIVE_SENSORS,
                   window: int = ROLL_WINDOW,
                   baseline_cycles: int = BASELINE_CYCLES) -> pd.DataFrame:
    """Trailing rolling statistics + deviation from each engine's healthy baseline.

    Causality: every feature at cycle ``t`` uses rows ``<= t`` only (trailing
    ``rolling``, ``shift`` backwards). Unit id and cycle index are deliberately
    **not** features — the model must read degradation from sensors, not age.
    """
    out = df.sort_values(["unit", "cycle"]).reset_index(drop=True).copy()
    grouped = out.groupby("unit", sort=False)
    for c in sensors:
        mean = grouped[c].transform(lambda s: s.rolling(window, min_periods=1).mean())
        std = grouped[c].transform(
            lambda s: s.rolling(window, min_periods=2).std()).fillna(0.0)
        base = grouped[c].transform(lambda s: float(s.iloc[:baseline_cycles].mean()))
        slope = grouped[c].transform(
            lambda s: (s - s.shift(window)) / float(window)).fillna(0.0)
        out[f"{c}_mean"] = mean
        out[f"{c}_std"] = std
        out[f"{c}_delta"] = out[c] - base
        out[f"{c}_slope"] = slope
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.endswith(("_mean", "_std", "_delta", "_slope"))]


# --------------------------------------------------------------------------------------
# evaluation (cycle-domain mirrors of evaluate.match_events)
# --------------------------------------------------------------------------------------
def alert_episodes_cycles(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Collapse consecutive positive cycles per unit into one alert episode."""
    d = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    pred = d["score"].to_numpy() >= threshold
    units = d["unit"].to_numpy()
    cycles = d["cycle"].to_numpy()
    rows: list[dict[str, Any]] = []
    start = None
    for i in range(len(d)):
        if pred[i] and (i == 0 or not pred[i - 1] or units[i] != units[i - 1]):
            start = i
        end = (i == len(d) - 1) or (not pred[i + 1]) or (units[i + 1] != units[i])
        if pred[i] and end and start is not None:
            rows.append({"unit": int(units[i]),
                         "alert_cycle": int(cycles[start]),
                         "last_cycle": int(cycles[i])})
    return pd.DataFrame(rows, columns=["unit", "alert_cycle", "last_cycle"])


def match_events_cycles(events: pd.DataFrame, episodes: pd.DataFrame,
                        horizon: int = HORIZON_CYCLES,
                        min_lead: int = MIN_LEAD_CYCLES) -> pd.DataFrame:
    """One row per engine failure: detected / detected_in_time / lead.

    Same semantics as ``evaluate.match_events``: an episode counts when it is
    active inside the horizon (``alert_cycle < event_cycle`` and
    ``last_cycle >= event_cycle - horizon``); lead is clamped to the horizon.
    """
    rows = []
    for ev in events.itertuples():
        cand = episodes[(episodes.unit == ev.unit)
                        & (episodes.alert_cycle < ev.event_cycle)
                        & (episodes.last_cycle >= ev.event_cycle - horizon)]
        if len(cand):
            first = cand.loc[cand.alert_cycle.idxmin()]
            warn_from = max(int(first.alert_cycle), int(ev.event_cycle) - horizon)
            lead = int(ev.event_cycle) - warn_from
            raw = int(ev.event_cycle) - int(first.alert_cycle)
        else:
            first, lead, raw = None, np.nan, np.nan
        rows.append({
            "unit": int(ev.unit),
            "event_cycle": int(ev.event_cycle),
            "last_observed_cycle": int(ev.last_observed_cycle),
            "rul_at_truncation": int(ev.rul_at_truncation),
            "observable": bool(ev.observable),
            "evaluable_in_time": bool(ev.evaluable_in_time),
            "alert_cycle": None if first is None else int(first.alert_cycle),
            "lead_cycles": lead,
            "lead_cycles_unclamped": raw,
            "detected": bool(len(cand)),
            "detected_in_time": bool(len(cand) and lead >= min_lead),
        })
    return pd.DataFrame(rows)


def false_alerts_cycles(episodes: pd.DataFrame, events: pd.DataFrame,
                        horizon: int = HORIZON_CYCLES) -> pd.DataFrame:
    """Episodes that never overlap any engine's failure horizon."""
    if not len(episodes):
        return episodes.assign(is_false=[])
    if not len(events):
        out = episodes.copy()
        out["is_false"] = True
        return out
    flags = []
    for ep in episodes.itertuples():
        in_horizon = ((events.unit == ep.unit)
                      & (events.event_cycle > ep.alert_cycle)
                      & (events.event_cycle - horizon <= ep.last_cycle)).any()
        flags.append(not bool(in_horizon))
    out = episodes.copy()
    out["is_false"] = flags
    return out


def sweep_thresholds_cycles(frame: pd.DataFrame, events: pd.DataFrame,
                            horizon: int = HORIZON_CYCLES,
                            min_lead: int = MIN_LEAD_CYCLES) -> pd.DataFrame:
    from sklearn.metrics import f1_score, precision_score, recall_score

    scores = frame["score"].to_numpy(float)
    if not len(scores):
        return pd.DataFrame()
    candidates = np.unique(np.round(np.concatenate(
        [[0.5, 0.9, 1.0], np.quantile(scores, np.arange(0.50, 0.999, 0.005))]), 6))
    n_units = frame["unit"].nunique()
    rows = []
    for thr in candidates:
        ep = alert_episodes_cycles(frame, float(thr))
        false = false_alerts_cycles(ep, events, horizon)
        outcomes = match_events_cycles(events, ep, horizon, min_lead)
        y = frame["y"].to_numpy(int)
        yhat = (scores >= thr).astype(int)
        n_false = int(false["is_false"].sum()) if len(false) else 0
        rows.append({
            "threshold": float(thr),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0),
            "event_recall": float(outcomes.detected.mean()),
            "event_recall_in_time": float(outcomes.detected_in_time.mean()),
            "n_episodes": len(ep),
            "false_alerts": n_false,
            "false_per_unit": n_false / n_units if n_units else np.inf,
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def choose_threshold_cycles(sweep: pd.DataFrame,
                            budget: float = FALSE_ALERT_BUDGET_PER_UNIT) -> dict[str, Any]:
    """Highest in-time event recall inside the false-alert budget."""
    if not len(sweep):
        return {"threshold": 0.5, "reason": "empty sweep", "budget": budget}
    feasible = sweep[sweep.false_per_unit <= budget]
    if len(feasible):
        best = feasible.sort_values(
            ["event_recall_in_time", "event_recall", "f1", "threshold"],
            ascending=[False, False, False, False]).iloc[0]
        reason = (f"best in-time event recall among {len(feasible)} operating "
                  f"points with <= {budget:.2f} false alerts/engine")
    else:
        best = sweep.assign(gap=(sweep.false_per_unit - budget).abs()).sort_values(
            ["gap", "event_recall_in_time"], ascending=[True, False]).iloc[0]
        reason = "no point met the budget; picked the closest operating point"
    return {"threshold": float(best.threshold), "reason": reason,
            "budget": budget, "row": best.to_dict()}


def evaluate_cycles(frame: pd.DataFrame, events: pd.DataFrame, threshold: float,
                    horizon: int = HORIZON_CYCLES,
                    min_lead: int = MIN_LEAD_CYCLES) -> dict[str, Any]:
    """Window-level + event-level metrics in the cycle domain."""
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    y = frame["y"].to_numpy(int)
    s = frame["score"].to_numpy(float)
    yhat = (s >= threshold).astype(int)
    episodes = alert_episodes_cycles(frame, threshold)
    false = false_alerts_cycles(episodes, events, horizon)
    outcomes = match_events_cycles(events, episodes, horizon, min_lead)
    n_false = int(false["is_false"].sum()) if len(false) else 0

    observable = outcomes[outcomes.observable]
    in_time = outcomes[outcomes.evaluable_in_time]
    lead = outcomes.loc[outcomes.detected, "lead_cycles"].dropna()
    lead_raw = outcomes.loc[outcomes.detected, "lead_cycles_unclamped"].dropna()

    return {
        "split_size_rows": int(len(frame)),
        "n_units": int(frame.unit.nunique()),
        "n_units_with_event": int(len(outcomes)),
        "n_observable_events": int(len(observable)),
        "n_evaluable_in_time": int(len(in_time)),
        "threshold": float(threshold),
        "window": {
            "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else None,
            "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None,
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat, zero_division=0)),
            "f1": float(f1_score(y, yhat, zero_division=0)),
            "positive_rate": float(y.mean()),
        },
        "events": {
            "event_recall": float(observable.detected.mean()) if len(observable) else None,
            "event_recall_in_time": float(in_time.detected_in_time.mean()) if len(in_time) else None,
            "detected": int(observable.detected.sum()),
            "detected_in_time": int(in_time.detected_in_time.sum()),
            "lead_median": float(lead.median()) if len(lead) else None,
            "lead_min": float(lead.min()) if len(lead) else None,
            "lead_max": float(lead.max()) if len(lead) else None,
            "lead_median_unclamped": float(lead_raw.median()) if len(lead_raw) else None,
            "lead_max_unclamped": float(lead_raw.max()) if len(lead_raw) else None,
        },
        "alerts": {
            "episodes": int(len(episodes)),
            "false_alerts": n_false,
            "false_per_unit": n_false / frame.unit.nunique() if len(frame) else None,
            "false_per_1000_cycles": (n_false / len(frame) * 1000) if len(frame) else None,
        },
        "_episodes": episodes,
        "_outcomes": outcomes,
        "_false": false,
    }


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------
def _public(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def run_external_validation(base: Path | None = None, download: bool = False,
                            write_reports: bool = True,
                            n_train_units: int = N_TRAIN_UNITS,
                            verbose: bool = True) -> dict[str, Any]:
    """Train on FD001 train engines, tune on held-out engines, test on NASA test."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_raw, test_raw, rul_final = load_cmapss(base, download=download)
    paths = cmapss_paths(base)
    provenance = {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                  for name, path in paths.items() if path.exists()}

    train_lab = add_rul(train_raw)
    test_lab = add_rul(test_raw, rul_final)
    train_feat = build_features(train_lab)
    test_feat = build_features(test_lab)
    cols = feature_columns(train_feat)

    # grouped split by engine — never by row
    all_units = sorted(train_feat.unit.unique())
    if not 0 < n_train_units < len(all_units):
        raise ValueError(f"n_train_units must leave at least one validation "
                         f"engine (got {n_train_units} of {len(all_units)})")
    train_units = all_units[:n_train_units]
    val_units = all_units[n_train_units:]
    tr = train_feat[train_feat.unit.isin(train_units)].copy()
    va = train_feat[train_feat.unit.isin(val_units)].copy()
    te = test_feat.copy()
    for part in (tr, va, te):
        part["y"] = (part["rul"] <= HORIZON_CYCLES).astype(int)

    models = {
        "hist_gbdt": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=SEED),
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=0.5)),
    }
    for model in models.values():
        model.fit(tr[cols], tr["y"])
    val_scores = {name: m.predict_proba(va[cols])[:, 1] for name, m in models.items()}
    test_scores = {name: m.predict_proba(te[cols])[:, 1] for name, m in models.items()}

    # validation events: every train engine ran to failure at its last cycle
    def events_from(frame: pd.DataFrame) -> pd.DataFrame:
        last = frame.groupby("unit")["cycle"].max()
        rul = frame.groupby("unit")["rul"].min()      # 0 at the final cycle
        return pd.DataFrame({
            "unit": last.index.astype(int),
            "event_cycle": last.to_numpy(int),
            "last_observed_cycle": last.to_numpy(int),
            "rul_at_truncation": rul.to_numpy(int),
            "observable": True,
            "evaluable_in_time": True,
        })

    val_events = events_from(va)
    last = te.groupby("unit")["cycle"].max()
    rul_trunc = te.groupby("unit")["rul"].min()
    test_events = pd.DataFrame({
        "unit": last.index.astype(int),
        "event_cycle": (last + rul_trunc).to_numpy(int),
        "last_observed_cycle": last.to_numpy(int),
        "rul_at_truncation": rul_trunc.to_numpy(int),
        "observable": (rul_trunc <= HORIZON_CYCLES).to_numpy(),
        "evaluable_in_time": ((rul_trunc >= MIN_LEAD_CYCLES)
                              & (rul_trunc <= HORIZON_CYCLES)).to_numpy(),
    })

    results: dict[str, Any] = {}
    for name, model in models.items():
        va_frame = pd.DataFrame({"unit": va.unit.to_numpy(), "cycle": va.cycle.to_numpy(),
                                 "y": va.y.to_numpy(), "score": val_scores[name]})
        sweep = sweep_thresholds_cycles(va_frame, val_events)
        choice = choose_threshold_cycles(sweep)
        te_frame = pd.DataFrame({"unit": te.unit.to_numpy(), "cycle": te.cycle.to_numpy(),
                                 "y": te.y.to_numpy(), "score": test_scores[name]})
        metrics = evaluate_cycles(te_frame, test_events, choice["threshold"])
        results[name] = {
            "threshold": choice["threshold"],
            "threshold_reason": choice["reason"],
            "validation": {
                "pr_auc": float(average_precision_score(va.y, val_scores[name])),
                "best_val_event_recall_in_time": float(sweep.event_recall_in_time.max()),
            },
            "test": _public(metrics),
        }

    champion = "hist_gbdt"
    report = {
        "dataset": {
            "name": "NASA C-MAPSS FD001",
            "citation": "Saxena & Goebel (2008), NASA Prognostics Data Repository",
            "train_rows": int(len(train_raw)), "train_units": int(train_raw.unit.nunique()),
            "test_rows": int(len(test_raw)), "test_units": int(test_raw.unit.nunique()),
            "informative_sensors": list(INFORMATIVE_SENSORS),
            "dropped_constant_sensors": list(CONSTANT_SENSORS),
            "provenance": provenance,
        },
        "protocol": {
            "horizon_cycles": HORIZON_CYCLES, "min_lead_cycles": MIN_LEAD_CYCLES,
            "roll_window": ROLL_WINDOW, "baseline_cycles": BASELINE_CYCLES,
            "n_train_units": len(train_units), "n_val_units": len(val_units),
            "false_alert_budget_per_unit": FALSE_ALERT_BUDGET_PER_UNIT,
            "split": "grouped by engine (80 train / 20 validation); NASA test held out",
            "features": len(cols),
            "note": "methodology validation, not MRI model transfer; no dollar costs "
                    "exist in C-MAPSS so the MRI cost model is not applied",
        },
        "results": results,
        "champion": champion,
    }

    if write_reports:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "external_validation.json").write_text(
            json.dumps(report, indent=2, default=float), encoding="utf-8")
        try:
            _plot_validation(te, test_scores[champion], test_events,
                             results[champion]["threshold"], report)
        except Exception as exc:                     # plotting must never break the run
            if verbose:
                print(f"  (plots skipped: {exc})")

    if verbose:
        for name, r in results.items():
            t = r["test"]
            print(f"  [{name}] threshold {r['threshold']:.3f} | PR-AUC "
                  f"{t['window']['pr_auc']:.3f} | event recall "
                  f"{t['events']['event_recall']:.0%} ({t['events']['detected']}/"
                  f"{t['n_observable_events']}) | in time "
                  f"{t['events']['event_recall_in_time']:.0%} "
                  f"({t['events']['detected_in_time']}/{t['n_evaluable_in_time']}) | "
                  f"false/unit {t['alerts']['false_per_unit']:.2f}")
    return report


def _plot_validation(te: pd.DataFrame, scores: np.ndarray, events: pd.DataFrame,
                     threshold: float, report: dict[str, Any]) -> None:
    """Score trajectories for event-bearing engines + lead-time histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    frame = te.copy()
    frame["score"] = scores

    chosen = events[events.observable].sort_values("rul_at_truncation").unit.tolist()[:4]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(11, 2.6 * len(chosen)),
                             sharex=False)
    axes = np.atleast_1d(axes)
    for ax, unit in zip(axes, chosen):
        g = frame[frame.unit == unit].sort_values("cycle")
        ev = events[events.unit == unit].iloc[0]
        ax.plot(g.cycle, g.score, color="#1f77b4", lw=1.4, label="predicted risk")
        ax.axhline(threshold, color="crimson", ls="--", lw=1.1,
                   label=f"threshold {threshold:.2f}")
        ax.axvline(ev.event_cycle, color="black", lw=1.5, label="failure")
        ax.axvspan(ev.event_cycle - HORIZON_CYCLES,
                   ev.event_cycle - MIN_LEAD_CYCLES, color="orange", alpha=0.10,
                   lw=0, label="96h-analogue horizon")
        ax.set_title(f"Engine {unit} — RUL at truncation {int(ev.rul_at_truncation)} "
                     f"cycles", fontsize=9)
        ax.set_ylabel("p(fail <= 30)")
        ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("flight cycle")
    fig.suptitle("C-MAPSS FD001 — external validation on NASA test engines", y=0.995)
    fig.tight_layout()
    fig.savefig(fig_dir / "cmapss_score_trajectories.png", dpi=130)
    plt.close(fig)

    outcomes = report["results"][report["champion"]]["test"]
    lead = outcomes["events"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [lead["lead_median"], lead["lead_min"], lead["lead_max"]]
    if all(v is not None for v in vals):
        ax.bar(["min", "median", "max"], vals, color="#4c72b0")
        ax.axhline(MIN_LEAD_CYCLES, color="crimson", ls="--",
                   label=f"required {MIN_LEAD_CYCLES} cycles")
        ax.set_ylabel("lead time before failure (cycles)")
        ax.legend(fontsize=9)
    ax.set_title("C-MAPSS FD001 — detection lead time (external validation)")
    fig.tight_layout()
    fig.savefig(fig_dir / "cmapss_lead_times.png", dpi=130)
    plt.close(fig)


__all__ = ["run_external_validation", "load_cmapss", "download_cmapss",
           "add_rul", "build_features", "feature_columns",
           "alert_episodes_cycles", "match_events_cycles", "false_alerts_cycles",
           "sweep_thresholds_cycles", "choose_threshold_cycles", "evaluate_cycles",
           "HORIZON_CYCLES", "MIN_LEAD_CYCLES", "CMAPSS_DIR"]
