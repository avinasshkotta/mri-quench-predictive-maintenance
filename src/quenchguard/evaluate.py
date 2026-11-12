"""Evaluation: threshold policy, alert episodes, lead time, cost model, plots.

The metrics here are deliberately *business* metrics, not textbook ones:

* **event recall @ lead >= 72 h** — the acceptance criterion from the case study
* **false alerts per magnet-month** — the <= 1 page/magnet/month constraint
* **lead-time distribution** — does the warning actually arrive early enough?
* **dollar cost model** — $120k repair + 5 days downtime vs. a $2k wasted visit

Definitions
-----------
*alert episode*  – a maximal run of consecutive positive windows on one magnet; the
                   alert fires at the **first** window of the run.
*true alert*     – an episode whose first window lies within ``horizon_hours`` before
                   a real quench.
*false alert*    – any other episode (the magnet never fails inside the horizon).
*detected in time* – event whose earliest matching alert is >= ``min_lead_hours``
                   before the failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from .config import Config, FIGURES_DIR, default_config
from .features import FEATURE_GROUPS

sns.set_theme(style="whitegrid", context="notebook")


# --------------------------------------------------------------------------------------
# score frames / episodes
# --------------------------------------------------------------------------------------
def make_score_frame(split, scores: np.ndarray, name: str = "split") -> pd.DataFrame:
    """Tidy frame: one row per scored sequence."""
    return pd.DataFrame({
        "magnet_id": split.magnets,
        "t": pd.to_datetime(split.times),
        "y": split.y.astype(int),
        "score": np.asarray(scores, dtype=float),
        "split": name,
    }).sort_values(["magnet_id", "t"]).reset_index(drop=True)


def magnet_months(frame: pd.DataFrame) -> float:
    """Observation time in magnet-months (denominator for the false-page budget)."""
    span = frame.groupby("magnet_id")["t"].agg(lambda s: (s.max() - s.min()).total_seconds())
    return float((span / 86400.0).sum() / 30.4375)


def alert_episodes(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Collapse consecutive positive windows into one alert per run."""
    d = frame.sort_values(["magnet_id", "t"]).reset_index(drop=True)
    pred = (d["score"].to_numpy() >= threshold)
    magnet = d["magnet_id"].to_numpy()
    t = d["t"].to_numpy()
    score = d["score"].to_numpy()

    prev_pred = np.r_[False, pred[:-1]]
    new_magnet = np.r_[True, magnet[1:] != magnet[:-1]]
    starts = np.where(pred & (~prev_pred | new_magnet))[0]

    rows = []
    for s in starts:
        e = s + 1
        while (e < len(pred)) and pred[e] and magnet[e] == magnet[s]:
            e += 1
        rows.append({
            "magnet_id": magnet[s],
            "alert_time": pd.Timestamp(t[s]),
            "last_time": pd.Timestamp(t[e - 1]),
            "peak_score": float(score[s:e].max()),
            "windows": int(e - s),
        })
    return pd.DataFrame(rows, columns=["magnet_id", "alert_time", "last_time",
                                       "peak_score", "windows"])


def match_events(events: pd.DataFrame, episodes: pd.DataFrame, cfg: Config
                 ) -> pd.DataFrame:
    """One row per quench: the warning episode, lead time, detected flags.

    An episode *overlaps* the horizon when it starts before the failure and is still
    active (or starts) no earlier than ``horizon_hours`` before it. Alerts that began
    before the horizon are clamped to it, so lead time never exceeds the labelling
    horizon — the model is only ever scored against what it was trained to predict.
    """
    horizon = pd.Timedelta(hours=cfg.window.horizon_hours)
    min_lead = cfg.window.min_lead_hours
    rows = []
    for ev in events.itertuples():
        cand = episodes[
            (episodes.magnet_id == ev.magnet_id)
            & (episodes.alert_time < ev.event_time)
            & (episodes.last_time >= ev.event_time - horizon)
        ] if len(episodes) else episodes
        if len(cand):
            first = cand.loc[cand.alert_time.idxmin()]
            warn_from = max(first.alert_time, ev.event_time - horizon)
            lead = (ev.event_time - warn_from).total_seconds() / 3600.0
            raw_lead = (ev.event_time - first.alert_time).total_seconds() / 3600.0
        else:
            first, lead, raw_lead = None, np.nan, np.nan
        rows.append({
            "magnet_id": ev.magnet_id,
            "event_time": ev.event_time,
            "alert_time": None if first is None else first.alert_time,
            "lead_hours": lead,
            "lead_hours_unclamped": raw_lead,
            "detected": bool(len(cand)),
            "detected_in_time": bool(len(cand) and lead >= min_lead),
        })
    return pd.DataFrame(rows)


def false_alerts(episodes: pd.DataFrame, events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Episodes that never overlap any event's horizon."""
    horizon = pd.Timedelta(hours=cfg.window.horizon_hours)
    if not len(episodes):
        return episodes.assign(is_false=[])
    if not len(events):                     # no events in range -> every alert is false
        out = episodes.copy()
        out["is_false"] = True
        return out
    flags = []
    for ep in episodes.itertuples():
        in_horizon = ((events.magnet_id == ep.magnet_id)
                      & (events.event_time > ep.alert_time)
                      & (events.event_time - horizon <= ep.last_time)).any()
        flags.append(not in_horizon)
    out = episodes.copy()
    out["is_false"] = flags
    return out


# --------------------------------------------------------------------------------------
# threshold policy
# --------------------------------------------------------------------------------------
def sweep_thresholds(frame: pd.DataFrame, events: pd.DataFrame, cfg: Config
                     ) -> pd.DataFrame:
    """Operate-point trade-off on a validation frame."""
    scores = frame["score"].to_numpy()
    if len(scores) == 0:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1",
                                     "event_recall_in_time", "false_alerts",
                                     "false_per_magnet_month"])
    quantiles = np.arange(0.50, 0.999, 0.005)
    candidates = np.unique(np.round(np.concatenate(
        [[0.5, 0.9, 1.0], np.quantile(scores, quantiles)]), 6))
    mm = magnet_months(frame)
    rows = []
    for thr in candidates:
        ep = alert_episodes(frame, float(thr))
        false = false_alerts(ep, events, cfg)
        out = match_events(events, ep, cfg)
        y, s = frame["y"].to_numpy(), scores
        yhat = (s >= thr).astype(int)
        rows.append({
            "threshold": float(thr),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0),
            "event_recall_in_time": float(out.detected_in_time.mean()) if len(out) else 0.0,
            "n_alerts": len(ep),
            "false_alerts": int(false["is_false"].sum()) if len(false) else 0,
            "false_per_magnet_month": (float(false["is_false"].sum()) / mm) if mm else np.inf,
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def choose_threshold(sweep: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Highest event recall that still respects the (safety-tightened) budget."""
    raw_budget = cfg.threshold.false_alert_budget_per_magnet_month
    budget = raw_budget * cfg.threshold.selection_safety_factor
    if not len(sweep):
        return {"threshold": 0.5, "reason": "empty sweep", "budget": budget}
    feasible = sweep[sweep.false_per_magnet_month <= budget]
    if len(feasible):
        best = feasible.sort_values(
            ["event_recall_in_time", "f1", "threshold"],
            ascending=[False, False, False]).iloc[0]
        reason = (f"best lead-time recall among {len(feasible)} operating points with "
                  f"<= {budget:.2f} false pages/magnet-month "
                  f"({raw_budget:g} budget x {cfg.threshold.selection_safety_factor:g} safety)")
    else:  # never happens with well-behaved scores: fall back to the closest budget
        best = sweep.assign(gap=(sweep.false_per_magnet_month - budget).abs()).sort_values(
            ["gap", "event_recall_in_time"], ascending=[True, False]).iloc[0]
        reason = "no point met the tightened budget; picked the closest operating point"
    return {"threshold": float(best.threshold), "reason": reason,
            "budget": budget, "row": best.to_dict()}


# --------------------------------------------------------------------------------------
# metrics + cost
# --------------------------------------------------------------------------------------
def classification_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    yhat = (scores >= threshold).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    return {
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "pr_auc": float(average_precision_score(y, scores)) if len(y) else 0.0,
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else 0.0,
        "accuracy": float((yhat == y).mean()) if len(y) else 0.0,
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "prevalence": float(np.mean(y)) if len(y) else 0.0,
        "n_windows": int(len(y)),
    }


def cost_summary(outcomes: pd.DataFrame, n_false: int, cfg: Config) -> dict[str, Any]:
    c = cfg.cost
    n_events = len(outcomes)
    detected = int(outcomes.detected_in_time.sum()) if n_events else 0
    missed = n_events - detected
    reactive = n_events * c.emergency_total
    predictive = (missed * c.emergency_total + detected * c.planned_total
                  + n_false * c.false_alarm_usd)
    return {
        "n_events": n_events,
        "detected_in_time": detected,
        "missed": missed,
        "false_alerts": int(n_false),
        "cost_per_quench_usd": c.emergency_total,
        "cost_per_planned_repair_usd": c.planned_total,
        "cost_per_false_alarm_usd": c.false_alarm_usd,
        "reactive_total_usd": reactive,
        "predictive_total_usd": predictive,
        "net_savings_usd": reactive - predictive,
        "saving_per_detected_event_usd": c.saving_per_detected_event,
    }


def evaluate_split(frame: pd.DataFrame, events: pd.DataFrame, cfg: Config,
                   threshold: float) -> dict[str, Any]:
    """Everything measured on one split at a fixed operating point."""
    mm = magnet_months(frame)
    ep = alert_episodes(frame, threshold)
    false = false_alerts(ep, events, cfg)
    n_false = int(false["is_false"].sum()) if len(false) else 0
    outcomes = match_events(events, ep, cfg)
    lead = outcomes.loc[outcomes.detected, "lead_hours"]

    return {
        "split_size_windows": len(frame),
        "n_quench_events": len(events),
        "threshold": threshold,
        "magnet_months": round(mm, 3),
        "classification": classification_metrics(frame["y"].to_numpy(),
                                                 frame["score"].to_numpy(), threshold),
        "alerts": {
            "episodes": len(ep),
            "false_alerts": n_false,
            "false_per_magnet_month": round(n_false / mm, 3) if mm else None,
            "true_alerts": len(ep) - n_false,
        },
        "events": {
            "detected": int(outcomes.detected.sum()) if len(outcomes) else 0,
            "detected_in_time": int(outcomes.detected_in_time.sum()) if len(outcomes) else 0,
            "event_recall": float(outcomes.detected.mean()) if len(outcomes) else 0.0,
            "event_recall_in_time": float(outcomes.detected_in_time.mean()) if len(outcomes) else 0.0,
            "lead_hours_median": float(lead.median()) if len(lead) else None,
            "lead_hours_min": float(lead.min()) if len(lead) else None,
            "lead_hours_max": float(lead.max()) if len(lead) else None,
            "pct_lead_ge_72h": float((lead >= cfg.window.min_lead_hours).mean() * 100)
            if len(lead) else 0.0,
        },
        "cost": cost_summary(outcomes, n_false, cfg),
        "_episodes": ep,
        "_outcomes": outcomes,
    }


def write_metrics(payload: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, bool) or isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, int) or isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, float) or isinstance(o, (np.floating,)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, str):
            return o
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, pd.DataFrame):
            return clean(o.to_dict(orient="records"))
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return str(o)

    path.write_text(json.dumps(clean(payload), indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------------------
def _save(fig, fname: str, close: bool = False) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / fname
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    if close:
        plt.close(fig)
    return out


def plot_sensor_story(frame: pd.DataFrame, magnet: str, event_time: pd.Timestamp,
                      episodes: pd.DataFrame, cfg: Config, fname: str = "sensor_story.png",
                      close: bool = False) -> plt.Figure:
    """Four sensor panels around one quench, with the alert and the deadline."""
    plt.close("all")
    ph = cfg.physics
    horizon = pd.Timedelta(hours=cfg.window.horizon_hours)
    lead_deadline = pd.Timedelta(hours=cfg.window.min_lead_hours)
    g = frame[frame.magnet_id == magnet]
    g = g[(g.window_end >= event_time - horizon - pd.Timedelta(days=2))
          & (g.window_end <= event_time + pd.Timedelta(days=1))]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    ax = axes[0]
    ax.plot(g.window_end, g.p_mean, lw=1.4, color="#1f77b4")
    ax.axhline(ph.pressure_critical_mpa, color="crimson", ls="--", lw=1.2,
               label=f"failure signal < {ph.pressure_critical_mpa} MPa")
    ax.set_ylabel("Helium pressure (MPa)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(g.window_end, g.t_mean, lw=1.4, color="#ff7f0e")
    ax.axhline(ph.cold_head_critical_k, color="crimson", ls="--", lw=1.2,
               label=f"failure signal > {ph.cold_head_critical_k} K")
    ax.axhline(ph.cold_head_baseline_k, color="grey", ls=":", lw=1,
               label=f"baseline {ph.cold_head_baseline_k} K")
    ax.set_ylabel("Cold head (K)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(g.window_end, g.v_rms, lw=1.4, color="#2ca02c", label="RMS (g)")
    ax2 = ax.twinx()
    ax2.plot(g.window_end, g.v_fault_ratio, lw=1.0, color="#9467bd", alpha=0.8,
             label="fault-band ratio")
    ax2.set_ylabel("fault-band ratio")
    ax.set_ylabel("Vibration RMS (g)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    ax.plot(g.window_end, g.dropout_count, lw=1.2, color="#8c564b", label="dropouts/window")
    ax.set_ylabel("RF dropouts")
    ax.set_xlabel("window close")
    ax.legend(loc="upper right", fontsize=8)

    for a in axes:
        a.axvspan(event_time - horizon, event_time - lead_deadline, color="orange",
                  alpha=0.10, lw=0)
        a.axvline(event_time, color="crimson", lw=1.6)
    for a in axes:
        a.axvline(event_time - lead_deadline, color="darkorange", ls="--", lw=1.2)
    if len(episodes):
        for _, ep in episodes[episodes.magnet_id == magnet].iterrows():
            for a in axes:
                a.axvline(ep.alert_time, color="green", ls="--", lw=1.4)
    fig.suptitle(f"{magnet} — quench at {event_time:%Y-%m-%d %H:%M} | "
                 f"orange band = 96 h horizon, orange line = 72 h deadline, "
                 f"green = model alert", y=1.00)
    return _save(fig, fname, close)


def plot_curves(y: np.ndarray, score_sets: dict[str, np.ndarray],
                threshold: float, fname: str = "pr_roc_curves.png",
                close: bool = False) -> plt.Figure:
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, s in score_sets.items():
        p, r, _ = _pr_points(y, s)
        axes[0].plot(r, p, lw=1.8, label=f"{name} (AP {average_precision_score(y, s):.3f})")
        fpr, tpr, _ = roc_curve(y, s)
        axes[1].plot(fpr, tpr, lw=1.8,
                     label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
    yhat = (score_sets[list(score_sets)[0]] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    if tp + fp:
        axes[0].scatter([tp / (tp + fn)], [tp / (tp + fp)], marker="*", s=220,
                        color="black", zorder=5, label="operating point")
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision")
    axes[0].set_title("Precision–Recall"); axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], ls="--", color="grey")
    axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate")
    axes[1].set_title("ROC"); axes[1].legend(fontsize=8)
    return _save(fig, fname, close)


def _pr_points(y: np.ndarray, scores: np.ndarray):
    from sklearn.metrics import precision_recall_curve
    return precision_recall_curve(y, scores)


def plot_lead_times(outcomes: pd.DataFrame, min_lead: int, horizon: int = 96,
                    fname: str = "lead_times.png", close: bool = False) -> plt.Figure:
    """First-warning lead time per event (unclamped), with the acceptance line."""
    plt.close("all")
    col = "lead_hours_unclamped" if "lead_hours_unclamped" in outcomes.columns else "lead_hours"
    fig, ax = plt.subplots(figsize=(9, 5))
    lead = outcomes.loc[outcomes.detected, col].dropna()
    if len(lead):
        ax.hist(lead, bins=12, color="#4c72b0", edgecolor="white")
        ax.axvline(min_lead, color="crimson", ls="--", lw=2,
                   label=f"required {min_lead} h")
        ax.axvline(horizon, color="darkorange", ls=":", lw=2,
                   label=f"label horizon {horizon} h")
        ax.axvline(lead.median(), color="black", ls=":", lw=2,
                   label=f"median {lead.median():.1f} h")
        for x in lead:
            ax.annotate(f"{x:.0f}h", (x, ax.get_ylim()[1] * 0.92), ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no alerts", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("lead time before quench (hours)")
    ax.set_ylabel("number of events")
    ax.set_title("Detection lead time per quench event (first warning)")
    ax.legend(fontsize=9)
    return _save(fig, fname, close)


def plot_importance(imp: pd.DataFrame, fname: str = "feature_importance.png",
                    close: bool = False) -> plt.Figure:
    plt.close("all")
    top = imp.head(15).iloc[::-1]
    group = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    colors = sns.color_palette("Set2", n_colors=len(FEATURE_GROUPS))
    palette = {g: colors[i] for i, g in enumerate(FEATURE_GROUPS)}
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature, top.importance,
            color=[palette.get(group.get(f, ""), "grey") for f in top.feature])
    ax.set_xlabel("mean |gradient × input| (standardised units)")
    ax.set_title("What drives the alert — top 15 features")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, list(FEATURE_GROUPS), fontsize=8, loc="lower right")
    return _save(fig, fname, close)


def plot_cost(cost: dict, fname: str = "cost_comparison.png",
              close: bool = False) -> plt.Figure:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["reactive\n(fix after quench)", "predictive\n(plan the failure)"]
    vals = [cost["reactive_total_usd"], cost["predictive_total_usd"]]
    bars = ax.bar(labels, vals, color=["#c44e52", "#55a868"], width=0.55)
    for b, v in zip(bars, vals):
        ax.annotate(f"${v:,.0f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("test-period cost (USD)")
    ax.set_title(f"Cost of the test period — net saving ${cost['net_savings_usd']:,.0f}")
    return _save(fig, fname, close)


def plot_history(history: list[dict], fname: str = "training_history.png",
                 close: bool = False) -> plt.Figure:
    plt.close("all")
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(df.epoch, df.train_loss, marker="o", label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss"); axes[0].legend()
    axes[0].set_title("Training loss")
    axes[1].plot(df.epoch, df.val_ap, marker="o", label="val PR-AUC")
    axes[1].plot(df.epoch, df.val_roc, marker="s", label="val ROC-AUC")
    axes[1].axhline(df.val_ap.max(), color="grey", ls=":", lw=1)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score"); axes[1].legend()
    axes[1].set_title("Validation metrics (early-stopping signal)")
    return _save(fig, fname, close)
