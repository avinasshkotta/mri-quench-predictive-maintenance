"""Fleet fairness audit (Fairlearn) — does the detector protect every magnet?

Scope and honesty notes
-----------------------
* The sensitive feature available in this study is ``magnet_id`` (the fleet's
  six units). In a real deployment the audit extends to vendor, install year
  and site — the harness takes any column, so that is a data-schema change,
  not a code change.
* Per-magnet event counts in the test slice are tiny (4 events fleet-wide), so
  per-group recall is *noisy*: we therefore audit two things separately —
  (a) **safety equity**: every magnet that HAS a test event must have been
  detected at or above its frozen threshold (recall = 1.0 per event-bearing
  magnet), and (b) **alert-rate equity**: selection-rate disparity across
  magnets (max/min ratio) must stay within FAIRNESS_POLICY bounds.
* Metrics are computed with Fairlearn's ``MetricFrame`` so the by-group table
  is the standard artifact reviewers expect.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, default_config

# Governance policy (documented; challengeable numbers live in one place)
MAX_SELECTION_RATE_RATIO = 3.0     # among alerting magnets (undefined vs zero)
MAX_SELECTION_RATE_DIFFERENCE = 0.20   # demographic-parity difference (20 pts)
MAX_FPR_DIFFERENCE = 0.10          # false-page burden spread across the fleet
EVENT_RECALL_REQUIREMENT = 1.0     # every test event must be detected, per magnet
WINDOW_RECALL_WARN = 0.90          # window-level recall below this -> warn (noise)
WINDOW_RECALL_FAIL = 0.50          # window-level recall below this -> fail
PR_AUC_SPREAD_WARN = 0.10          # informational: per-magnet PR-AUC spread


def group_metrics(frame: pd.DataFrame, threshold: float,
                  extra_metrics: dict | None = None) -> dict[str, Any]:
    """Fairlearn MetricFrame over ``magnet_id``: selection rate, TPR, FPR, count."""
    from fairlearn.metrics import (MetricFrame, count, false_positive_rate,
                                   selection_rate, true_positive_rate)

    y_true = frame["y"].to_numpy(int)
    y_pred = (frame["score"].to_numpy(float) >= threshold).astype(int)
    groups = frame["magnet_id"].to_numpy()
    metrics = {
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "count": count,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred,
                         sensitive_features=groups)
    return {"overall": _clean(mf.overall), "by_group": _clean(mf.by_group)}


def _clean(obj: Any) -> Any:
    """MetricFrame results (Series / DataFrame / scalars) -> JSON-able dicts."""
    if isinstance(obj, pd.DataFrame):
        return {str(k): _clean(v) for k, v in obj.to_dict(orient="index").items()}
    if isinstance(obj, pd.Series):
        return {str(k): (None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def per_magnet_pr_auc(frame: pd.DataFrame) -> dict[str, float | None]:
    """AP per magnet; None where a magnet has no positive windows."""
    from sklearn.metrics import average_precision_score

    out: dict[str, float | None] = {}
    for magnet, g in frame.groupby("magnet_id"):
        if g["y"].nunique() < 2:
            out[str(magnet)] = None
        else:
            out[str(magnet)] = float(average_precision_score(g["y"], g["score"]))
    return out


def audit_verdict(by_group: dict[str, dict], frame: pd.DataFrame,
                  threshold: float,
                  events: pd.DataFrame | None = None,
                  horizon_hours: float = 96.0,
                  min_lead_hours: float = 72.0) -> dict[str, Any]:
    """Apply the fairness policy; returns a verdict dict (pass/warn/fail).

    ``events`` (quench events with ``magnet_id``/``event_time``) enables the
    stronger, event-level safety-equity check; without it the verdict falls
    back to window-level recall with a noise tolerance.
    """
    findings: list[str] = []
    severity = "pass"

    # (a) safety equity — EVENT-level recall per magnet when events are given
    if events is not None:
        ev = _event_recall_by_magnet(frame, events, threshold, horizon_hours,
                                     min_lead_hours)
    else:
        ev = {}
    recalls = _window_recall_by_magnet(frame, threshold)

    for magnet, recall in ev.items():
        if recall is not None and recall < EVENT_RECALL_REQUIREMENT:
            findings.append(f"{magnet}: event recall {recall:.0%} < "
                            f"{EVENT_RECALL_REQUIREMENT:.0%} — magnet under-protected")
            severity = "fail"
    for magnet, recall in recalls.items():
        if recall is not None and magnet not in ev:
            if recall < WINDOW_RECALL_FAIL:
                findings.append(f"{magnet}: window recall {recall:.0%} < "
                                f"{WINDOW_RECALL_FAIL:.0%} — systematically missing "
                                "positive windows (no events to cross-check)")
                severity = "fail"
            elif recall < WINDOW_RECALL_WARN:
                findings.append(f"{magnet}: window recall {recall:.0%} < "
                                f"{WINDOW_RECALL_WARN:.0%}")
                severity = "warn" if severity == "pass" else severity

    # (b) alert-rate equity — disparity among magnets that actually alert.
    # Clean magnets that never page are *correct*, so a max/min ratio against
    # zero is undefined; parity is only meaningful among alerting units.
    rates = {m: v["selection_rate"] for m, v in by_group.items()
             if v.get("selection_rate") is not None and v["selection_rate"] > 0}
    if len(rates) >= 2:
        diff = float(max(rates.values()) - min(rates.values()))
        ratio = float(max(rates.values()) / min(rates.values()))
    else:
        diff, ratio = 0.0, 1.0
    if diff > MAX_SELECTION_RATE_DIFFERENCE:
        findings.append(f"selection-rate difference {diff:.3f} > "
                        f"{MAX_SELECTION_RATE_DIFFERENCE:g} — alert burden uneven")
        severity = "fail" if severity == "fail" else "warn"
    elif ratio > MAX_SELECTION_RATE_RATIO:
        findings.append(f"selection-rate ratio {ratio:.2f} > "
                        f"{MAX_SELECTION_RATE_RATIO:g} — alert burden uneven")
        severity = "fail" if severity == "fail" else "warn"

    # (c) false-page burden — FPR spread across the whole fleet (zeros count)
    fprs = {m: v["false_positive_rate"] for m, v in by_group.items()
            if v.get("false_positive_rate") is not None}
    fpr_diff = float(max(fprs.values()) - min(fprs.values())) if fprs else 0.0
    if fpr_diff > MAX_FPR_DIFFERENCE:
        findings.append(f"false-positive-rate difference {fpr_diff:.3f} > "
                        f"{MAX_FPR_DIFFERENCE:g} — paging burden uneven")
        severity = "fail" if severity == "fail" else "warn"

    # informational: per-magnet PR-AUC spread
    aps = {m: v for m, v in per_magnet_pr_auc(frame).items() if v is not None}
    spread = (max(aps.values()) - min(aps.values())) if len(aps) >= 2 else 0.0
    if spread > PR_AUC_SPREAD_WARN:
        findings.append(f"per-magnet PR-AUC spread {spread:.3f} > "
                        f"{PR_AUC_SPREAD_WARN} (informational)")
        severity = "warn" if severity == "pass" else severity

    return {
        "verdict": severity,
        "findings": findings or ["no fairness policy breach detected"],
        "event_recall_by_magnet": ev if events is not None else recalls,
        "window_recall_by_magnet": recalls,
        "n_alerting_magnets": len(rates),
        "selection_rate_difference": round(diff, 4),
        "selection_rate_ratio": round(ratio, 4),
        "max_selection_rate_difference": MAX_SELECTION_RATE_DIFFERENCE,
        "max_selection_rate_ratio": MAX_SELECTION_RATE_RATIO,
        "fpr_difference": round(fpr_diff, 4),
        "max_fpr_difference": MAX_FPR_DIFFERENCE,
        "event_recall_requirement": EVENT_RECALL_REQUIREMENT,
        "pr_auc_by_magnet": {k: (round(v, 4) if v is not None else None)
                             for k, v in per_magnet_pr_auc(frame).items()},
        "pr_auc_spread": round(float(spread), 4),
    }


def _window_recall_by_magnet(frame: pd.DataFrame,
                             threshold: float) -> dict[str, float | None]:
    """Window-level recall per magnet; None where a magnet has no positives."""
    out: dict[str, float | None] = {}
    for magnet, g in frame.groupby("magnet_id"):
        if g["y"].sum() == 0:
            out[str(magnet)] = None
            continue
        pred = (g["score"].to_numpy() >= threshold).astype(int)
        out[str(magnet)] = float((pred[g["y"].to_numpy() == 1]).mean())
    return out


def _event_recall_by_magnet(frame: pd.DataFrame, events: pd.DataFrame,
                            threshold: float, horizon_hours: float,
                            min_lead_hours: float = 72.0
                            ) -> dict[str, float | None]:
    """Event-level recall per magnet, using the *pipeline's* matching rule.

    Delegates to :func:`quenchguard.evaluate.match_events` so the fairness
    audit can never disagree with the published metrics: an episode counts as
    a hit when it is active inside the horizon (``alert_time < event_time`` and
    ``last_time >= event_time - horizon``), and lead time is clamped to the
    horizon. An early alert that persists into the horizon is therefore a
    detection with ``horizon_hours`` of lead — not a miss. Recall here is the
    fraction of events detected **in time** (lead >= ``min_lead_hours``).

    Magnets without test events -> ``None`` (not applicable, not a failure).
    """
    from types import SimpleNamespace

    from .evaluate import alert_episodes, match_events

    episodes = alert_episodes(frame, threshold)
    t_min, t_max = frame["t"].min(), frame["t"].max()
    in_range = events[(events.event_time >= t_min)
                      & (events.event_time <= t_max)]
    cfg_like = SimpleNamespace(window=SimpleNamespace(
        horizon_hours=horizon_hours, min_lead_hours=min_lead_hours))
    outcomes = match_events(in_range, episodes, cfg_like)

    out: dict[str, float | None] = {}
    for magnet in frame["magnet_id"].unique():
        rows = outcomes[outcomes.magnet_id == magnet] if len(outcomes) else outcomes
        out[str(magnet)] = (float(rows.detected_in_time.mean())
                            if len(rows) else None)
    return out


def run_fairness_audit(cfg: Any | None = None, frame: pd.DataFrame | None = None,
                       threshold: float | None = None,
                       events: pd.DataFrame | None = None,
                       write_reports: bool = True,
                       verbose: bool = True) -> dict[str, Any]:
    """Full audit: champion-LSTM scores on the test split, grouped by magnet.

    Pass a pre-computed ``frame`` (``magnet_id``/``y``/``score``) for tests or
    offline analysis; pass ``events`` to enable event-level safety equity.
    Otherwise the model + test split + quench events are loaded/built.
    """
    cfg = cfg or default_config()
    if frame is None:
        import torch

        from .evaluate import make_score_frame
        from .lstm import load_model, predict_scores
        from .sequences import prepare_data

        payload = __import__("torch").load(cfg.model_path, map_location="cpu",
                                           weights_only=False)
        threshold = float(threshold if threshold is not None
                          else payload.get("meta", {}).get("threshold", 0.5))
        splits, _scaler, _labelled = prepare_data(cfg, verbose=False)
        model = load_model(cfg.model_path, device=torch.device("cpu"))
        scores = predict_scores(model, splits["test"].X, torch.device("cpu"))
        frame = make_score_frame(splits["test"], scores, "test")
        if events is None:
            from .data_gen import load_quench_events

            events = load_quench_events(cfg)

    result = group_metrics(frame, threshold)
    verdict = audit_verdict(result["by_group"], frame, threshold, events=events,
                            horizon_hours=cfg.window.horizon_hours,
                            min_lead_hours=cfg.window.min_lead_hours)
    audit = {
        "threshold": float(threshold),
        "n_windows": int(len(frame)),
        "n_magnets": int(frame.magnet_id.nunique()),
        "metrics": result,
        **verdict,
        "notes": [
            "Sensitive feature: magnet_id (the only grouping in this study).",
            "In production extend to vendor / install year / site — the audit "
            "takes any column.",
            "Safety equity uses the pipeline's own event matching "
            "(evaluate.match_events: episode active inside the 96 h horizon, "
            "lead clamped to the horizon, detected_in_time requires >= 72 h), "
            "so the audit can never disagree with the published metrics.",
            "Per-magnet recall is noisy with 4 test events; the policy therefore "
            "requires recall=1.0 per event-bearing magnet instead of averaging.",
        ],
    }
    if write_reports:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "fairness.json").write_text(json.dumps(audit, indent=2),
                                                   encoding="utf-8")
    if verbose:
        print(f"  magnets: {audit['n_magnets']} | windows: {audit['n_windows']:,} "
              f"| threshold {threshold:.3f}")
        per_event = ", ".join(
            f"{m}={v:.0%}" for m, v in audit["event_recall_by_magnet"].items()
            if v is not None)
        print("  event recall (>=72 h lead) per event-bearing magnet: "
              + (per_event or "(no test events)"))
        print(f"  alert-burden disparity: selection diff "
              f"{audit['selection_rate_difference']:.3f} "
              f"(max {MAX_SELECTION_RATE_DIFFERENCE:g}), ratio "
              f"{audit['selection_rate_ratio']:.2f} "
              f"(max {MAX_SELECTION_RATE_RATIO:g}), FPR diff "
              f"{audit['fpr_difference']:.3f} (max {MAX_FPR_DIFFERENCE:g})")
        print(f"  verdict: {audit['verdict'].upper()} — {audit['findings'][0]}")
    return audit


__all__ = ["run_fairness_audit", "group_metrics", "audit_verdict",
           "per_magnet_pr_auc", "MAX_SELECTION_RATE_RATIO",
           "EVENT_RECALL_REQUIREMENT", "WINDOW_RECALL_WARN", "WINDOW_RECALL_FAIL"]
