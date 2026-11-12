"""Feature-drift monitoring with Evidently (PSI) — the retraining trigger.

Scope (matches the retraining policy in ``docs/model_card.md``):

* **Reference** = the training split of the gold feature frame — the
  distribution the model was actually fitted on.
* **Current** = the most recent hold-out slice (or any new batch supplied by
  the caller), i.e. "what the model is being asked to score today".
* **Metric** = Population Stability Index per feature, computed by Evidently's
  ``DataDriftPreset(num_method="psi")``; a feature is *drifted* when PSI > 0.2.
* **Verdict** = retraining trigger when any *key* feature (top-8 by SHAP
  importance, read from ``metrics.json``) drifts, or when more than 25 % of
  all features drift.

Outputs: ``reports/drift.json`` (machine-readable verdict) and
``reports/drift_report.html`` (the Evidently dashboard for humans).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, default_config

PSI_ALERT = 0.2          # model card retraining policy
SHARE_ALERT = 0.25       # >25 % of features drifted -> fleet-wide shift
KEY_FEATURES_TOP_N = 8

_DRIFTED_DISPLAY = "Value drift for "


def _drift_snapshot(reference: pd.DataFrame, current: pd.DataFrame,
                    columns: list[str]) -> tuple[dict[str, float], float, float]:
    """Run Evidently's PSI preset; returns (per-column PSI, drifted count, drifted share).

    Raises ValueError if a column is missing or data is non-numeric.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset

    missing = [c for c in columns if c not in reference.columns or c not in current.columns]
    if missing:
        raise ValueError(f"columns missing from dataframes: {missing}")
    ref, cur = reference[columns], current[columns]
    if not all(np.issubdtype(t, np.number) for t in ref.dtypes):
        raise ValueError("drift check expects numeric feature columns only")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = Report(metrics=[DataDriftPreset(
            columns=columns, num_method="psi", threshold=PSI_ALERT,
            include_tests=False)])
        snap = report.run(reference_data=ref, current_data=cur)

    psi: dict[str, float] = {}
    drifted_count, drifted_share = 0.0, 0.0
    for value in snap.metric_results.values():
        name = value.display_name
        if name.startswith(_DRIFTED_DISPLAY):
            col = name[len(_DRIFTED_DISPLAY):]
            psi[col] = float(value.to_simple_dict())
        elif "Drifted Columns" in name:
            d = value.to_simple_dict()
            drifted_count = float(d["count"])
            drifted_share = float(d["share"])
    missing_psi = [c for c in columns if c not in psi]
    if missing_psi:
        raise RuntimeError(f"Evidently did not return PSI for {missing_psi}")
    return psi, drifted_count, drifted_share


def key_features(metrics_path: Path | None = None) -> list[str]:
    """Top-N features by SHAP importance from ``metrics.json`` (or [] if absent)."""
    path = metrics_path or default_config().metrics_path
    try:
        shap_block = json.loads(path.read_text(encoding="utf-8"))["shap"]
    except (OSError, KeyError, json.JSONDecodeError):
        return []
    ranked = sorted(shap_block.get("global_importance", []),
                    key=lambda r: r["mean_abs_shap"], reverse=True)
    return [r["feature"] for r in ranked[:KEY_FEATURES_TOP_N]]


def drift_verdict(psi: dict[str, float], keys: list[str],
                  drifted_share: float) -> dict[str, Any]:
    """The retraining decision: any key feature drifted, or a fleet-wide shift."""
    drifted = sorted((c for c, v in psi.items() if v > PSI_ALERT),
                     key=lambda c: psi[c], reverse=True)
    key_drifted = sorted(set(drifted) & set(keys), key=lambda c: psi[c], reverse=True)
    retrain = bool(key_drifted) or drifted_share > SHARE_ALERT
    return {
        "retrain_recommended": retrain,
        "reason": ("key feature(s) drifted: " + ", ".join(key_drifted)
                   if key_drifted else
                   f"drifted share {drifted_share:.0%} > {SHARE_ALERT:.0%}"
                   if drifted_share > SHARE_ALERT else "within thresholds"),
        "psi_alert": PSI_ALERT,
        "key_features": keys,
        "key_drifted": key_drifted,
        "drifted_features": drifted,
        "drifted_share": drifted_share,
        "psi": {k: round(v, 6) for k, v in sorted(psi.items(),
                                                  key=lambda kv: kv[1], reverse=True)},
    }


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                       columns: list[str], html_path: Path | None = None) -> Path:
    """Save the human-readable Evidently dashboard."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    html_path = html_path or (REPORTS_DIR / "drift_report.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = Report(metrics=[DataDriftPreset(
            columns=columns, num_method="psi", threshold=PSI_ALERT,
            include_tests=False)])
        snap = report.run(reference_data=reference[columns],
                          current_data=current[columns])
        snap.save_html(str(html_path))
    return html_path


def run_drift_check(cfg: Any | None = None,
                    reference: pd.DataFrame | None = None,
                    current: pd.DataFrame | None = None,
                    columns: list[str] | None = None,
                    write_reports: bool = True,
                    verbose: bool = True) -> dict[str, Any]:
    """Full drift check: reference vs current feature frame.

    Defaults: reference = train split, current = test split (most recent 20 %).
    Returns the verdict dict; optionally writes ``reports/drift.json`` +
    ``reports/drift_report.html``.
    """
    from .features import FEATURE_COLUMNS
    from .sequences import prepare_data

    cfg = cfg or default_config()
    if reference is None or current is None:
        _splits, _scaler, labelled = prepare_data(cfg, verbose=False)
        reference = labelled.loc[labelled.split == "train", FEATURE_COLUMNS]
        current = labelled.loc[labelled.split == "test", FEATURE_COLUMNS]
    if columns is None:
        shared = [c for c in reference.columns if c in current.columns]
        cols = [c for c in shared
                if np.issubdtype(reference[c].dtype, np.number)
                and np.issubdtype(current[c].dtype, np.number)]
    else:
        cols = columns
    if not cols:
        raise ValueError("no shared numeric feature columns")

    if verbose:
        print(f"  reference rows: {len(reference):,} | current rows: {len(current):,} "
              f"| features: {len(cols)}")
    psi, _count, share = _drift_snapshot(reference, current, cols)
    keys = key_features()
    verdict = drift_verdict(psi, keys, share)

    if write_reports:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "drift.json").write_text(
            json.dumps(verdict, indent=2), encoding="utf-8")
        build_drift_report(reference, current, cols)
    if verbose:
        top = verdict["psi"]
        worst = list(top.items())[:3]
        print(f"  drifted: {len(verdict['drifted_features'])}/{len(cols)} "
              f"(share {share:.0%}) | worst PSI: " +
              ", ".join(f"{c}={v:.3f}" for c, v in worst))
        print(f"  verdict: {'RETRAIN RECOMMENDED' if verdict['retrain_recommended'] else 'stable'} "
              f"({verdict['reason']})")
    return verdict


__all__ = ["run_drift_check", "drift_verdict", "build_drift_report",
           "key_features", "PSI_ALERT", "SHARE_ALERT"]
