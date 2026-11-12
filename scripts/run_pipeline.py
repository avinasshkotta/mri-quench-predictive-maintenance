#!/usr/bin/env python
"""End-to-end QuenchGuard pipeline: data -> features -> model -> report.

Usage
-----
    python scripts/run_pipeline.py                  # full run (generates data if needed)
    python scripts/run_pipeline.py --regenerate     # force fresh synthetic data
    python scripts/run_pipeline.py --epochs 8      # quick run
    python scripts/run_pipeline.py --device cpu --no-plots

Outputs
-------
    data/raw/*.parquet          synthetic multi-rate sensor streams (BRONZE)
    data/silver/*.parquet       cleaned, quality-gated sensor tables (SILVER)
    data/gold/*.parquet         engineered features (GOLD)
    data/lineage/*.json         per-layer checksums + schema (data lineage)
    models/lstm_detector.pt     trained LSTM
    reports/metrics.json        every metric quoted in the README
    reports/threshold_sweep.csv validation operating-point sweep
    reports/figures/*.png       plots used by the notebook / report
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quenchguard.baselines import baseline_scores, train_baselines            # noqa: E402
from quenchguard.config import REPORTS_DIR, default_config                    # noqa: E402
from quenchguard.data_gen import generate_fleet, load_quench_events           # noqa: E402
from quenchguard.evaluate import (choose_threshold, evaluate_split,          # noqa: E402
                                  make_score_frame, plot_cost, plot_curves,
                                  plot_history, plot_importance, plot_lead_times,
                                  plot_sensor_story, sweep_thresholds,
                                  write_metrics)
from quenchguard.features import FEATURE_COLUMNS                            # noqa: E402
from quenchguard.lstm import (device_summary, gradient_importance,           # noqa: E402
                              predict_scores, save_model, train_lstm)
from quenchguard.medallion import run_medallion                             # noqa: E402
from quenchguard.sequences import prepare_data                               # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerate", action="store_true", help="rebuild synthetic data")
    ap.add_argument("--epochs", type=int, default=None, help="override training epochs")
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-plots", action="store_true", help="skip figure generation")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    cfg = default_config()
    overrides = {k: v for k, v in
                 {"epochs": args.epochs, "device": args.device}.items() if v is not None}
    if overrides:
        cfg = replace(cfg, train=replace(cfg.train, **overrides))

    print("=" * 78)
    print("QuenchGuard | LSTM predictive maintenance for MRI magnet quench")
    print("=" * 78)
    print(f"device        : {device_summary(cfg)}")
    print(f"window        : {cfg.window.step_minutes} min steps, {cfg.window.seq_len} "
          f"steps = {cfg.window.seq_len * cfg.window.step_minutes // 60} h look-back")
    print(f"label horizon : {cfg.window.horizon_hours} h "
          f"(accept lead >= {cfg.window.min_lead_hours} h)")
    print(f"cost model    : quench ${cfg.cost.emergency_total:,.0f} vs "
          f"planned ${cfg.cost.planned_total:,.0f} vs "
          f"false alarm ${cfg.cost.false_alarm_usd:,.0f}")

    # 1 -- data ----------------------------------------------------------------------
    print("\n[1/7] synthetic sensor data")
    if args.regenerate:
        cfg.features_path.unlink(missing_ok=True)
        cfg.dataset_path.unlink(missing_ok=True)
    if args.regenerate or not cfg.pressure_path.exists():
        generate_fleet(cfg)
    else:
        print("  raw data present (use --regenerate to rebuild)")

    # 2 -- medallion (bronze -> silver quality gates -> gold features) --------------
    print("\n[2/7] medallion: bronze -> silver (Great Expectations gates) -> gold")
    run_medallion(cfg, verbose=True)

    # 3 -- labels + sequences --------------------------------------------------------
    print("\n[3/7] labelling, time-ordered split, sequence construction")
    splits, _scaler, labelled = prepare_data(cfg)
    events = load_quench_events(cfg)

    # 4 -- baselines -----------------------------------------------------------------
    print("\n[4/7] baselines (the LSTM must beat these)")
    models = train_baselines(cfg, splits["train"].X, splits["train"].y,
                             splits["val"].X, splits["val"].y)

    # 5 -- LSTM ----------------------------------------------------------------------
    print("\n[5/7] LSTM training")
    model, history, device = train_lstm(cfg, splits)

    val_scores = {"lstm": predict_scores(model, splits["val"].X, device),
                  **baseline_scores(models, splits["val"].X)}
    test_scores = {"lstm": predict_scores(model, splits["test"].X, device),
                   **baseline_scores(models, splits["test"].X)}
    test_frame = make_score_frame(splits["test"], test_scores["lstm"], "test")
    val_frame = make_score_frame(splits["val"], val_scores["lstm"], "val")

    # 6 -- one operating point per model, selected by the same policy -------------
    print("\n[6/7] threshold selection on validation (false-page budget)")
    val_events = events[(events.event_time >= val_frame.t.min())
                        & (events.event_time <= val_frame.t.max())]
    thresholds: dict[str, float] = {}
    choices: dict[str, dict] = {}
    sweeps = {}
    for name, s in val_scores.items():
        frame = make_score_frame(splits["val"], s, "val")
        sweep = sweep_thresholds(frame, val_events, cfg)
        choice = choose_threshold(sweep, cfg)
        thresholds[name] = float(choice["threshold"])
        choices[name], sweeps[name] = choice, sweep
        print(f"  {name:<10} threshold = {thresholds[name]:.3f}  ({choice['reason']})")

    # 7 -- evaluation ----------------------------------------------------------------
    print("\n[7/7] test-set evaluation")
    test_events = events[(events.event_time >= test_frame.t.min())
                         & (events.event_time <= test_frame.t.max())]
    report = {}
    for name, s in test_scores.items():
        frame = make_score_frame(splits["test"], s, "test")
        report[name] = evaluate_split(frame, test_events, cfg, thresholds[name])

    imp = gradient_importance(model, splits["test"].X, FEATURE_COLUMNS, device)

    # 8 -- artifacts -----------------------------------------------------------------
    print("\n[artifacts] saving model, figures, metrics")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        outcomes = report["lstm"]["_outcomes"]
        episodes = report["lstm"]["_episodes"]
        plot_history(history)
        plot_curves(splits["test"].y, test_scores, thresholds["lstm"])
        if len(outcomes):
            earliest = outcomes.sort_values("lead_hours", na_position="first").iloc[0]
            plot_sensor_story(labelled, earliest.magnet_id, earliest.event_time,
                              episodes, cfg)
            plot_lead_times(outcomes, cfg.window.min_lead_hours)
        plot_importance(imp)
        plot_cost(report["lstm"]["cost"])

    save_model(model, cfg.model_path, meta={
        "threshold": thresholds["lstm"], "features": FEATURE_COLUMNS,
        "seq_len": cfg.window.seq_len, "horizon_hours": cfg.window.horizon_hours,
        "val_pr_auc": float(max(h["val_ap"] for h in history)),
    })
    # Serving artifacts (Phase C): the scaler must travel with the model or the
    # API would score unscaled inputs (training/serving skew), and the HistGBDT
    # is what produces SHAP explanations at request time.
    import joblib

    from quenchguard.config import MODELS_DIR
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(_scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(models["hist_gbdt"], MODELS_DIR / "hist_gbdt.joblib")
    if sweeps:
        sweep_all = pd.concat(
            [sw.assign(model=name) for name, sw in sweeps.items()], ignore_index=True)
        sweep_all.to_csv(REPORTS_DIR / "threshold_sweep.csv", index=False)
    pd.DataFrame(history).to_csv(REPORTS_DIR / "training_history.csv", index=False)

    # 8 -- XAI: SHAP on the interpretable baseline ------------------------------------
    print("\n[xai] SHAP explanations (TreeExplainer on hist_gbdt)")
    import numpy as np

    from quenchguard.baselines import last_step
    from quenchguard.explain import explain_case

    X_test_last = last_step(splits["test"].X)
    test_proba = models["hist_gbdt"].predict_proba(X_test_last)[:, 1]
    shap_case = int(np.argmax(test_proba))   # explain the most confident positive
    shap_info = explain_case(
        models["hist_gbdt"], X_test_last, FEATURE_COLUMNS,
        case_index=shap_case, figures_dir=REPORTS_DIR / "figures", prefix="shap",
    )
    print("  top factors for the highest-risk test window: " + ", ".join(
        f"{f['feature']} ({f['shap_value']:+.3f}, {f['direction']})"
        for f in shap_info["top_factors"]))

    payload = {
        "config": {
            "sequence_length": cfg.window.seq_len,
            "step_minutes": cfg.window.step_minutes,
            "horizon_hours": cfg.window.horizon_hours,
            "min_lead_hours": cfg.window.min_lead_hours,
            "thresholds": thresholds,
            "threshold_reason": {k: v.get("reason") for k, v in choices.items()},
            "false_alert_budget": cfg.threshold.false_alert_budget_per_magnet_month,
            "selection_safety_factor": cfg.threshold.selection_safety_factor,
            "cost": {"emergency_total_usd": cfg.cost.emergency_total,
                     "planned_total_usd": cfg.cost.planned_total,
                     "false_alarm_usd": cfg.cost.false_alarm_usd},
        },
        "results": report,
        "importance": imp.head(15).to_dict(orient="records"),
        "shap": shap_info,
        "training": {"best_val_pr_auc": float(max(h["val_ap"] for h in history)),
                     "epochs_run": len(history)},
    }
    path = write_metrics(payload, cfg.metrics_path)

    # 9 -- MLflow tracking + model registry -------------------------------------------
    print("\n[mlflow] logging experiment run + registering champion model")
    from quenchguard.tracking import log_run

    run_info = log_run(cfg=cfg, report=report, thresholds=thresholds,
                       history=history, model=model)
    if run_info:
        print(f"  run_id={run_info['run_id']}  model={run_info['model_uri']}")
    else:
        print("  (tracking skipped -- see warning above)")

    print("\n" + "=" * 78)
    print(f"{'model':<12} {'PR-AUC':>7} {'prec':>6} {'rec':>6} "
          f"{'event>=72h':>11} {'lead med':>9} {'false/mo':>9} {'net save':>10}")
    for name, res in report.items():
        ev, al, cl = res["events"], res["alerts"], res["classification"]
        lead = ev["lead_hours_median"]
        print(f"{name:<12} {cl['pr_auc']:>7.3f} {cl['precision']:>6.3f} "
              f"{cl['recall']:>6.3f} {ev['event_recall_in_time']:>10.0%} "
              f"{(f'{lead:.1f} h' if lead is not None else '-'):>9} "
              f"{al['false_per_magnet_month']:>9.2f} "
              f"${res['cost']['net_savings_usd']:>9,.0f}")
    print("-" * 78)
    lstm = report["lstm"]
    print(f"acceptance   : event recall >= 72 h  = "
          f"{lstm['events']['event_recall_in_time']:.0%} "
          f"({lstm['events']['detected_in_time']}/{lstm['n_quench_events']}) | "
          f"false alerts = {lstm['alerts']['false_per_magnet_month']}/magnet-month "
          f"(budget {cfg.threshold.false_alert_budget_per_magnet_month:g})")
    print(f"lead times   : {', '.join(f'{v:.1f} h' for v in sorted(lstm['_outcomes'].lead_hours.dropna())) or 'none'}")
    print(f"cost         : reactive ${lstm['cost']['reactive_total_usd']:,.0f} -> "
          f"predictive ${lstm['cost']['predictive_total_usd']:,.0f} "
          f"(saving ${lstm['cost']['net_savings_usd']:,.0f})")
    print(f"\nwrote {path}")
    print(f"elapsed {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
