# Model Card — QuenchGuard LSTM Quench Detector

> Status: **stub** — sections marked _(Phase G)_ are completed during the
> governance phase (full EU AI Act mapping, fleet fairness audit, HITL policy).
> This card is versioned with the code and attached to every MLflow run.

## Model details

| | |
|---|---|
| **Name** | `quenchguard_lstm` (MLflow registry, alias `champion`) |
| **Architecture** | 2-layer LSTM (hidden 64) + linear head, 12 h look-back (48 × 15-min steps × 36 features) |
| **Task** | Binary: will this magnet quench within the next 96 h? |
| **Framework** | PyTorch ≥ 2.2 |
| **Training code** | `scripts/run_pipeline.py`, `src/quenchguard/{features,labeling,sequences,lstm}.py` |
| **Threshold** | Selected on validation only under the false-page budget (≤ 0.6 × 1 false page/magnet-month), then frozen |

## Intended use

* **In scope:** early warning of MRI magnet quench for clinical engineering
  teams, ≥ 72 h before the event, to enable planned service instead of
  emergency repair.
* **Out of scope:** autonomous action. Alerts require human confirmation
  before any technician dispatch (EU AI Act Art. 14 human oversight).
* **Users:** MRI service coordinators, clinical engineering, vendor field
  engineers.

## Training data

* Synthetic multi-rate telemetry from `quenchguard.data_gen` — 6 magnets × 90
  days, 1-min pressure/cold-head streams, 15-min vibration/RF windows,
  9 labelled quench events plus hard negatives (refills, service visits,
  near-miss leaks).
* Time-ordered 60/20/20 split; scaler fit on train only; 7-day post-event
  embargo.
* Data lineage: `data/lineage/*.json` (sha256 per layer). Quality gates:
  `src/quenchguard/quality.py`.

## Metrics (held-out test, seed 42)

| Metric | Value |
|---|---|
| PR-AUC | 0.998 |
| Event recall at ≥ 72 h lead | 100 % (4/4) |
| Median lead time | 96.0 h |
| False pages / magnet-month | 0.64 (budget 1.0) |
| Net saving vs. reactive | $452,000 per study period |

_Benchmark baselines (same features): HistGBDT PR-AUC 0.997, LogReg 0.975._

## Explainability

* **Global + local SHAP** (TreeExplainer) on the `hist_gbdt` baseline —
  beeswarm + per-prediction waterfall (`reports/figures/shap_*.png`,
  `metrics.json → shap`).
* **Gradient attribution** for the LSTM (`importance` in `metrics.json`).
* Serving-time responses include the top-3 contributing features.

## Fairness _(Phase G)_

Fleet-fairness audit across magnet sites/vendors/ages (Fairlearn MetricFrame)
— to be filled.

## Limitations

* Trained on **synthetic** data; physics parameters are plausible, not
  clinically validated. Real-site fine-tuning required before production use.
* Study period is 90 days — seasonal duty-cycle shifts are not represented.
* Scores are only meaningful for the 36 documented features with the same
  15-min windowing (training-serving skew risk if windowing changes).

## Retraining policy

Triggered by data drift (Evidently PSI > 0.2 on key features) or by the
scheduled monthly review; candidate models must beat the champion on
validation PR-AUC and stay within the false-page budget before alias promotion.

## Contact

Capstone project — AI Career Compass. Repository: `mri-quench-predictive-maintenance`.
