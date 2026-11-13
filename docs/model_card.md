# Model Card — QuenchGuard LSTM Quench Detector

> Status: **complete** (v1.0). This card is versioned with the code and attached
> to every MLflow run. The EU AI Act obligation mapping lives in
> [`docs/eu_ai_act.md`](eu_ai_act.md).

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
* **Out of scope:** autonomous action. The LangGraph maintenance agent may
  investigate and *draft* tickets, but `create_ticket` sits behind a
  LangGraph `interrupt` — a human operator approves or rejects before anything
  is written (EU AI Act Art. 14 human oversight, enforced by orchestration).
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
* Serving-time responses include the top-3 contributing features
  (`/predict → explanation`).
* The maintenance agent (`scripts/agent_demo.py`) cites the same SHAP factors
  and this card's documentation (TF-IDF RAG) when it drafts tickets.

## Fairness

Audited with Fairlearn (`MetricFrame`) on the held-out test split, grouped by
`magnet_id` (the only grouping available in this study; the harness accepts any
column, so vendor / install year / site is a schema change, not a code change).
Reproduce with `python scripts/audit_fairness.py`; machine-readable results in
`reports/fairness.json`.

| Check | Policy | Result | Verdict |
|---|---|---|---|
| Event recall ≥ 72 h, per event-bearing magnet | 100 % (4/4 events) | MRI-01/02/05/06 = 100 % | pass |
| Selection-rate difference among alerting magnets | ≤ 0.20 | 0.126 | pass |
| Selection-rate ratio among alerting magnets | ≤ 3.0 | 1.40 | pass |
| False-positive-rate spread across the fleet | ≤ 0.10 | 0.066 | pass |
| Per-magnet PR-AUC spread (informational) | ≤ 0.10 | 0.0013 | pass |

**Overall: PASS.** Two magnets (MRI-03, MRI-04) have no test events and
correctly never alert; a max/min ratio against zero is undefined, so parity is
measured **among the four alerting magnets** rather than penalising clean units.
Per-magnet event counts are tiny (one event each), which is why the safety
policy requires *every* event to be detected instead of averaging recall.
Known limitation: with 4 test events this audit can detect gross
under-protection, not small per-group gaps — a production audit needs months of
per-site data.

## Human oversight

Oversight is enforced by orchestration, not by prompt instructions:

* The detector only **raises alerts**; it never dispatches technicians.
* The LangGraph maintenance agent (`src/quenchguard/agent.py`) investigates and
  **drafts** a service ticket, but `create_ticket` is wrapped in a LangGraph
  `interrupt()`. The graph pauses, a human operator approves or rejects, and
  only then is `reports/tickets/*.md` written. A rejection reason is fed back
  to the model.
* Tests assert both the interrupt and the filesystem invariant (no ticket file
  exists before approval), and the offline eval harness
  (`scripts/eval_agent.py`) re-checks it per release.
* Operators see the probability, the frozen threshold and the top-3 SHAP
  factors at the point of decision, so an alert is contestable.

## Monitoring and lifecycle

* **Drift:** Evidently PSI (alert > 0.2) between the training reference and the
  current batch (`scripts/monitor.py`); a retraining recommendation fires when
  any SHAP-key feature drifts or > 25 % of features drift. The current study
  legitimately flags late-period pressure shift (PSI 1.56 on
  `p_roll30_mean`) — see `reports/drift.json`.
* **Performance:** PR-AUC, event recall and false-page rate are recomputed on
  every pipeline run; the fairness audit is part of the pipeline
  (`metrics.json → fairness`).
* **Retraining:** monthly scheduled review **or** drift trigger. Candidates
  must beat the champion on validation PR-AUC, stay inside the false-page
  budget **and** pass the fairness audit before the `champion` alias moves.
* **Rollback:** the previous registry version remains addressable; alias
  promotion is reversible without redeploying code.

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
