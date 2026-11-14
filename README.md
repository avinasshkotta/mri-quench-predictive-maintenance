# QuenchGuard — LSTM Predictive Maintenance for MRI Magnet Quench

**Case 1 of a predictive-maintenance study series: early warning of an MRI magnet quench
from multi-rate sensor time-series.**

A quench is the worst-day event for an MRI site: **$120,000** in repairs plus **5 days**
of scanner downtime (~$195k total). The helium pressure, cold head temperature,
gradient-coil vibration and RF power output all degrade before the failure — but on
*completely different time-scales*. This repo trains a PyTorch **LSTM** to fuse their
12-hour trends and warn **≥ 72 hours** ahead, while paging clinical staff **≤ 1 time per
magnet per month**.

| Deliverable | Path |
|---|---|
| Walkthrough notebook (16 markdown steps, executed) | [`notebooks/01_predictive_maintenance_lstm.ipynb`](notebooks/01_predictive_maintenance_lstm.ipynb) |
| **Interactive HTML report** (Plotly dashboard: hover, zoom, live threshold slider, sortable tables) | [`reports/quenchguard_report.html`](reports/quenchguard_report.html) |
| Static notebook export (HTML reference copy) | [`reports/notebook_export.html`](reports/notebook_export.html) |
| End-to-end pipeline | [`scripts/run_pipeline.py`](scripts/run_pipeline.py) |
| Machine-readable results | [`reports/metrics.json`](reports/metrics.json) |
| Case-study write-up (RCTC) | [`docs/case-study-rctc.md`](docs/case-study-rctc.md) |

---

## Results (held-out test period, seed 42)

| Model | PR-AUC | ROC-AUC | Event recall ≥72 h | Lead time (median) | False pages / magnet-month | Net saving |
|---|---|---|---|---|---|---|
| **LSTM (48×36)** | **0.998** | **0.999** | **100 % (4/4)** | **96.0 h** | **0.64** (budget 1) | **$452,000** |
| HistGradientBoosting (last window) | 0.997 | 1.000 | 100 % (4/4) | 93.6 h | 0.96 | $450,000 |
| LogisticRegression (last window) | 0.975 | 0.993 | 50 % (2/4) | 76.3 h | 0.00 | $228,000 |

* Thresholds are selected **on validation only**, under a tightened false-page budget
  (1 × 0.6 per magnet-month), then frozen for test.
* Cost model: reactive quench **$195,000** vs. planned repair **$81,000** vs. false
  alarm **$2,000** → **$114k saved per prevented quench**.

```
test quench events        : 4
detected with >= 72 h lead: 4/4 (100%)
lead times (h)            : 95.8, 96.0, 96.0, 96.0
false alerts              : 2 = 0.64/magnet-month (budget 1)
```

---

## Why a sequence model?

The failure signals are staggered — this is the whole point of the case:

| Sensor | Failure signal | Crosses it… before the quench |
|---|---|---|
| Helium pressure | drift below **2.0 MPa** | ≈ 79 h (14-day ramp) |
| Cold head temp | rise above **5.5 K** | ≈ **27 h** — far too late |
| Vibration | 100–200 Hz fault band, 120 Hz resonance | ≈ 44–84 h |
| RF power | dropout bursts | ≈ 24 h |

A static threshold on any single channel is either far too late or far too noisy. The
LSTM reads **48 consecutive 15-minute windows (12 h) × 36 engineered features** and fuses
the *trends*, which is what buys the three-day warning. The data also contains hard
negatives — helium refills, cryo-cooler hiccups, RF coil tuning, service visits, and
near-miss leaks that a top-up fixes — so single-sensor rules page constantly.

---

## Repository layout

```
mri-quench-predictive-maintenance/
├── notebooks/
│   └── 01_predictive_maintenance_lstm.ipynb   # executed walkthrough (16 steps)
├── scripts/
│   ├── run_pipeline.py                        # data → medallion → model → XAI → drift → fairness → reports
│   ├── build_medallion.py                     # Bronze → Silver → Gold + quality gates
│   ├── build_report.py                        # interactive HTML dashboard (Plotly)
│   ├── build_notebook.py                      # build + execute + export notebook HTML
│   ├── monitor.py                             # drift CLI (--check exits 1 on retraining trigger)
│   ├── audit_fairness.py                      # fleet fairness CLI (--check exits 1 on policy fail)
│   ├── validate_cmapss.py                     # NASA C-MAPSS external validation (--download)
│   ├── demo_app.py                            # Streamlit interactive demo (4 tabs)
│   ├── agent_demo.py                          # LangGraph agent demo (OpenAI, HITL ticket)
│   ├── eval_agent.py                          # offline agent eval (+ optional RAGAS)
│   └── serve.py                               # uvicorn entry point for the API
├── src/quenchguard/
│   ├── config.py          # every physics threshold, split, cost — one place
│   ├── data_gen.py        # synthetic multi-rate streams + labelled events (BRONZE)
│   ├── medallion.py       # Bronze → Silver → Gold builds + lineage manifests
│   ├── quality.py         # Great Expectations suites (silver quality gates)
│   ├── features.py        # 36 causal features (case-study table)  → GOLD
│   ├── labeling.py        # 96 h horizon, 7-day post-event embargo
│   ├── sequences.py       # time-ordered splits, scaling, 48-step windows
│   ├── baselines.py       # logreg + HistGBDT (must be beaten)
│   ├── lstm.py            # model, training loop, gradient attribution
│   ├── evaluate.py        # threshold policy, lead time, cost model, plots
│   ├── explain.py         # SHAP: TreeExplainer global/local explanations
│   ├── tracking.py        # MLflow experiment tracking + model registry
│   ├── serving.py         # FastAPI: /predict with SHAP top-factors
│   ├── monitoring.py      # Evidently PSI drift checks (retraining trigger)
│   ├── agent.py           # LangGraph maintenance agent (tools + RAG + HITL)
│   ├── llmops.py          # token/cost ledger + LangSmith config + eval rows
│   ├── fairness.py        # Fairlearn fleet audit (event recall + alert parity)
│   └── external_validation.py  # NASA C-MAPSS FD001 methodology validation
├── data/
│   ├── raw/               # BRONZE  : as-is simulator output (never mutated)
│   ├── silver/            # SILVER  : cleaned, deduplicated, range-checked tables
│   ├── gold/              # GOLD    : model-ready feature frame
│   ├── external/cmapss/   # NASA FD001 files (downloaded on demand, git-ignored)
│   └── lineage/           # per-layer manifests: sha256, rows, schema, config
├── models/                # LSTM + scaler + HistGBDT (serving artifacts, git-ignored)
├── tests/                 # 110 pytest tests (leakage, quality, XAI, tracking, API, drift, agent, llmops, fairness, external)
├── reports/               # metrics.json, sweeps, SHAP figures, interactive + notebook HTML
├── docs/
│   ├── case-study-rctc.md
│   ├── eu_ai_act.md       # AI Act obligation mapping + risk register
│   ├── external_validation.md  # C-MAPSS protocol, results, censoring caveats
│   └── model_card.md      # model card, attached to every MLflow run
├── Dockerfile             # serving image (FastAPI + CPU torch)
├── requirements.txt
└── pyproject.toml
```

---

## Quickstart

```bash
git clone <repo-url> mri-quench-predictive-maintenance
cd mri-quench-predictive-maintenance
pip install -r requirements.txt

python scripts/run_pipeline.py            # full run: data → medallion → model → XAI → reports
python scripts/build_report.py           # interactive HTML report (open in a browser)
python -m pytest                          # 110 tests

# external validation: NASA C-MAPSS FD001 (real turbofan data, downloaded on demand)
python scripts/validate_cmapss.py --download   # -> reports/external_validation.json

# interactive browser demo (4 tabs: fleet, drilldown, agent chat + HITL, governance)
streamlit run scripts/demo_app.py

# governance: fleet fairness audit (Fairlearn, grouped by magnet)
python scripts/audit_fairness.py          # -> reports/fairness.json (pass/warn/fail)
python scripts/audit_fairness.py --check  # CI mode: exit 1 on policy failure

# agent observability + evaluation (LLMOps)
python scripts/eval_agent.py              # offline structural eval -> reports/agent_eval.json
python scripts/eval_agent.py --ragas      # + RAGAS faithfulness (needs OPENAI_API_KEY)
# LangSmith tracing (optional): set LANGCHAIN_API_KEY before scripts/agent_demo.py

# maintenance agent (LangGraph tool loop + OpenAI; requires OPENAI_API_KEY)
# put OPENAI_API_KEY=sk-... in the repo's .env (copy .env.example; git-ignored)
$env:OPENAI_API_KEY = "sk-..."             # or export it in PowerShell
python scripts/agent_demo.py              # 3 scenarios: investigate, HITL ticket, guardrail
python scripts/agent_demo.py --interactive

# experiment tracking + model registry (SQLite store at ./mlflow.db)
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000   # browse runs

# drift monitoring (retraining trigger; --check exits 1 when it fires)
python scripts/monitor.py                  # train-vs-test PSI report + drift_report.html
python scripts/monitor.py --check          # CI mode: non-zero exit on retraining trigger

# serving: LSTM + SHAP explanations over HTTP
python scripts/serve.py                   # http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"magnet_id\": \"M3\", \"window\": [[...48 x 36 raw features...]]}"

# containerised serving (build after a pipeline run — models/ must exist)
docker build -t quenchguard .
docker run --rm -p 8000:8000 quenchguard

# data layer only (Bronze → Silver → Gold, with quality gates)
python scripts/build_medallion.py          # build all three layers
python scripts/build_medallion.py --check  # validate only, write nothing (CI job)

# notebook + static HTML export
python scripts/build_notebook.py          # build, execute, export reports/notebook_export.html
```

Useful flags:

```bash
python scripts/run_pipeline.py --regenerate      # rebuild synthetic data from scratch
python scripts/run_pipeline.py --epochs 5        # quick iteration
python scripts/run_pipeline.py --device cpu      # no GPU required
python scripts/build_notebook.py --build         # write the .ipynb without executing
```

> **Windows + NVIDIA note:** `TrainConfig.disable_cudnn = True` makes the CUDA LSTM use
> the native backend; some driver/torch combinations crash at process teardown when a
> cuDNN-trained LSTM is destroyed. Set it to `False` if your stack shuts down cleanly.

---

## Methodology (the parts reviewers challenge)

1. **Medallion data layer with quality gates** — bronze (`data/raw`) is the
   append-only simulator output; silver (`data/silver`) is cleaned, deduplicated
   and range-checked, and every silver table must pass its Great Expectations
   suite (nulls, physics ranges, key uniqueness, row-count bounds) before it is
   written; gold (`data/gold`) is the model-ready feature frame. Each layer
   writes a lineage manifest (sha256, schema, config snapshot) into
   `data/lineage/` — the audit trail required by EU AI Act Art. 10/11.
2. **Causal features only** — backward-looking rolling stats; the 7-day trend is fit on
   *complete calendar days* and joined with a 1-day lag; NaNs forward-filled within a
   magnet, never back-filled. Enforced by `tests/test_features.py`.
3. **Label = quench in the next 96 h**, with the acceptance test at ≥ 72 h lead — the
   24 h slack lets the model fire on the early trend instead of the terminal collapse.
   7 days after each event are embargoed (magnet offline for repair).
4. **Global time splits** (60/20/20 by absolute timestamp, never shuffled); sequences
   must be contiguous and entirely inside one split; scaler fit on train only.
5. **One operating point per model, chosen on validation** under the false-page budget
   (1 × 0.6 safety factor), then frozen. Test is scored exactly once.
6. **Business metrics, not textbook ones**: event recall at ≥ 72 h lead, false pages per
   magnet-month, lead-time distribution, and a dollar cost model.
7. **XAI where it pays** — SHAP TreeExplainer on the `hist_gbdt` baseline gives exact
   global (beeswarm) and local (waterfall + top-3 factors per alert) explanations; the
   LSTM keeps gradient attribution, because SHAP over 48×36 windows is orders of
   magnitude slower with no extra decision value. Both are written to
   `reports/figures/shap_*.png` and `metrics.json → shap`.
8. **Experiment tracking + registry** — every run logs params, all README metrics and
   artifacts to MLflow (SQLite store, `mlflow.db`), and the trained LSTM is registered
   as `quenchguard_lstm` with the `champion` alias. Tracking failures are warnings, never
   pipeline failures.
9. **Drift monitoring as the retraining trigger** — Evidently PSI (alert > 0.2) between
   the train-split reference and the current batch; a retraining recommendation fires
   when any SHAP-key feature drifts or > 25 % of features drift (`scripts/monitor.py
   --check` is CI-ready). The pipeline writes the verdict to `metrics.json → drift`,
   `reports/drift.json` and `reports/drift_report.html`.
10. **Agentic coordination with human-in-the-loop** — a LangGraph ReAct agent
    (`agent.py`) investigates alerts with *real* tools (champion-LSTM scoring, raw
    telemetry from the silver layer, TF-IDF RAG over the model card/case study/README)
    and drafts service tickets — but `create_ticket` sits behind a LangGraph
    `interrupt`: nothing is written until a human operator approves (EU AI Act
    Art. 14 oversight enforced at the orchestration level). Checkpointed per
    `thread_id`; tests run against a scripted LLM, so CI needs no API key.
11. **LLMOps: cost, tracing, evaluation** — a token ledger scrapes real usage
    from AI messages and prices it against a rate table into
    `reports/agent_costs.json`; LangSmith tracing is opt-in via
    `LANGCHAIN_API_KEY` (no hard dependency); an offline evaluation harness
    (`scripts/eval_agent.py`) runs four scenarios through the *real* graph and
    asserts the behavioural contract — tool order, tool payload contracts from
    the actual model/data, the HITL invariant, guardrail refusal — into
    `reports/agent_eval.json`, with optional RAGAS faithfulness scoring when an
    OpenAI key is available.
12. **Fleet fairness as a release gate** — Fairlearn `MetricFrame` grouped by
    `magnet_id`; safety equity requires **100 % event recall (≥ 72 h lead) on
    every event-bearing magnet**, measured with the pipeline's own
    `evaluate.match_events` so the audit can never disagree with the published
    metrics; alert-burden parity uses demographic-parity **difference** among
    alerting magnets (clean magnets that never page are not penalised) plus a
    false-positive-rate spread check. `scripts/audit_fairness.py --check` is
    CI-ready; results land in `metrics.json → fairness` and
    `reports/fairness.json`. The AI Act mapping and risk register live in
    [`docs/eu_ai_act.md`](docs/eu_ai_act.md).
13. **External validation on real data** — the MRI data is synthetic, so the
    *methodology* is validated on NASA C-MAPSS FD001 (100 run-to-failure
    turbofan engines + 100 official held-out engines): the same causal rolling
    features, engine-grouped 80/20 split, false-alert-budgeted threshold and
    event-level lead-time evaluation. HistGBDT reaches **PR-AUC 0.892, 84 %
    event recall, 75 % in-time recall (≥ 15 cycles), 0.18 false alerts/engine**
    on the official test set; the logistic baseline is lower on every metric.
    This is a framework validation, not a model-transfer claim — see
    [`docs/external_validation.md`](docs/external_validation.md) for the
    censoring caveats and honest denominators.

---

## Tests

```bash
python -m pytest
```

25 tests covering: window mapping and causal trend slopes, horizon labelling invariance
(truncating the future never changes a label), offline embargo, split-boundary safety,
gap rejection, alert-episode collapsing, horizon clamping of early alerts, false-alert
classification, the threshold policy's budget compliance, and cost arithmetic.
Plus 9 medallion tests: silver cleaning rules (dedupe, causal ffill, range filters),
Great Expectations gates passing on fleet data and failing on corrupt input, lineage
checksum verification, and gold-layer schema/no-NaN invariants.
Plus 9 Phase B tests: SHAP top-factor ordering/direction and the efficiency property
(base value + contributions = model output), figure/payload generation, MLflow metric
flattening, param/metric/tag logging to an isolated SQLite store, `champion` alias
registration with model round-trip loading, and the guarantee that a tracking outage
returns `None` instead of raising.
Plus 9 serving tests: health/model-info contracts, probability + alert + top-3 SHAP
factors in `/predict`, shape and NaN rejection (422), determinism across calls,
degraded mode when artifacts are missing (503), and `Artifacts.load` round-trip from
disk.
Plus 9 drift tests: PSI flags a shifted synthetic feature and stays quiet on stable
ones, retraining verdicts (key-feature drift, fleet-wide share, all-stable), bad-input
rejection, `metrics.json` key-feature extraction, report writing, and (when gold data
exists) the real train-vs-test check.
Plus 13 agent tests: TF-IDF RAG ranking, tool-loop round-trip, system-prompt gating,
scripted guardrail refusal, HITL interrupt → approval writes the ticket while
rejection blocks it, thread isolation (checkpointing), ticket markdown format, and
real-model tool calls (skip when artifacts are absent).
Plus 12 LLMOps tests: cost arithmetic + price table, unknown-model flagging, ledger
totals/persistence, usage scraping from AI messages, LangSmith env handling (incl.
no os.environ mutation), git-ignored `.env` loading that never overrides real env
vars, transcript flattening for RAGAS rows, and the full offline
eval harness (4/4 scenarios, incl. the ticket-without-approval filesystem invariant).
Plus 8 fairness tests: MetricFrame contract (selection / TPR / FPR / count per
magnet), hard failure when a magnet misses its events, clean magnets that never
page not penalising parity, alert-burden disparity detection, PR-AUC skipping
magnets without positives, report writing, and the real champion-LSTM audit
(event recall 100 % on all four event-bearing magnets, verdict `pass`).
Plus 15 external-validation tests: RUL reconstruction for truncated test engines,
causality (truncating the future cannot change a past feature), identity/age columns
excluded from the feature set, episode collapsing that never leaks across engines,
horizon clamping, censored-engine false-alert accounting, the 25/16 censoring
denominators, an end-to-end run on synthetic C-MAPSS-shaped files, split validation,
and — when the NASA files are present — the real FD001 shapes and headline result.
Plus 2 demo-app tests (Streamlit `AppTest`): the four-tab UI renders fleet metrics
and governance reports without exception, and a missing OpenAI key produces the
friendly warning instead of a crash.

---

## License

MIT — see [LICENSE](LICENSE).
