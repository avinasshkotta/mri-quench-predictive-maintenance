# QuenchGuard — project report

**MRI magnet quench early warning: predictive model, explainability, LLM agent,
governance and deployment.**

> Numbers below are taken from the committed artifacts in `reports/` and are
> reproducible with the commands in §12. This is a synthetic-data study: the
> methodology is externally validated, the MRI numbers are not clinical claims.

---

## 1. Executive summary

| Headline | Value (held-out test period) |
|---|---|
| Champion model | LSTM over 48 × 36 windows (12 h look-back) |
| PR-AUC | **0.998** (logreg 0.975, HistGBDT 0.997) |
| Event recall at ≥ 72 h lead | **100 %** (4/4 test quench events) |
| Median lead time | **96 h** (clamped at the 96 h label horizon) |
| False pages | **0.64 / magnet-month** (budget: 1.0) |
| Net saving | **$452,000** over the test period |
| Fairness verdict | **PASS** (event recall 100 % on all 4 event-bearing magnets) |
| External validation | NASA C-MAPSS FD001: PR-AUC **0.892**, event recall **84 %**, 0.18 false/engine |
| Agent | LangGraph tool loop + RAG + human approval gate, offline eval **4/4** |
| LLM evidence | RAGAS faithfulness **0.75**, answer relevancy **0.9001**; 2 calls, **$0.000204** |
| Tests / CI | **146 tests**, 7 CI jobs (Linux/Windows × py3.10/3.12, pipeline, Docker, frontend, notebook) |

The system answers three questions a hospital actually asks: *when will this
magnet quench?* (model), *why does it think so?* (SHAP), and *what should we do,
with a human in control?* (agent + approval gate).

---

## 2. Problem and business case

An MRI magnet quench is a sudden loss of superconductivity: the coil dumps its
stored energy, helium boils off, and the scanner is offline for days.

| Scenario | Cost |
|---|---|
| Reactive (quench happens) | $120,000 repair + 5 days downtime (~$15,000/day) = **$195,000** |
| Planned repair (hit) | 55 % of repair + one scheduled downtime day = **$81,000** |
| False alarm | **$2,000** per wasted service visit |

A correct early warning therefore saves ≈ **$114,000** minus the cost of false
pages. The operating constraint is not accuracy but **pager burden**: clinical
engineering can absorb at most **1 false page per magnet-month** before staff
start ignoring alerts.

---

## 3. Acceptance criteria

1. Detect every quench with **≥ 72 h** lead time on the held-out period.
2. Stay within **≤ 1 false page / magnet-month**.
3. Beat two classical baselines on the business metrics.
4. Explain each alert with its top contributing factors.
5. Audit what happened, who approved actions, and how the model was validated.

---

## 4. Architecture

```
sensors (1-min pressure/temp, 1 kHz vibration, 100 ms RF)
   │
   ▼
Bronze ──► Silver (Great Expectations) ──► Gold (36 causal features)
   │                                         │
   │                                         ▼
   │                          48-step window ─► LSTM ─► P(quench ≤ 96 h)
   │                                         │
   │                                         ▼
   │                          frozen threshold 0.90 ─► alert + SHAP top-3
   │                                         │
   │                    ┌────────────────────┴─────────────────────┐
   │                    ▼                                          ▼
   │            FastAPI /predict                      LangGraph agent (tools + RAG)
   │            + Art. 12 audit log                   └─► HITL interrupt ─► ticket + email
   │                    │                                          │
   └────────────────────┴──────────────► Next.js customer UI (SSE streaming)
```

Everything runs from **one Docker image**: static UI at `/`, streaming chat at
`/api/*`, model API at `/predict`.

---

## 5. Data

Real quench telemetry is proprietary and rare, so the four streams are
simulated with documented physics ramps (`config.py`), at their real sample
rates. Six magnets × 90 days produce:

- **777,600** 1-minute pressure readings (same for cold head),
- **51,834** gold windows of 36 features,
- **9** quench events across the timeline (3 train / 2 val / 4 test).

**Medallion layers + gates**

| Layer | Guarantee |
|---|---|
| Bronze (`data/raw`) | append-only simulator output, never mutated |
| Silver (`data/silver`) | deduplicated, range-checked, Great Expectations suite per table (nulls, physics ranges, key uniqueness, row-count bounds) |
| Gold (`data/gold`) | model-ready frame; schema and no-NaN invariants tested |

Each layer writes a **lineage manifest** (`data/lineage/*.json`) with sha256
per file, row counts, schema and the config snapshot — the Art. 10/11 audit
trail.

**Label** `y(t) = 1` iff a quench occurs on that magnet in `(t, t + 96 h]`.
Seven days after each event are **embargoed** (magnet offline for repair), so
the model never trains on post-mortem data. `ttf_hours` is stored for analysis
and is never a model input.

---

## 6. Features (36, all causal)

| Sensor | Examples |
|---|---|
| Helium pressure | rolling means (5/30 min), deltas, minutes below 2.0 MPa, 7-day trend slope |
| Cold head | deviation from 4.2 K, spike count, rolling std, minutes above 5.5 K |
| Vibration | RMS, 1 h rolling max/std, FFT peak, 100–200 Hz fault-band energy and ratio, 120 Hz line |
| RF power | rolling max, stability ratio, dropout count/rate |

No-leakage rules, all enforced by tests: rolling windows look backwards only;
the 7-day trend is fitted on complete calendar days with a one-day lag; 1-minute
data is bucketed as `(t − 15 min, t]`; NaNs are forward-filled within a magnet,
never back-filled; the scaler is fitted on training rows only.

**Splits** are global time cuts (60/20/20), never shuffled. Each sequence must
be contiguous and entirely inside one split.

---

## 7. Model and training

```
input (batch, 48, 36)
  └─ LSTM(hidden 64, layers 2, dropout 0.3) ─ last hidden
       └─ dropout ─ Linear(64, 1) ─ logit
```

- Loss: BCE-with-logits with a **capped** `pos_weight` (positive windows ≈ 4 %).
- AdamW + cosine schedule, gradient clipping 1.0, early stopping on **validation
  PR-AUC**.
- Baselines: class-weighted logistic regression and HistGBDT on the **last
  window only** — the LSTM additionally sees 12 h of history, so any win is
  attributable to temporal context.

---

## 8. Operating point policy

1. Sweep ~100 thresholds **on validation only**.
2. Convert false alerts to **false pages per magnet-month**.
3. Keep only points under `1.0 × 0.6` (safety factor for val→test shift).
4. Among those, maximise **event recall at ≥ 72 h lead**.
5. Freeze the threshold (0.90) and score the test split **once**.

`PREDICTION_THRESHOLD` can override the frozen value at runtime for operations,
without rebuilding the image.

---

## 9. Results

| Model | Threshold | PR-AUC | Window recall | Event recall ≥72 h | Median lead | False / magnet-month | Net saving |
|---|---|---|---|---|---|---|---|
| **LSTM (champion)** | 0.900 | **0.998** | 0.999 | **100 %** | **96 h** | **0.64** | **$452,000** |
| HistGBDT | 0.994 | 0.997 | 0.958 | 100 % | 93.6 h | 0.96 | $450,000 |
| Logistic regression | 0.999 | 0.975 | 0.658 | 50 % | 76.3 h | 0.00 | $228,000 |

Reading the table honestly:

- The HistGBDT is close on PR-AUC but **pages 50 % more often** (0.96 vs 0.64 /
  magnet-month) — exactly the constraint that matters operationally.
- Logistic regression never false-pages, but misses **half** the events (lead
  76 h when it does fire): cheap but unsafe.
- Lead times are clamped at the 96 h label horizon; the true sensor crossover
  for pressure is ≈ 79 h, which is why the warning cannot be much earlier.

---

## 10. Explainability

- **Global**: SHAP beeswarm over the test set (`reports/figures/shap_beeswarm.png`).
- **Local**: per-prediction top-3 factors with direction, returned by `/predict`
  and rendered in the UI.
- The explained model is a HistGBDT, so `TreeExplainer` gives exact
  game-theoretic values; the SHAP efficiency property (base value + contributions
  = model output) is asserted in `tests/test_explain.py`.
- LIME is not used (tree ensemble, not a black box); Grad-CAM does not apply
  (tabular time-series, no images). The rationale is documented in
  `docs/mlops_alignment.md`.

---

## 11. Monitoring, fairness and governance

**Drift** (`reports/drift.json`): Evidently PSI (threshold 0.2). The train→test
comparison flags **25 % of features** as drifted — the largest PSI values are
`p_roll30_mean`, `p_roll5_mean`, `p_mean`, `v_rms_rollmax_1h`, `v_rms` — and
sets `retrain_recommended: true`. `scripts/monitor.py --check` exits non-zero,
so CI turns red. This is a deliberate gate, not an automated retrain.

**Fairness** (`reports/fairness.json`): verdict **PASS**.

| Policy metric | Value | Limit |
|---|---|---|
| Event recall, per event-bearing magnet | 100 % (MRI-01/02/05/06) | 100 % |
| Selection-rate difference (alerting magnets) | 0.126 | ≤ 0.20 |
| Selection-rate ratio | 1.40 | ≤ 3.0 |
| FPR difference | 0.066 | ≤ 0.10 |

Fairlearn's raw `demographic_parity_difference` **0.439** and
`equalized_odds_difference` **1.0** are also reported. They look alarming, and
the reason is important: MRI-03 and MRI-04 have **no events**, so they correctly
never alert (selection rate 0, TPR 0). A naive parity metric punishes the model
for *not* paging healthy assets. That is why the deployment gate uses event-level
recall and alerting-only parity; DPD/EOD are kept for the audit trail and
discussed explicitly.

**EU AI Act mapping** (`docs/eu_ai_act.md`): Arts. 10 (data governance), 11
(model card), 12 (**per-prediction JSONL audit log**), 13 (SHAP explanations),
14 (**HITL gate**), 15 (drift + input validation). Art. 43 is documented as out
of scope for a capstone.

---

## 12. Agent layer (LLM, RAG, tools, human oversight)

The assistant is a LangGraph ReAct agent with four read-only tools
(`fleet_risk`, `score_magnet`, `sensor_summary`, `doc_search`) plus one gated
write tool (`create_ticket`).

- **Grounding**: answers come from tool results and TF-IDF retrieval over the
  model card, case study and README. RAGAS faithfulness 0.75, answer relevancy
  0.9001 on the small evaluation set.
- **Human gate**: `create_ticket` runs `interrupt()` before writing; the UI
  renders **Approve / Reject**; approval writes the ticket markdown and sends
  the email alert (SMTP, fail-safe).
- **Guardrail**: the system prompt limits scope to fleet maintenance and requires
  refusing clinical advice; a scripted refusal scenario is part of the eval.
- **Robustness**: aborted streams are repaired (dangling `tool_calls` are closed
  with cancellation messages); a pending approval returns `409` on a new message.
- **Evaluation**: 4/4 offline scenarios (`reports/agent_eval.json`), including
  the filesystem invariant that no ticket exists before approval.

**LLMOps**: token ledger prices dated snapshots; the production wrapper adds
retries with backoff, a 60 s timeout and a per-call token ceiling
(`LLM_MAX_TOKENS`, default 1024). Live demo cost: 2 calls, 1,154 tokens,
**$0.000204**.

**MCP**: the read-only tools are also exposed over the standard
Model Context Protocol (`python -m quenchguard.mcp_server`), so any MCP client
can discover them through one interface. `create_ticket` is deliberately absent
(least privilege).

---

## 13. Serving, UI and audit trail

| Surface | Purpose |
|---|---|
| `GET /` | customer web UI (static Next.js export) |
| `POST /api/chat` · `/api/chat/resume` | SSE agent stream: tokens, tool events, citations, interrupt |
| `GET /api/fleet`, `/api/magnet/{id}`, `/api/magnet/{id}/telemetry` | visual panels |
| `POST /predict` | probability + threshold + SHAP top-3 |
| `GET /health`, `/model-info` | readiness, model source/version |

Every `/predict` writes one JSONL record (Art. 12): UTC timestamp, magnet,
input sha256, probability, alert, threshold, SHAP top-3, model source/version,
latency. Model provenance is explicit (`MODEL_SOURCE=local|registry`,
`@champion` alias).

---

## 14. Testing and CI

**146 tests** (1 conditional skip) covering leakage, data quality, features,
sequences, labels, SHAP, MLflow/tracking, serving (env threshold, audit log,
registry fallback), drift, agent (tools, RAG, HITL both outcomes, thread
isolation), LLMOps, fairness (policy + DPD/EOD), MCP tool surface, email alerts,
web UI/SSE and external validation.

CI (`.github/workflows/ci.yml`): tests on **py3.10/3.12 × Ubuntu/Windows**,
end-to-end pipeline smoke test, Docker image build **including a check that `/`
serves the UI**, frontend static-export build, and notebook execution + HTML
export.

---

## 15. Deployment

One image: multi-stage Dockerfile (Node builds the UI, Python serves UI + API).

```bash
docker build -t quenchguard .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... quenchguard
```

- **Fly.io**: `fly.toml` included (1 GB shared machine, auto-stop, `/health`).
- **Google Cloud Run**: `--memory 1Gi --max-instances 1 --min-instances 0 --port 8000`.
  Single instance is required because the HITL approval state is in-process
  (`MemorySaver`); migrate to `langgraph-checkpoint-sqlite` for horizontal scaling.
- Configuration is environment-only: `OPENAI_API_KEY`, `PREDICTION_THRESHOLD`,
  `MODEL_SOURCE`, `PREDICTION_LOG_PATH`, `LLM_MAX_TOKENS`, `SMTP_*`,
  `TICKET_ALERT_TO`, `UI_ALLOW_ORIGINS`.

---

## 16. Limitations and risks (stated, not hidden)

1. **Synthetic MRI data** — plausible physics, not clinically validated. The
   methodology is externally validated on C-MAPSS, not the MRI model.
2. **Probabilities are uncalibrated**; 0.90 is a policy threshold, not a clinical
   probability. Isotonic calibration is a documented next step.
3. **Small event sample** — 4 test quenches; report the lead-time distribution,
   not a single mean.
4. **RAG is TF-IDF** — offline and deterministic; production would use embeddings
   + a vector store with reranking (and context-recall evaluation).
5. **Retraining is a gate**, not a loop; promotion remains manual.
6. **No production hallucination sampling**; RAGAS runs on demand.
7. **Single-instance HITL state** constrains autoscaling.

---

## 17. Roadmap

1. Walk-forward (rolling-origin) evaluation instead of one test cut.
2. Isotonic calibration on validation before the threshold sweep.
3. Vector-store RAG + context recall; scheduled index refresh.
4. Durable checkpointer + ticket store (SQLite/Postgres) for multi-instance.
5. Automated retraining with staged promotion (shadow → canary → prod).
6. Real telemetry adapter: same four parquet schemas, pipeline unchanged.

---

## 18. Reproduce

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py            # data → medallion → model → XAI → drift → fairness
python scripts/build_notebook.py          # model-training notebook (executed)
python scripts/build_project_notebook.py  # this project's end-to-end notebook
python scripts/serve.py                   # UI + API at http://127.0.0.1:8000
python -m pytest                          # 146 tests
```

Artifacts: `reports/metrics.json` (all headline numbers),
`reports/threshold_sweep.csv`, `reports/drift.json`, `reports/fairness.json`,
`reports/external_validation.json`, `reports/agent_eval.json`,
`reports/ragas_scores.json`, `reports/agent_costs.json`,
`reports/figures/*.png`, `reports/quenchguard_report.html`,
`reports/project_walkthrough.html`.

---

## 19. References

- NASA C-MAPSS turbofan degradation dataset (Saxena & Goebel) — external validation.
- SHAP (Lundberg & Lee, 2017) — additive feature attribution.
- Fairlearn — `MetricFrame`, demographic parity / equalized odds.
- Evidently AI — data drift, PSI.
- MLflow — tracking and model registry (`@champion` alias).
- LangGraph — stateful agent graphs, `interrupt()` human-in-the-loop.
- EU AI Act (Regulation 2024/1689) — Arts. 10–15, 43.
- Great Expectations — data quality gates.
- RAGAS — RAG evaluation (faithfulness, answer relevancy).
