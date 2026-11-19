# MLOps / LLMOps alignment — how QuenchGuard maps to the course material

This page is the evidence map for Week 5 (Modern AI Stack: LLM → RAG → Agent →
MCP) and Week 7 (MLOps, Deployment, XAI & Governance). Each row points at the
code or report a reviewer can open.

---

## 1. The four ops disciplines (Week 7 core concept)

| Discipline | What it adds | Where it lives in QuenchGuard |
|---|---|---|
| **DevOps** | CI/CD, containers, IaC | `.github/workflows/ci.yml` (4-OS test matrix, pipeline smoke test, Docker build, frontend build), `Dockerfile` |
| **MLOps** | data/model versioning, registry, drift, retraining | lineage manifests (`data/lineage/`), MLflow tracking + `@champion` registry alias (`tracking.py`), Evidently PSI drift + trigger (`monitoring.py`), fairness gate (`fairness.py`) |
| **AIOps** | AI to operate IT systems | *Not implemented as a product feature.* The LSTM anomaly pattern (sequence reconstruction/attention) is the same family described for log-anomaly AIOps; noted as out of scope, not claimed |
| **LLMOps** | prompt versioning, RAG, evals, token cost | prompts in Git (`agent.py:SYSTEM_PROMPT`), TF-IDF RAG (`doc_search`), offline eval + RAGAS (`eval_agent.py`, `reports/agent_eval.json`, `reports/ragas_scores.json`), token/cost ledger (`llmops.py`, `reports/agent_costs.json`), retries/timeout/token ceiling (`llmops.build_chat_model`) |

**Versus the naive reading:** DevOps would have shipped this model and called it
done. MLOps adds "is it still good?" (drift + fairness + registry). LLMOps adds
"is the agent still grounded, cheap and safe?" (RAGAS + ledger + guardrails +
HITL). Both are measured, not asserted.

---

## 2. LLMOps — the 7 production steps

| Step | Status | Evidence |
|---|---|---|
| 1. Version prompts in Git | ✅ | `SYSTEM_PROMPT` and tool descriptions are code-reviewed in `agent.py` |
| 2. RAG with chunking + retrieval | ✅ (TF-IDF, as taught) | `_chunk_markdown` + `_rag_index` + `doc_search` in `agent.py`; production upgrade path = embeddings + vector DB |
| 3. RAGAS evaluation set | ✅ | faithfulness + answer relevancy in `reports/ragas_scores.json`; context recall not measured (known gap) |
| 4. Guardrails before serving | ✅ | scope guardrail in the system prompt + tested refusal scenario; agent **tool** allow-list; no write tool reachable without HITL |
| 5. Token cost per query | ✅ | `TokenLedger` prices dated snapshots (`canonical_model`) → `reports/agent_costs.json` |
| 6. Hallucination monitoring | ⚠️ offline only | RAGAS faithfulness runs on demand, not sampled from production traffic |
| 7. Keep the index fresh | ⚠️ per-process | the TF-IDF index rebuilds at process start; no scheduled re-index job |

---

## 3. The 9 agent failure modes → implementation

| # | Failure mode | Mitigation here |
|---|---|---|
| 1 | Hallucinated answers | RAG grounding (`doc_search`) + "never invent sensor numbers" rule + RAGAS faithfulness |
| 2 | Wrong tool chosen | explicit tool descriptions + LangGraph conditional routing (`route`) |
| 3 | Outdated output | tools read live model/telemetry, not training-time memory |
| 4 | Prompt injection | system-prompt scope guardrail + tool allow-list; retrieved text is untrusted input (documented) |
| 5 | Conflicting agents | single orchestrator, one state graph, no multi-agent handoff |
| 6 | Forgetting context | LangGraph `MemorySaver` checkpointing keyed by `thread_id` |
| 7 | Repetitive loops | LangGraph default `recursion_limit` + ticket rejection tells the model not to retry without new evidence |
| 8 | Context overload | retrieval k=3 chunks, tool results truncated in the API (`tool_result` 1,500 chars) |
| 9 | Hard to debug | LangSmith tracing (opt-in), SSE tool trace in the UI, structured offline eval |

---

## 4. XAI tool choice (why SHAP only)

| Tool | Scope | Verdict for this system |
|---|---|---|
| **SHAP TreeExplainer** | global + local | ✅ used — champion comparison model is a HistGBDT; global beeswarm + local waterfall + per-prediction top-3 |
| LIME | local, model-agnostic | ✗ not needed: the explained model is a tree ensemble, not a black box; LIME's stochastic local approximation would add noise without regulatory benefit |
| Grad-CAM | CNN images | ✗ not applicable: no image input. MRI magnet data is tabular time-series |

The course's rule is "choose by model type"; documented absence is the
defensible answer, not an omission.

---

## 5. EU AI Act evidence (high-risk treatment, Arts. 8–15)

| Article | Implementation | Evidence |
|---|---|---|
| Art. 10 data governance | synthetic data card + medallion lineage + Fairlearn audit | `docs/model_card.md`, `data/lineage/`, `reports/fairness.json` |
| Art. 11 documentation | model card attached to every MLflow run | `docs/model_card.md`, `tracking.py` |
| **Art. 12 logging** | **per-prediction JSONL audit log** (timestamp, input sha256, probability, alert, threshold, SHAP top-3, model source/version); approved tickets also email the maintenance inbox (SMTP, fail-safe) | `serving.append_prediction_log`, `notify.notify_ticket`, `reports/prediction_log.jsonl` |
| Art. 13 transparency | SHAP top-3 in every `/predict` response; `/api/magnet/{id}` panel | `serving.py`, `webui.py` |
| Art. 14 human oversight | LangGraph `interrupt()` before `create_ticket`; UI Approve/Reject | `agent.py`, `webui.py` |
| Art. 15 robustness | Evidently PSI drift + retraining trigger; Pydantic validation; NaN/shape rejection | `monitoring.py`, `serving.py` |
| Art. 43 conformity | documented as out of scope for a capstone (see `docs/eu_ai_act.md` known gaps) | `docs/eu_ai_act.md` |

**Fairlearn measures.** `reports/fairness.json` reports both the *policy* metrics
and the course-contract metrics:

- `demographic_parity_difference` and `equalized_odds_difference` from
  `fairlearn.metrics`, computed over **all** magnets;
- the deployment policy (`selection_rate_difference` ≤ 0.20 among **alerting**
  magnets, FPR spread ≤ 0.10) because clean magnets correctly never alert.

The gap is the honest answer to the course's own interview question: raw DPD
penalises a model for *not* paging healthy assets (DPD 0.44, EOD 1.0 here are
driven by MRI-03/04 having no events), which is why safety equity is measured
at the **event** level (recall per event-bearing magnet) instead.

---

## 6. Serving, deployment and configuration

| Pattern | Implementation |
|---|---|
| FastAPI + Pydantic + SHAP at serving time | `serving.py` (`/predict`) |
| Registry-backed serving | `MODEL_SOURCE=registry` loads the `@champion` run's `serving/` artifacts; automatic fallback to `models/` |
| Threshold as environment variable | `PREDICTION_THRESHOLD` overrides the frozen artifact value |
| Audit logging togglable | `PREDICTION_LOG=off` or `PREDICTION_LOG_PATH=/data/audit.jsonl` |
| Container | multi-stage Dockerfile: Next.js static export + API in one image |
| MCP connection layer | `quenchguard.mcp_server` (stdio / SSE / streamable-http), read-only tools |

---

## 7. Known gaps (stated, not hidden)

1. **DVC** is not used; data versioning is served by sha256 lineage manifests plus
   the committed `data/demo/` bundle. Same guarantee (a commit identifies its
   data), different tool.
2. **Automated retraining** is a gate, not a loop: `monitor.py --check` fails CI
   and recommends retraining; promotion still requires a human.
3. **RAGAS context recall** is not measured; faithfulness + answer relevancy are.
4. **RAG is TF-IDF**, not embeddings + vector store — deliberate (offline,
   deterministic, zero cost). The upgrade path is documented.
5. **No staging environment**; dev and prod differ by configuration
   (`MODEL_SOURCE`, `PREDICTION_THRESHOLD`) and image, not by a full environment.
6. **No production hallucination sampling**; RAGAS runs on demand.
