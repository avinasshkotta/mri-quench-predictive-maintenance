# EU AI Act obligation mapping — QuenchGuard

> Regulation (EU) 2024/1689 (AI Act). This document maps the obligations that
> would apply to QuenchGuard to concrete artifacts in this repository, and
> states honestly where the system currently sits. It is a **design-target
> mapping**, not a legal opinion.

## 1. Classification (the first governance decision)

| Question | Answer | Consequence |
|---|---|---|
| Is it a medical device under MDR 2017/745? | **No.** It forecasts equipment failure; it does not diagnose, treat or monitor a patient. | Not Annex I high-risk via the medical-device route. |
| Is it an Annex III system? | **No.** Annex III point 2 covers safety components of critical digital infrastructure, road traffic, and water/gas/heating/electricity supply. Hospital facilities maintenance is not listed. | Not automatically high-risk. |
| Does it interact with people? | Yes — the maintenance agent is a chatbot/assistant used by staff. | **Art. 50 transparency** applies (disclose AI interaction). |
| Does it process personal data? | No patient data; operator identity appears only in ticket approval. | GDPR is out of scope here, but ticket logs must stay minimal. |

**Determination:** QuenchGuard is a **limited-risk** system. We nevertheless
implement the high-risk control set (Arts. 9–15, 26) **voluntarily**, because
its alerts drive safety-relevant maintenance decisions on clinical equipment.
That is the defensible position: classify honestly, then over-control.

## 2. Obligation map

Legend: ✅ implemented · 🟡 partial / documented gap · ⬜ not applicable.

### Provider-side (Arts. 8–15) — treated as if high-risk

| Art. | Obligation | How QuenchGuard satisfies it | Evidence |
|---|---|---|---|
| 9 | Risk management system | Risk register below; quantified failure modes; safety-tightened threshold budget (0.6 × 1 false page/magnet-month); cost model of misses vs false pages | `docs/eu_ai_act.md` §3, `src/quenchguard/evaluate.py` |
| 10 | Data governance | Medallion layers (bronze immutable → silver validated → gold model-ready); Great Expectations gates; sha256 lineage per layer; time-ordered split with 7-day post-event embargo; leakage tests | `src/quenchguard/{medallion,quality}.py`, `data/lineage/*.json`, `tests/test_leakage.py` |
| 11 | Technical documentation | This repo: model card, case study, architecture, reproducible pipeline, pinned requirements | `docs/model_card.md`, `README.md`, `requirements.txt` |
| 12 | Record-keeping / logging | **Per-prediction JSONL audit log** (UTC timestamp, magnet, input sha256, probability, alert, threshold, SHAP top-3, model source/version) via `append_prediction_log`; MLflow run per pipeline execution; per-layer lineage; agent checkpointing per `thread_id`; ticket files record approver role and UTC timestamp | `src/quenchguard/serving.py`, `reports/prediction_log.jsonl`, `src/quenchguard/tracking.py`, `src/quenchguard/agent.py`, `reports/tickets/*.md` |
| 13 | Transparency & instructions for use | Model card states intended use, out-of-scope use, metrics, limitations; `/predict` returns probability, threshold and top-3 SHAP factors; agent replies are grounded in tool results | `docs/model_card.md`, `src/quenchguard/serving.py` |
| 14 | Human oversight | `create_ticket` is behind a LangGraph `interrupt()` — no write before human approval; rejection reasons are fed back; detector only alerts, never acts | `src/quenchguard/agent.py`, `tests/test_agent.py::test_ticket_interrupt_blocks_until_approved` |
| 15 | Accuracy, robustness, cybersecurity | Published PR-AUC 0.998 / 100 % event recall @72 h / 0.64 false pages per magnet-month, reproducible on GPU; NaN/shape rejection at the API; PSI drift monitoring with retraining trigger; dependency pinning; secrets via env only | `README.md`, `reports/metrics.json`, `src/quenchguard/monitoring.py`, `tests/test_api.py` |
| 17 | Quality management system | CI: test suite, Docker build + live smoke test, drift check CLI, offline agent eval gate | `.github/workflows/ci.yml` |

### Deployer-side (Art. 26) — hospital / service organisation

| Art. 26(1) | Use per instructions | Deployer must keep the model card and honour the documented scope. |
|---|---|---|
| 26(2) | Assign human oversight | Named clinical-engineering operator owns the approval queue. |
| 26(3) | Ensure input data relevance | Sensor feed must match the 36 documented features and 15-min windowing; the drift check verifies this continuously. |
| 26(4) | Monitor operation | `scripts/monitor.py --check` in the site's scheduled job; alerts on PSI breach. |
| 26(5) | Log retention | MLflow + ticket logs retained for the site's audit period (≥ 6 months recommended). |
| 26(6) | Inform workers | Staff using the agent are told it is an AI assistant (Art. 50). |

### Transparency (Art. 50)

| Requirement | Implementation |
|---|---|
| Disclose AI interaction | `scripts/agent_demo.py` prints the model name and that a human gate is armed; the system prompt requires the agent to identify itself as the maintenance coordinator agent. |
| Mark synthetic content | Agent output is a draft ticket, always labelled "approved by: human operator (HITL gate)" — never presented as an autonomous decision. |

### Post-market (Arts. 72–73)

| Art. | Obligation | Implementation |
|---|---|---|
| 72 | Post-market monitoring | Drift + performance + fairness are recomputed every pipeline run and written to `metrics.json`; monthly review is the documented cadence. |
| 73 | Serious-incident reporting | Missed quench with patient impact is a reportable incident; `reports/metrics.json → results.events` provides the evidence trail (event, lead time, detected flag). |
| — | Fundamental-rights impact assessment (Art. 27) | ⬜ Not required at limited risk. If a deployer classifies the system high-risk, this must be added before go-live. |

## 3. Risk register (Art. 9)

| ID | Risk | Likelihood | Impact | Control | Residual |
|---|---|---|---|---|---|
| R1 | Missed quench (false negative) | Low | High (equipment damage, service disruption) | Threshold tuned for event recall, not accuracy; 100 % event recall @72 h on test; human review of every alert | Accepted, monitored |
| R2 | Alert fatigue from false pages | Medium | Medium | False-page budget 1.0/magnet-month, safety factor 0.6; measured 0.64 | Low |
| R3 | Distribution shift (sensor recalibration, season) | Medium | High | PSI monitoring, retraining trigger, monthly review | Low |
| R4 | Uneven protection across magnets | Low | High | Fairlearn audit; event-level recall must be 100 % per event-bearing magnet | Low (4-event sample limit) |
| R5 | Agent takes an unsafe action | Low | High | No write tools outside the HITL gate; scope guardrail; tool-grounded answers only | Low |
| R6 | Hallucinated sensor values | Medium | Medium | Agent may only cite tool results; eval harness checks payload contracts | Low |
| R7 | Training-serving skew (windowing change) | Medium | High | Feature order frozen in `FEATURE_COLUMNS`; API validates shape; drift check | Low |
| R8 | Prompt injection via ticket text | Low | Medium | Agent has no free-text write path; ticket fields are validated (priority enum) | Low |

## 4. Known gaps (stated, not hidden)

1. **Synthetic training data.** Physics is plausible but not clinically
   validated; real-site fine-tuning and a prospective silent trial are required
   before production.
2. **Four test events.** The fairness audit can catch gross under-protection,
   not small per-group gaps. Production needs months of per-site data.
3. **No formal QMS certification** (ISO 13485 / ISO 42001). Art. 17 is mapped
   to CI controls, not to a certified management system.
4. **Human-factors testing** of the approval UI has not been performed.
5. **No post-market incident workflow tooling** — the evidence trail exists,
   the organisational process is the deployer's responsibility.

## 5. Change control

Any change to the model, features, threshold, fairness policy or agent tools
requires: (a) full test suite green, (b) pipeline metrics reproduced,
(c) fairness audit still `pass`, (d) this document and the model card updated
in the same commit. The `champion` alias only moves after all four.
