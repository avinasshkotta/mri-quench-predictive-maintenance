# QuenchGuard — project guide (how to use it)

Everything you need to run, demo and extend the system, in one place.

---

## 1. What you get

| Piece | What it does | Where |
|---|---|---|
| **Customer web app** | chat with the maintenance agent, live fleet risk, SHAP + telemetry panels, Approve/Reject tickets | `frontend/` (served at `/`) |
| **Streaming API** | SSE agent stream: tokens, tool calls, citations, HITL interrupts | `/api/*` |
| **Model API** | probability + threshold + SHAP top-3 per window, Art. 12 audit log | `/predict` |
| **Agent CLI** | scripted end-to-end demo (investigate → ticket → guardrail) | `scripts/agent_demo.py` |
| **MCP server** | read-only tools over a standard protocol (stdio / SSE / HTTP) | `src/quenchguard/mcp_server.py` |
| **Notebooks** | 01 model training walkthrough · 02 whole-project walkthrough | `notebooks/` |
| **Reports** | interactive HTML dashboard · project walkthrough HTML | `reports/quenchguard_report.html`, `reports/project_walkthrough.html` |

The committed `models/` triple (~0.6 MB) and `data/demo/` bundle (~0.6 MB) mean
the app runs **without** re-running the pipeline or regenerating data.

---

## 2. Prerequisites

- Python **3.10–3.13**
- Node.js **20+** (only for frontend development; production uses the built export)
- An OpenAI API key (chat only — the model API, panels and tests work without it)
- Docker (optional, for the single-container deployment)

---

## 3. Quick start

```bash
git clone https://github.com/avinasshkotta/mri-quench-predictive-maintenance
cd mri-quench-predictive-maintenance
pip install -r requirements.txt

# optional: full rebuild of data → model → reports (GPU not required)
python scripts/run_pipeline.py --device cpu

# run the product locally
python scripts/serve.py            # http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000** — the UI is served from `frontend/out` (a
pre-built export; rebuild it with `npm run build` in `frontend/`).

For chat, put your key in the repo `.env`:

```bash
cp .env.example .env      # then set OPENAI_API_KEY=sk-...
```

Tests: `python -m pytest` → 146 passed, 1 conditional skip.

---

## 4. Using the customer web app

### Fleet risk
Cards for every magnet with the latest champion-LSTM probability and an ALERT
badge if it crosses the frozen threshold (0.90). The highest-risk magnet is
selected automatically.

### Magnet drilldown
Click a card (or let the agent select one): probability vs baseline vs
threshold, **SHAP top-3** with direction, and 72 h of pressure / cold-head /
vibration sparklines from the silver layer.

### Agent chat
Type a question such as:

- *Which magnet is at highest risk right now?*
- *Explain the risk factors for MRI-05.*
- *What does the model card say about limitations?*
- *Draft and submit a low-priority service ticket for MRI-06 with a summary and recommended action.*

While it works you see: streaming tokens, each tool call (`fleet_risk`,
`score_magnet`, `sensor_summary`, `doc_search`) with its result, and **citation
chips** from the docs. Hitting **Stop** aborts; the next message repairs the
thread automatically.

### Human approval (the important part)
When the agent drafts a ticket the chat **pauses**: a card shows the magnet,
priority, summary and recommended action, with **✅ Approve** and **🚫 Reject**
(optional reason). Nothing is written until you click. On approval:

1. `reports/tickets/<timestamp>_<magnet>.md` is created,
2. an **email alert** is sent if SMTP is configured,
3. the agent summarises what happened in the conversation.

If you try to send another message while an approval is pending the API answers
`409` — approve or reject first.

### Ticket email alerts
Set these in `.env` (port **587**, STARTTLS):

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASSWORD=app-password
SMTP_FROM=you@gmail.com
TICKET_ALERT_TO=maintenance@example.com,biomed@example.com
```

No configuration → the notifier is a no-op (tickets still work). Delivery
failures are warnings; a mail outage never loses a ticket.

---

## 5. Command-line walkthroughs

```bash
# agent: 3 scenarios (investigate, HITL ticket, guardrail refusal)
python scripts/agent_demo.py
python scripts/agent_demo.py --interactive

# MCP server (read-only tools) + client round-trip
$env:PYTHONPATH = "src"              # bash: export PYTHONPATH=src
python -m quenchguard.mcp_server     # stdio
python -m quenchguard.mcp_server --transport streamable-http --port 8765
python scripts/mcp_client_demo.py --tool fleet_risk

# notebooks (executed copies are committed)
python scripts/build_notebook.py            # 01: trains the LSTM
python scripts/build_project_notebook.py    # 02: whole project, no training

# reports
python scripts/build_report.py              # interactive HTML dashboard
python scripts/monitor.py                   # drift report + retraining verdict
python scripts/monitor.py --check           # CI mode (exit 1 when retrain fires)
python scripts/audit_fairness.py            # fairness audit (pass/warn/fail)
python scripts/validate_cmapss.py --download # NASA C-MAPSS external validation
```

---

## 6. API reference

Base URL: `http://127.0.0.1:8000` (also `/docs` for the OpenAPI UI).

| Method & path | Purpose |
|---|---|
| `GET /api/status` | LLM key, artifacts, data source, email configured |
| `POST /api/chat` | SSE stream for one turn `{message, thread_id}` |
| `POST /api/chat/resume` | resume the HITL gate `{thread_id, approved, reason?}` |
| `GET /api/fleet` | fleet probabilities (cached ~2 min) |
| `GET /api/magnet/{id}` | score + SHAP top-3 |
| `GET /api/magnet/{id}/telemetry?hours=72` | chart series |
| `POST /predict` | score a raw 48 × 36 window |
| `GET /health`, `/model-info` | readiness and model provenance |

**SSE events** from `/api/chat`: `meta`, `token`, `tool_start`, `tool_result`,
`citation`, `interrupt`, `halted`, `done`, `error`.

```bash
# streaming chat (curl prints the frames as they arrive)
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"which magnet is at highest risk?","thread_id":"demo-1"}'

# approve the pending ticket
curl -N -X POST http://127.0.0.1:8000/api/chat/resume \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo-1","approved":true}'

# score one window (48 × 36 zeros is fine for a smoke test)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"magnet_id\":\"MRI-05\",\"window\":[[0,0,...]]}"
```

---

## 7. Configuration

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | – | chat/agent (503 without it) |
| `PREDICTION_THRESHOLD` | artifact value (0.90) | override the alert threshold, no rebuild |
| `MODEL_SOURCE` | `local` | `registry` loads the MLflow `@champion` triple, falls back to `models/` |
| `PREDICTION_LOG_PATH` | `reports/prediction_log.jsonl` | Art. 12 audit log location |
| `PREDICTION_LOG` | on | `off` disables prediction logging |
| `LLM_MAX_TOKENS` | `1024` | per-call completion ceiling |
| `UI_ALLOW_ORIGINS` | `localhost:3000,127.0.0.1:3000` | CORS for a separately hosted UI |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_USE_TLS`, `TICKET_ALERT_TO` | – | ticket email alerts |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | registry location for `MODEL_SOURCE=registry` |

---

## 8. Frontend development

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000, talks to the API on :8000
npm run build      # static export -> frontend/out (served by FastAPI)
```

The UI resolves the API automatically: same origin in production, `:8000` on
localhost. For split hosting (e.g. Vercel + a remote API) build with
`NEXT_PUBLIC_API_BASE=https://your-api.example.com` and set `UI_ALLOW_ORIGINS`
on the API.

---

## 9. Docker / deployment

```bash
docker build -t quenchguard .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... quenchguard
```

- **Fly.io**: `fly.toml` is included; set secrets and `fly deploy`.
- **Cloud Run**: `--memory 1Gi --max-instances 1 --min-instances 0 --port 8000`
  (single instance keeps the in-memory HITL state correct).
- **Email**: add the `SMTP_*` / `TICKET_ALERT_TO` secrets; port 587 is allowed on
  both platforms.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Chat returns **503** `OPENAI_API_KEY is not configured` | add the key to `.env` (or the container env) and restart |
| Fleet/drilldown says artifacts missing | `models/` is committed; otherwise run `python scripts/run_pipeline.py --device cpu` |
| `409 a ticket approval is pending` | approve or reject the open ticket first |
| Next message after **Stop** fails with a tool_calls error | fixed by thread repair; update to the latest commit |
| `Address already in use` | another process on 8000 → `python scripts/serve.py --port 8001` |
| UI loads but panels are empty | backend not running, or `frontend/out` missing (`cd frontend && npm run build`) |
| CORS error in dev | add your dev origin to `UI_ALLOW_ORIGINS` |
| No email arrives | `SMTP_HOST`/`TICKET_ALERT_TO` unset, port 25 blocked (use 587), or provider rejected the login — check the warning in the server log |
| First request after idle is slow | serverless cold start (torch import); warm it or keep one instance running |
| Tests fail on Windows PowerShell exit code | run `python -m pytest -q` directly; a pipeline through `Select-Object` can mask the exit code |

---

## 11. Where things live

```
src/quenchguard/        pipeline, model, XAI, drift, fairness, agent, API, MCP, notifier
scripts/                run_pipeline, serve, demos, evals, notebook builders, validators
frontend/               Next.js customer UI (static export)
notebooks/              01 model training · 02 end-to-end walkthrough
docs/                   model card, EU AI Act, external validation, MLOps alignment,
                        project report, this guide, interview Q&A
reports/                metrics, sweeps, figures, fairness, drift, agent eval, HTML dashboards
data/lineage/           sha256 manifests per layer
```

---

## 12. FAQ

**Do I need a GPU?** No. Training runs on CPU (`--device cpu`); inference is
milliseconds per window.

**Can I use real data?** Yes — match the four parquet schemas in `data/silver/`
and re-run the pipeline; no code changes.

**How do I change the alert threshold?** `PREDICTION_THRESHOLD=0.85` for a
runtime override; to re-tune properly, change the false-page budget in
`config.py` and re-run the pipeline (validation only).

**Why didn't the agent create my ticket?** It only writes after human approval,
and only when it decides a ticket is warranted. Ask explicitly and approve the
card.

**How do I see the audit trail?** `reports/prediction_log.jsonl` (JSONL, one
record per prediction) and `reports/tickets/*.md`.

**Where are the tests for X?** `tests/` — 146 tests grouped by module; the README
lists what each group covers.
