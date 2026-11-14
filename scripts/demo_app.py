#!/usr/bin/env python
"""QuenchGuard interactive demo (Streamlit).

    python scripts/run_pipeline.py     # once: train artifacts + reports
    streamlit run scripts/demo_app.py  # then open the printed URL

Four tabs:

1. **Fleet**        — live champion-LSTM risk for every magnet (real tools).
2. **Drilldown**    — one magnet: probability, SHAP top-3, raw telemetry charts.
3. **Agent chat**   — the full LangGraph agent: tools, RAG, guardrail and the
                      human-in-the-loop ticket gate (approve/reject buttons).
4. **Governance**   — fairness verdict, drift status, external validation,
                      agent eval and RAGAS scores straight from reports/.

Secrets: repo ``.env`` (git-ignored) or Streamlit Cloud ``st.secrets``.
Artifacts are git-ignored, so the fleet/drilldown/chat tabs need a local
``python scripts/run_pipeline.py``; the governance tab degrades gracefully.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
import streamlit as st

from quenchguard.llmops import load_env

load_env()                                   # repo .env, real env vars win
try:                                         # Streamlit Cloud secrets
    for key in ("OPENAI_API_KEY", "LANGCHAIN_API_KEY"):
        if key in st.secrets:
            os.environ.setdefault(key, str(st.secrets[key]))
except Exception:
    pass

REPORTS = REPO_ROOT / "reports"

# --------------------------------------------------------------------------------------
# cached loaders (the same production code paths the API and agent use)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading champion LSTM + scaler + HistGBDT…")
def _artifacts():
    from quenchguard.serving import Artifacts

    return Artifacts.load()


@st.cache_data(show_spinner="Scoring the fleet…")
def _fleet() -> dict[str, Any]:
    from quenchguard.agent import fleet_risk

    return json.loads(fleet_risk())


@st.cache_data(show_spinner="Scoring magnet…")
def _score(magnet: str) -> dict[str, Any]:
    from quenchguard.agent import score_magnet

    return json.loads(score_magnet(magnet))


@st.cache_data(show_spinner="Reading silver telemetry…")
def _sensor_series(magnet: str, hours: int = 72) -> pd.DataFrame:
    from quenchguard.agent import data_file

    specs = (("pressure.parquet", "timestamp", "pressure_mpa", "helium_pressure_mpa"),
             ("cold_head.parquet", "timestamp", "cold_head_k", "cold_head_k"),
             ("vibration.parquet", "window_end", "rms_g", "vibration_rms_g"))
    out = pd.DataFrame()
    for fname, ts, val, label in specs:
        path = data_file(fname)
        if path is None:
            continue
        df = pd.read_parquet(path)
        df = df[df.magnet_id == magnet].sort_values(ts)
        cut = df[ts].max() - pd.Timedelta(hours=hours)
        out[label] = df[df[ts] >= cut].set_index(ts)[val]
    return out.sort_index()


def _report(name: str) -> dict[str, Any] | None:
    path = REPORTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _artifacts_available() -> bool:
    return (REPO_ROOT / "models" / "lstm_detector.pt").exists()


# --------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="QuenchGuard — MRI quench early warning",
                   page_icon="🧲", layout="wide")
st.title("🧲 QuenchGuard — predictive maintenance for MRI magnet quench")
st.caption("Early warning ≥ 72 h before a quench · champion LSTM + Fairlearn gate + "
           "LangGraph agent with a human-in-the-loop ticket gate")

tab_fleet, tab_drill, tab_agent, tab_gov = st.tabs(
    ["🚨 Fleet risk", "🔬 Magnet drilldown", "🤖 Maintenance agent", "🏛 Governance"])


# ---- tab 1: fleet --------------------------------------------------------------------
with tab_fleet:
    if not _artifacts_available():
        st.warning("Trained artifacts missing — run `python scripts/run_pipeline.py` "
                   "first (also produces the governance reports).")
    else:
        fleet = _fleet()
        rows = fleet.get("fleet", [])
        st.subheader("Latest-window quench risk per magnet")
        cols = st.columns(max(1, len(rows)))
        for col, row in zip(cols, rows):
            col.metric(
                row["magnet_id"],
                f"{row['quench_probability']:.4f}",
                "⚠️ ALERT" if row["alert"] else "nominal",
                delta_color="inverse" if row["alert"] else "off")
        if rows:
            top = rows[0]
            st.info(f"Highest risk: **{top['magnet_id']}** at "
                    f"p={top['quench_probability']:.4f} — alert threshold 0.90 "
                    f"(frozen; chosen on validation under the false-page budget).")
            st.bar_chart({r["magnet_id"]: r["quench_probability"] for r in rows})


# ---- tab 2: drilldown ------------------------------------------------------------------
with tab_drill:
    if not _artifacts_available():
        st.warning("Trained artifacts missing — run `python scripts/run_pipeline.py` first.")
    else:
        magnets = [r["magnet_id"] for r in _fleet().get("fleet", [])] or ["MRI-01"]
        magnet = st.selectbox("Magnet", magnets)
        score = _score(magnet)
        left, right = st.columns([1, 2])
        with left:
            st.metric("quench probability (next 96 h)",
                      f"{score['quench_probability']:.4f}",
                      f"baseline {score['baseline_probability']:.4f}")
            st.metric("threshold", f"{score['threshold']:.2f}",
                      "ALERT" if score["alert"] else "below threshold")
        with right:
            st.markdown("**Top-3 SHAP factors** (hist_gbdt cross-check)")
            factors = score.get("explanation", [])
            if factors:
                st.table(pd.DataFrame([
                    {"feature": f["feature"],
                     "SHAP": round(f["shap_value"], 4),
                     "direction": f.get("direction", "")}
                    for f in factors]))
        st.markdown(f"**Telemetry — last 72 h** ({magnet})")
        series = _sensor_series(magnet)
        if not series.empty:
            st.line_chart(series)
        else:
            st.caption("no telemetry found — run the pipeline or "
                       "`scripts/build_demo_bundle.py`")


# ---- tab 3: agent ---------------------------------------------------------------------
def _stream_investigation(graph, thread_id: str, user_text: str) -> dict[str, Any] | None:
    """Stream one user turn; returns interrupt payload when the HITL gate fires."""
    status = st.status("Agent investigating…", expanded=True)
    interrupted = None
    for update in graph.stream({"messages": [("user", user_text)]},
                               {"configurable": {"thread_id": thread_id}},
                               stream_mode="updates"):
        for node, payload in update.items():
            if node == "__interrupt__":
                p = payload[0].value if isinstance(payload, list) else payload.value
                interrupted = p
                continue
            for msg in (payload.get("messages") or []):
                kind = getattr(msg, "type", "?")
                if kind == "ai":
                    for call in (getattr(msg, "tool_calls", None) or []):
                        args = ", ".join(f"{k}={str(v)[:50]}"
                                         for k, v in call["args"].items())
                        status.write(f"🔧 tool → `{call['name']}({args})`")
                    if msg.content and not getattr(msg, "tool_calls", None):
                        status.write(f"🤖 {str(msg.content)[:300]}")
    status.update(label="Investigation complete", state="complete", expanded=False)
    return interrupted


with tab_agent:
    st.markdown("The agent may **score the model, read raw telemetry and search the "
                "docs** — but `create_ticket` sits behind a LangGraph `interrupt`: "
                "nothing is written until a human approves. (EU AI Act Art. 14, "
                "enforced at the orchestration level.)")
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY not set — put it in the repo `.env` "
                   "(copy `.env.example`) or in Streamlit Cloud secrets.")
    else:
        if "agent_messages" not in st.session_state:
            st.session_state.agent_messages = []
        if "agent_graph" not in st.session_state:
            from langchain_openai import ChatOpenAI

            from quenchguard.agent import build_graph

            st.session_state.agent_graph = build_graph(
                ChatOpenAI(model="gpt-4o-mini", temperature=0))
        graph = st.session_state.agent_graph
        thread = "streamlit-demo"

        for role, content in st.session_state.agent_messages:
            with st.chat_message(role):
                st.markdown(content)

        pending = st.session_state.get("pending_ticket")
        if pending:
            st.error("⏸ **HUMAN GATE — the agent drafted a ticket and stopped.**")
            st.write(f"magnet: **{pending.get('magnet_id')}** · priority: "
                     f"**{pending.get('priority')}**")
            st.markdown(f"> {pending.get('summary')}")
            st.markdown(f"Action: {pending.get('recommended_action')}")
            reason = st.text_input("Rejection reason (used only if you reject):",
                                   "operator rejected in demo UI")
            c1, c2 = st.columns(2)
            if c1.button("✅ Approve and write ticket"):
                from langgraph.types import Command

                final = graph.invoke(Command(resume={"approved": True}),
                                     {"configurable": {"thread_id": thread}})
                text = str(final["messages"][-1].content)
                st.session_state.agent_messages.append(("assistant", text))
                del st.session_state["pending_ticket"]
                st.rerun()
            if c2.button("🚫 Reject"):
                from langgraph.types import Command

                final = graph.invoke(
                    Command(resume={"approved": False, "reason": reason}),
                    {"configurable": {"thread_id": thread}})
                text = str(final["messages"][-1].content)
                st.session_state.agent_messages.append(("assistant", text))
                del st.session_state["pending_ticket"]
                st.rerun()

        user = st.chat_input("Ask the maintenance agent… (e.g. *which magnet is at "
                             "highest risk right now?*)")
        if user:
            with st.chat_message("user"):
                st.markdown(user)
            if not _artifacts_available():
                st.warning("Agent tools need trained artifacts — run the pipeline first.")
            else:
                interrupted = _stream_investigation(graph, thread, user)
                if interrupted:
                    st.session_state.pending_ticket = interrupted.get("args", {})
                    st.rerun()
                else:
                    result = graph.get_state({"configurable": {"thread_id": thread}})
                    msgs = (result.values.get("messages") if result.values else []) or []
                    text = str(msgs[-1].content) if msgs else "(no reply)"
                    st.session_state.agent_messages.append(("assistant", text))
                    with st.chat_message("assistant"):
                        st.markdown(text)


# ---- tab 4: governance -----------------------------------------------------------------
with tab_gov:
    st.subheader("Release gates (from reports/)")
    cols = st.columns(4)
    metrics = _report("metrics.json")
    fairness = _report("fairness.json")
    drift = _report("drift.json")
    ext = _report("external_validation.json")
    agent_eval = _report("agent_eval.json")
    ragas = _report("ragas_scores.json")

    if metrics:
        lstm = metrics.get("results", {}).get("lstm", {})
        cols[0].metric("MRI PR-AUC", f"{lstm.get('classification', {}).get('pr_auc', 0):.3f}")
        cols[1].metric("event recall ≥72 h",
                       f"{lstm.get('events', {}).get('event_recall_in_time', 0):.0%}")
        cols[2].metric("false pages / magnet-month",
                       f"{lstm.get('alerts', {}).get('false_per_magnet_month', 0):.2f}")
        cols[3].metric("net saving / quench",
                       f"${lstm.get('cost', {}).get('net_savings_usd', 0):,.0f}")
    else:
        st.caption("reports/metrics.json not found — run the pipeline.")

    c1, c2, c3 = st.columns(3)
    if fairness:
        c1.metric("fairness verdict", fairness.get("verdict", "?").upper())
        recall = fairness.get("event_recall_by_magnet", {})
        c1.write("event recall (≥72 h): " + ", ".join(
            f"{m}={v:.0%}" for m, v in recall.items() if v is not None))
    if drift:
        c2.metric("drift → retrain?", "yes" if drift.get("retrain_recommended") else "no",
                  f"{drift.get('drifted_share', 0):.0%} features drifted")
    if ext:
        champ = ext.get("results", {}).get(ext.get("champion", ""), {})
        test = champ.get("test", {})
        c3.metric("C-MAPSS PR-AUC (external)",
                  f"{test.get('window', {}).get('pr_auc', 0):.3f}",
                  f"event recall {test.get('events', {}).get('event_recall', 0):.0%}")

    if agent_eval:
        st.markdown(f"**Agent offline eval:** {agent_eval.get('n_passed', 0)}/"
                    f"{agent_eval.get('n_scenarios', 0)} scenarios passed")
    if ragas:
        st.markdown(f"**RAGAS (live judge):** faithfulness {ragas.get('faithfulness', 0):.2f}"
                    f" · answer relevancy {ragas.get('answer_relevancy', 0):.2f}")
    st.caption("Details: docs/model_card.md · docs/eu_ai_act.md · "
               "docs/external_validation.md")
