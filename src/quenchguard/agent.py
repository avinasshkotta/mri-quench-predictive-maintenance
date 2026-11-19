"""QuenchGuard maintenance agent — LangGraph tool loop + RAG + HITL ticket gate.

The business story (from the course capstone track): the LSTM raises an alert;
a **maintenance coordinator agent** investigates it and drafts a service
ticket. The agent must never act on its own — ticket creation sits behind a
LangGraph ``interrupt`` (human approval), enforcing EU AI Act Art. 14 human
oversight at the orchestration level.

Architecture (all LLM-agnostic — tests run on a scripted fake):

``START -> agent (LLM, tools bound) -> [route]
              |-> tools        (investigation: scores, sensors, doc search)
              |-> ticket_gate  (HITL: interrupt() before create_ticket)
              |-> END          (plain answer)
       tools -> agent
       ticket_gate -> agent``

* **Guardrail** is prompt-level (system prompt defines scope: predictive
  maintenance for MRI magnets only) — the demo script shows refusal.
* **RAG** is deterministic and local: TF-IDF over chunked repo docs
  (model card, case study, README) — no embedding API, fully reproducible.
* **Checkpointing**: pass any LangGraph checkpointer (``MemorySaver`` default)
  keyed by ``thread_id`` so investigations survive interruption.

Real usage: ``scripts/agent_demo.py`` with ``OPENAI_API_KEY`` set
(``gpt-4o-mini``). Tests inject :class:`ScriptedLLM`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from .config import REPORTS_DIR, REPO_ROOT

# --------------------------------------------------------------------------------------
# local RAG index (TF-IDF over repo docs — deterministic, no API cost)
# --------------------------------------------------------------------------------------
_DOC_FILES = (
    REPO_ROOT / "docs" / "model_card.md",
    REPO_ROOT / "docs" / "case-study-rctc.md",
    REPO_ROOT / "README.md",
)

_rag_cache: dict[str, Any] | None = None


def _chunk_markdown(path: Path) -> list[dict[str, str]]:
    """Split a markdown doc into retrievable chunks at '## ' boundaries."""
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks: list[dict[str, str]] = []
    sections = text.split("\n## ")
    for i, sec in enumerate(sections):
        if i == 0 and not sec.startswith("# "):
            heading = path.stem
        else:
            first_line, _, rest = sec.partition("\n")
            heading = first_line.lstrip("# ")
            sec = rest if rest else first_line
        body = sec.strip()
        for start in range(0, len(body), 1200):        # long sections -> slices
            part = body[start:start + 1200]
            if len(part.strip()) > 40:
                chunks.append({"source": path.name, "heading": heading, "text": part})
    return chunks


def _rag_index() -> dict[str, Any]:
    """Fit the TF-IDF index once; cache it module-wide."""
    global _rag_cache
    if _rag_cache is not None:
        return _rag_cache
    from sklearn.feature_extraction.text import TfidfVectorizer

    chunks = [c for f in _DOC_FILES if f.exists() for c in _chunk_markdown(f)]
    texts = [f"{c['heading']}\n{c['text']}" for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                                 ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    _rag_cache = {"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}
    return _rag_cache


def doc_search(query: str, k: int = 3) -> list[dict[str, str]]:
    """Top-k doc chunks by TF-IDF cosine similarity."""
    if not query.strip():
        return []
    idx = _rag_index()
    if not idx["chunks"]:
        return []
    sims = (idx["vectorizer"].transform([query]) @ idx["matrix"].T).toarray().ravel()
    top = np.argsort(sims)[::-1][:k]
    return [{**idx["chunks"][i], "score": float(sims[i])}
            for i in top if sims[i] > 0]


# --------------------------------------------------------------------------------------
# real-model tools (plain functions; wrapped with @tool in build_graph)
# --------------------------------------------------------------------------------------
_artifacts_cache = None


def _artifacts():
    global _artifacts_cache
    if _artifacts_cache is None:
        from .serving import Artifacts

        _artifacts_cache = Artifacts.load()      # LSTM + scaler + HistGBDT
    return _artifacts_cache


def data_file(fname: str) -> Path | None:
    """Locate a data file: real silver layer first, then the demo bundle.

    The committed bundle (``data/demo/``, built by
    ``scripts/build_demo_bundle.py``) lets the public cloud demo serve real
    model + telemetry paths without the git-ignored medallion data.
    """
    for base in ("silver", "demo"):
        path = REPO_ROOT / "data" / base / fname
        if path.exists():
            return path
    return None


def _gold() -> pd.DataFrame:
    for path in (REPO_ROOT / "data" / "gold" / "features.parquet",
                 REPO_ROOT / "data" / "demo" / "features.parquet"):
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError(
        "gold features missing — run scripts/run_pipeline.py "
        "(or scripts/build_demo_bundle.py for the cloud demo)")


def fleet_risk() -> str:
    """Score the latest window of every magnet with the champion LSTM."""
    art = _artifacts()
    df = _gold().sort_values(["magnet_id", "window_end"])
    rows = []
    for magnet, part in df.groupby("magnet_id"):
        window = part.iloc[-art.seq_len:][art.feature_names].values
        if len(window) < art.seq_len:
            continue
        out = art.predict(window)
        rows.append({"magnet_id": magnet,
                     "quench_probability": out["quench_probability"],
                     "alert": out["alert"]})
    rows.sort(key=lambda r: r["quench_probability"], reverse=True)
    return json.dumps({"fleet": rows, "n_scored": len(rows)})


def score_magnet(magnet_id: str) -> str:
    """Latest-window LSTM probability + SHAP top factors for one magnet."""
    art = _artifacts()
    df = _gold()
    part = df[df.magnet_id == magnet_id].sort_values("window_end")
    if part.empty:
        return json.dumps({"error": f"unknown magnet_id {magnet_id!r}",
                           "known": sorted(df.magnet_id.unique().tolist())})
    window = part.iloc[-art.seq_len:][art.feature_names].values
    out = art.predict(window)
    out["magnet_id"] = magnet_id
    out["window_end"] = str(part["window_end"].iloc[-1])
    return json.dumps(out)


def sensor_summary(magnet_id: str, hours: float = 24.0) -> str:
    """Recent telemetry statistics for one magnet from the silver layer."""
    out: dict[str, Any] = {"magnet_id": magnet_id, "window_hours": hours}
    specs = [("pressure.parquet", "timestamp", "pressure_mpa", "helium_pressure"),
             ("cold_head.parquet", "timestamp", "cold_head_k", "cold_head_k")]
    for fname, ts_col, val_col, key in specs:
        path = data_file(fname)
        if path is None:
            continue
        df = pd.read_parquet(path)
        df = df[df.magnet_id == magnet_id].sort_values(ts_col)
        cut = df[ts_col].max() - pd.Timedelta(hours=hours)
        recent = df[df[ts_col] >= cut][val_col]
        if len(recent):
            out[key] = {"min": round(float(recent.min()), 4),
                        "max": round(float(recent.max()), 4),
                        "mean": round(float(recent.mean()), 4),
                        "last": round(float(df[val_col].iloc[-1]), 4),
                        "n_readings": int(len(recent))}
    vib_path = data_file("vibration.parquet")
    if vib_path is not None:
        df = pd.read_parquet(vib_path)
        df = df[df.magnet_id == magnet_id].sort_values("window_end")
        cut = df["window_end"].max() - pd.Timedelta(hours=hours)
        recent = df[df["window_end"] >= cut]
        if len(recent):
            out["vibration"] = {
                "rms_g_max": round(float(recent.rms_g.max()), 4),
                "fault_ratio_max": round(float(recent.fault_ratio.max()), 4),
                "n_windows": int(len(recent))}
    return json.dumps(out)


def create_ticket(magnet_id: str, priority: str, summary: str,
                  recommended_action: str) -> str:
    """Write a service ticket file (ONLY called after human approval)."""
    if priority.lower() not in {"low", "medium", "high", "critical"}:
        return json.dumps({"error": f"invalid priority {priority!r}"})
    tickets = REPORTS_DIR / "tickets"
    tickets.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    created = datetime.now(timezone.utc).isoformat()
    path = tickets / f"{stamp}_{magnet_id}.md"
    path.write_text(
        f"# Service ticket — {magnet_id}\n\n"
        f"- created: {created}\n"
        f"- priority: **{priority}**\n"
        f"- approved by: human operator (HITL gate)\n\n"
        f"## Summary\n{summary}\n\n"
        f"## Recommended action\n{recommended_action}\n",
        encoding="utf-8")
    # Email alert after the file exists: the ticket file is the audit record,
    # so a mail outage must never lose or block a ticket.
    try:
        from .notify import notify_ticket

        notify_ticket(magnet_id, priority, summary, recommended_action,
                      str(path), created)
    except Exception:                              # noqa: BLE001 — fail-safe
        pass
    return json.dumps({"created": True, "path": str(path)})


# --------------------------------------------------------------------------------------
# the LangGraph agent
# --------------------------------------------------------------------------------------
class AgentState(TypedDict):
    """ReAct-style message state (LangGraph ``add_messages`` reducer)."""

    messages: Annotated[list, add_messages]


INVESTIGATION_TOOLS = [tool(fleet_risk), tool(score_magnet),
                       tool(sensor_summary), tool(doc_search)]
TICKET_TOOL = tool(create_ticket)
ALL_TOOLS = INVESTIGATION_TOOLS + [TICKET_TOOL]
TOOL_REGISTRY = {t.name: t for t in ALL_TOOLS}

SYSTEM_PROMPT = """You are QuenchGuard's maintenance coordinator agent for an MRI \
magnet fleet. Your job: investigate quench-risk alerts and draft service tickets.

Scope (guardrail): you ONLY handle predictive-maintenance questions about the MRI \
magnet fleet (sensor health, quench risk, service planning, documentation of this \
system). Politely refuse anything else (clinical advice, unrelated tasks, changing \
sensor hardware) and explain why in one sentence.

Rules:
1. Investigate before recommending: call score_magnet / fleet_risk for model output, \
sensor_summary for raw telemetry, search_docs for system documentation. Cite what \
you used.
2. The model output is evidence, not a decision: present probability, threshold and \
the SHAP factors, then reason about physics (helium pressure drifting below 2.0 MPa, \
cold head above 5.5 K, vibration RMS rising).
3. To open a service ticket, call create_ticket with a clear priority (low, medium, \
high, critical), a physics-grounded summary and one recommended action. A human \
operator must approve it — that happens automatically, do not claim approval yourself.
4. Never invent sensor numbers: everything must come from a tool result.
"""


def build_graph(llm: Any, checkpointer: Any | None = None) -> Any:
    """Compile the agent graph. ``llm`` must support ``.bind_tools`` (ChatOpenAI,
    or ScriptedLLM for tests)."""
    def agent_node(state: AgentState) -> dict:
        llm_tools = llm.bind_tools(ALL_TOOLS)
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        return {"messages": [llm_tools.invoke(msgs)]}

    def ticket_gate_node(state: AgentState) -> dict:
        """HITL gate: interrupt() on every create_ticket call before executing."""
        last = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        for call in (last.tool_calls or []):
            if call["name"] != "create_ticket":
                tool_messages.append(ToolMessage(
                    content=json.dumps(TOOL_REGISTRY[call["name"]].invoke(call["args"])),
                    tool_call_id=call["id"], name=call["name"]))
                continue
            decision = interrupt({                                  # <- HITL
                "type": "ticket_approval",
                "call_id": call["id"],
                "args": call["args"],
                "question": "A human operator must approve this service ticket.",
            })
            if decision.get("approved"):
                result = TICKET_TOOL.invoke(call["args"])
                tool_messages.append(ToolMessage(
                    content=f"[APPROVED by human operator] {result}",
                    tool_call_id=call["id"], name="create_ticket"))
            else:
                tool_messages.append(ToolMessage(
                    content=f"[REJECTED by human operator: "
                            f"{decision.get('reason', 'no reason given')}] "
                            "Ticket NOT created. Do not re-create it without new "
                            "evidence; summarise the rejection instead.",
                    tool_call_id=call["id"], name="create_ticket"))
        return {"messages": tool_messages}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return END
        if any(c["name"] == "create_ticket" for c in calls):
            return "ticket_gate"
        return "tools"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(INVESTIGATION_TOOLS))
    builder.add_node("ticket_gate", ticket_gate_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools",
                                                   "ticket_gate": "ticket_gate",
                                                   END: END})
    builder.add_edge("tools", "agent")
    builder.add_edge("ticket_gate", "agent")
    return builder.compile(checkpointer=checkpointer or MemorySaver())


class ScriptedLLM:
    """Deterministic LLM double for tests: replays scripted AIMessages."""

    def __init__(self, script: list[Any]):
        from langchain_core.messages import AIMessage

        self._script = list(script)
        self.prompts: list[list] = []
        self._ai = AIMessage

    def bind_tools(self, tools: Any) -> "ScriptedLLM":
        self._tools = tools
        return self

    def invoke(self, messages: list, **_kw: Any):
        self.prompts.append(list(messages))
        if self._script:
            return self._script.pop(0)
        return self._ai(content="done")


def run_agent(graph: Any, user_message: str, thread_id: str = "demo",
              resume: dict | None = None) -> Any:
    """Invoke (or resume) the graph for one conversation turn."""
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    if resume is not None:
        return graph.invoke(Command(resume=resume), config)
    return graph.invoke({"messages": [HumanMessage(content=user_message)]}, config)


__all__ = ["build_graph", "ScriptedLLM", "run_agent", "doc_search",
           "fleet_risk", "score_magnet", "sensor_summary", "create_ticket",
           "SYSTEM_PROMPT"]
