"""Customer-facing web API: streaming agent chat + live fleet data + static UI.

This module is the single entrypoint for the product UI:

* ``POST /api/chat`` and ``POST /api/chat/resume`` stream the LangGraph agent
  over Server-Sent Events (tokens, tool activity, RAG citations, HITL
  interrupts) so the browser can render answers live.
* ``GET /api/fleet``, ``GET /api/magnet/{id}`` and
  ``GET /api/magnet/{id}/telemetry`` feed the visual side panel.
* ``GET /api/status`` tells the UI whether the LLM key / artifacts / data
  are available before the first question.
* The model-serving routes from :mod:`quenchguard.serving` (``/predict``,
  ``/health``, ``/model-info``) stay mounted unchanged, and the **Next.js
  static export** in ``frontend/out`` (built by ``npm run build``) is served
  at ``/`` when present.

Design notes
------------
* The graph uses a ``MemorySaver`` checkpointer: one ``thread_id`` per
  browser conversation keeps the HITL ``interrupt()`` resumable.
* ``llm`` is injectable so tests can drive the real graph with
  :class:`quenchguard.agent.ScriptedLLM` (no network, no key).
* Chat is streamed with a **sync** generator: the agent's nodes are sync
  (LangChain + torch), and Starlette iterates sync generators in a
  threadpool, keeping the event loop free.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import build_graph, data_file, fleet_risk, score_magnet
from .config import MODELS_DIR, REPO_ROOT
from .llmops import DEFAULT_MODEL, build_chat_model, load_env
from .serving import create_app as create_serving_app

load_env()                     # repo .env (git-ignored); real env vars win

UI_DIR = REPO_ROOT / "frontend" / "out"
FLEET_TTL_SECONDS = float(os.environ.get("QUENCHGUARD_FLEET_TTL", "120"))


# --------------------------------------------------------------------------------------
# request schemas
# --------------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(min_length=1, max_length=128)


class ResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    approved: bool
    reason: str | None = Field(default=None, max_length=1000)


# --------------------------------------------------------------------------------------
# SSE helpers
# --------------------------------------------------------------------------------------
def _sse(event: str, data: Any) -> str:
    """One Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _text_of(content: Any) -> str:
    """Normalise message content (str, or list of content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content
                       if isinstance(part, dict))
    return ""


def _citations(tool_content: str) -> list[dict[str, Any]]:
    """Extract RAG citation cards from a ``doc_search`` tool result."""
    try:
        hits = json.loads(tool_content)
    except (TypeError, ValueError):
        return []
    out: list[dict[str, Any]] = []
    if isinstance(hits, list):
        for hit in hits:
            if isinstance(hit, dict) and hit.get("source"):
                try:
                    score = round(float(hit.get("score", 0.0)), 4)
                except (TypeError, ValueError):
                    score = 0.0
                out.append({"source": hit["source"],
                            "heading": hit.get("heading", ""),
                            "score": score})
    return out


def _repair_thread(graph: Any, config: dict[str, Any]) -> list[Any]:
    """Close tool calls left dangling by a client-aborted stream.

    An aborted SSE connection stops the graph between ``agent`` and ``tools``;
    the checkpoint then ends with an AIMessage carrying unanswered
    ``tool_calls``, and the next user message would make OpenAI reject the
    conversation (400: tool_calls must be followed by tool messages). We
    append cancellation ToolMessages so the thread stays usable. A *pending
    human approval* is different: the caller must resume it explicitly.
    """
    from langchain_core.messages import ToolMessage

    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        return []
    if getattr(snapshot, "interrupts", ()):
        raise HTTPException(
            status_code=409,
            detail="a ticket approval is pending for this thread — approve or reject it first")
    messages = snapshot.values.get("messages") or []
    if not messages:
        return []
    last = messages[-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return []
    answered = {getattr(m, "tool_call_id", None) for m in messages
                if getattr(m, "type", "") == "tool"}
    return [ToolMessage(content="[CANCELLED by the user before execution]",
                        tool_call_id=call["id"],
                        name=call.get("name") or "tool")
            for call in calls if call.get("id") not in answered]


# --------------------------------------------------------------------------------------
# app factory
# --------------------------------------------------------------------------------------
def create_app(llm: Any | None = None,
               artifacts: Any | None = None) -> FastAPI:
    """Build the customer API. ``llm`` is injectable for deterministic tests."""
    app = FastAPI(
        title="QuenchGuard",
        description=("MRI magnet quench early warning — streaming agent chat, "
                     "live fleet risk, SHAP explanations and the model API."),
        version="2.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.environ.get(
            "UI_ALLOW_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict[str, Any] = {"llm": llm, "graph": None}
    fleet_cache: dict[str, Any] = {"at": 0.0, "payload": None}

    # ---- the agent ------------------------------------------------------------------
    def _graph() -> Any:
        if state["graph"] is None:
            if state["llm"] is None:
                if not os.environ.get("OPENAI_API_KEY"):
                    raise HTTPException(
                        status_code=503,
                        detail=("OPENAI_API_KEY is not configured on the server. "
                                "Set it in the environment (or repo .env) and restart."))
                # streaming=True makes .invoke() emit token callbacks, which
                # LangGraph's "messages" stream mode forwards to the browser;
                # build_chat_model adds retries/timeout/token ceiling.
                state["llm"] = build_chat_model(streaming=True)
            state["graph"] = build_graph(state["llm"])
        return state["graph"]

    def _stream(payload: Any, config: dict[str, Any],
                thread_id: str) -> Iterator[str]:
        """Translate LangGraph stream events into SSE frames."""
        from langchain_core.messages import AIMessageChunk

        yield _sse("meta", {"thread_id": thread_id, "model": DEFAULT_MODEL})
        emitted_this_turn = 0
        try:
            for mode, data in _graph().stream(payload, config,
                                              stream_mode=["messages", "updates"]):
                if mode == "messages":
                    chunk, meta = data
                    if (meta or {}).get("langgraph_node") != "agent":
                        continue
                    text = _text_of(getattr(chunk, "content", ""))
                    if isinstance(chunk, AIMessageChunk):
                        # real LLM streaming: one event per token chunk
                        if text:
                            emitted_this_turn += 1
                            yield _sse("token", {"text": text})
                    elif text and emitted_this_turn == 0:
                        # non-streaming LLM: LangGraph replays the whole
                        # AIMessage at node end — emit it once as one token
                        emitted_this_turn += 1
                        yield _sse("token", {"text": text})
                    continue

                for node, update in (data or {}).items():
                    if node == "__interrupt__":
                        interrupts = update if isinstance(update, (list, tuple)) else [update]
                        intr = interrupts[0]
                        value = getattr(intr, "value", intr)
                        yield _sse("interrupt", value)
                        yield _sse("halted", {"reason": "human_approval_required"})
                        return

                    messages = (update or {}).get("messages") or []
                    if node == "agent":
                        last = messages[-1] if messages else None
                        calls = getattr(last, "tool_calls", None) or []
                        for call in calls:
                            yield _sse("tool_start", {"name": call.get("name"),
                                                      "args": call.get("args") or {}})
                        # Fallback: a non-streaming LLM (or ScriptedLLM) produced
                        # content without token callbacks — emit it once.
                        if not calls and emitted_this_turn == 0 and last is not None:
                            text = _text_of(getattr(last, "content", ""))
                            if text:
                                yield _sse("token", {"text": text})
                        emitted_this_turn = 0
                    elif node in ("tools", "ticket_gate"):
                        for msg in messages:
                            content = msg.get("content", "") if isinstance(msg, dict) \
                                else getattr(msg, "content", "")
                            name = (msg.get("name") if isinstance(msg, dict)
                                    else getattr(msg, "name", None)) or node
                            text = _text_of(content) if not isinstance(content, str) \
                                else content
                            yield _sse("tool_result", {
                                "name": name,
                                "content": text[:1500],
                                "truncated": len(text) > 1500,
                            })
                            if name == "doc_search":
                                for citation in _citations(text):
                                    yield _sse("citation", citation)
            yield _sse("done", {})
        except HTTPException as exc:
            yield _sse("error", {"message": str(exc.detail)})
        except Exception as exc:                       # surfaced to the browser
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    def _sse_response(payload: Any, thread_id: str,
                      require_pending: bool = False) -> StreamingResponse:
        config = {"configurable": {"thread_id": thread_id}}
        graph = _graph()                           # fail fast (503) before streaming
        if require_pending:
            snapshot = graph.get_state(config)
            if not snapshot or not getattr(snapshot, "next", ()):
                raise HTTPException(
                    status_code=409,
                    detail="no pending human approval for this thread")
        return StreamingResponse(
            _stream(payload, config, thread_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"})

    # ---- chat routes ----------------------------------------------------------------
    api = APIRouter(prefix="/api")

    @api.get("/status")
    def status() -> dict[str, Any]:
        from .notify import status as email_status

        silver = REPO_ROOT / "data" / "silver" / "pressure.parquet"
        demo = REPO_ROOT / "data" / "demo" / "pressure.parquet"
        return {
            "status": "ok",
            "llm_configured": bool(os.environ.get("OPENAI_API_KEY")),
            "model": DEFAULT_MODEL,
            "artifacts_loaded": (MODELS_DIR / "lstm_detector.pt").exists(),
            "data_source": "silver" if silver.exists() else
                           ("demo" if demo.exists() else "none"),
            **email_status(),
        }

    @api.post("/chat")
    def chat(req: ChatRequest) -> StreamingResponse:
        from langchain_core.messages import HumanMessage

        graph = _graph()
        config = {"configurable": {"thread_id": req.thread_id}}
        repairs = _repair_thread(graph, config)
        return _sse_response(
            {"messages": [*repairs, HumanMessage(content=req.message)]}, req.thread_id)

    @api.post("/chat/resume")
    def chat_resume(req: ResumeRequest) -> StreamingResponse:
        from langgraph.types import Command

        decision: dict[str, Any] = {"approved": req.approved}
        if req.reason:
            decision["reason"] = req.reason
        return _sse_response(Command(resume=decision), req.thread_id,
                             require_pending=True)

    # ---- visual panels --------------------------------------------------------------
    @api.get("/fleet")
    def fleet() -> dict[str, Any]:
        now = time.monotonic()
        if fleet_cache["payload"] is not None and \
                now - fleet_cache["at"] < FLEET_TTL_SECONDS:
            return {**fleet_cache["payload"], "cached": True}
        try:
            body = json.loads(fleet_risk())
            from .agent import _artifacts      # threshold is a frozen artifact value

            body["threshold"] = round(float(_artifacts().threshold), 6)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        fleet_cache.update({"at": now, "payload": body})
        return {**body, "cached": False}

    @api.get("/magnet/{magnet_id}")
    def magnet(magnet_id: str) -> dict[str, Any]:
        try:
            body = json.loads(score_magnet(magnet_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if "error" in body:
            raise HTTPException(status_code=404, detail=body["error"])
        return body

    @api.get("/magnet/{magnet_id}/telemetry")
    def telemetry(magnet_id: str, hours: int = 72) -> dict[str, Any]:
        hours = max(1, min(int(hours), 24 * 90))
        specs = (("pressure.parquet", "timestamp", "pressure_mpa", "pressure"),
                 ("cold_head.parquet", "timestamp", "cold_head_k", "cold_head"),
                 ("vibration.parquet", "window_end", "rms_g", "vibration"))
        out: dict[str, Any] = {"magnet_id": magnet_id, "hours": hours}
        for fname, ts_col, val_col, key in specs:
            path = data_file(fname)
            points: list[dict[str, Any]] = []
            if path is not None:
                frame = pd.read_parquet(path)
                frame = frame[frame["magnet_id"] == magnet_id].sort_values(ts_col)
                if len(frame):
                    cut = frame[ts_col].max() - pd.Timedelta(hours=hours)
                    frame = frame[frame[ts_col] >= cut]
                    step = max(1, len(frame) // 300)      # keep payloads small
                    frame = frame.iloc[::step]
                    points = [{"t": pd.Timestamp(ts).isoformat(),
                               "v": round(float(val), 4)}
                              for ts, val in zip(frame[ts_col], frame[val_col])]
            out[key] = points
        return out

    app.include_router(api)

    # ---- model-serving routes (unchanged contract: /predict, /health, /model-info) ---
    serving = create_serving_app(artifacts=artifacts)
    app.routes.extend(r for r in serving.routes
                      if type(r).__name__ == "APIRoute")

    # ---- static customer UI (Next.js export) — mounted last -------------------------
    if UI_DIR.exists():
        app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

    return app


# module-level app for `uvicorn quenchguard.webui:app`
app = create_app()

__all__ = ["create_app", "app", "ChatRequest", "ResumeRequest"]
