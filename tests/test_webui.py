"""Customer web API tests: SSE chat streaming, HITL resume, visual panels.

The graph is driven by :class:`quenchguard.agent.ScriptedLLM`, so these tests
exercise the *real* streaming translation layer, tool loop, RAG citations and
the human-in-the-loop interrupt without any network or OpenAI key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quenchguard.agent import ScriptedLLM       # noqa: E402
from quenchguard.webui import create_app        # noqa: E402

HAVE_ARTIFACTS = (REPO / "models" / "lstm_detector.pt").exists()
HAVE_DATA = ((REPO / "data" / "gold" / "features.parquet").exists()
             or (REPO / "data" / "demo" / "features.parquet").exists())


def _tc(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def _client(script: list) -> TestClient:
    return TestClient(create_app(llm=ScriptedLLM(script)))


def _events(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into ``[(event, data_json), ...]``."""
    out: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        event, data = None, []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].strip())
        if event and data:
            out.append((event, json.loads("\n".join(data))))
    return out


# --------------------------------------------------------------------------------------
# status / capabilities
# --------------------------------------------------------------------------------------
def test_status_reports_capabilities():
    response = TestClient(create_app(llm=ScriptedLLM([]))).get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body) >= {"llm_configured", "model", "artifacts_loaded", "data_source"}
    assert body["data_source"] in {"silver", "demo", "none"}
    assert "ticket_email_configured" in body


def test_model_routes_stay_mounted():
    """/health and /predict must survive the web UI app composition."""
    response = TestClient(create_app(llm=ScriptedLLM([]))).get("/health")
    assert response.status_code == 200
    assert {"status", "model_loaded"} <= set(response.json())


# --------------------------------------------------------------------------------------
# streaming chat
# --------------------------------------------------------------------------------------
def test_chat_streams_answer_with_meta_and_done():
    client = _client([AIMessage(content="MRI-05 is highest, well below threshold.")])
    response = client.post("/api/chat",
                           json={"message": "highest risk?", "thread_id": "t1"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _events(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "meta"
    assert events[0][1]["thread_id"] == "t1"
    assert kinds[-1] == "done"
    text = "".join(data["text"] for kind, data in events if kind == "token")
    assert "MRI-05" in text


def test_chat_emits_tool_activity_and_rag_citations():
    client = _client([
        _tc("doc_search", {"query": "quench risk", "k": 3}),
        AIMessage(content="According to the model card, the model is synthetic-data trained."),
    ])
    response = client.post(
        "/api/chat", json={"message": "what do the docs say?",
                           "thread_id": "t2"})
    events = _events(response.text)

    starts = [data["name"] for kind, data in events if kind == "tool_start"]
    assert "doc_search" in starts
    assert any(kind == "tool_result" and data["name"] == "doc_search"
               for kind, data in events)
    citations = [data for kind, data in events if kind == "citation"]
    assert citations and citations[0]["source"].endswith(".md")
    assert {"source", "heading", "score"} <= set(citations[0])
    text = "".join(data["text"] for kind, data in events if kind == "token")
    assert "model card" in text


def test_empty_message_is_rejected():
    client = _client([])
    assert client.post("/api/chat",
                       json={"message": "", "thread_id": "t"}).status_code == 422


def test_chat_503_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = TestClient(create_app()).post(
        "/api/chat", json={"message": "hello", "thread_id": "t"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_static_export_is_served_at_root(tmp_path, monkeypatch):
    """When frontend/out exists (production build), / serves it — APIs stay first."""
    from quenchguard import webui as W

    out = tmp_path / "frontend" / "out"
    out.mkdir(parents=True)
    (out / "index.html").write_text("<html><body>QuenchGuard UI</body></html>",
                                    encoding="utf-8")
    monkeypatch.setattr(W, "UI_DIR", out)
    client = TestClient(W.create_app(llm=ScriptedLLM([])))
    root = client.get("/")
    assert root.status_code == 200 and "QuenchGuard UI" in root.text
    assert client.get("/api/status").json()["status"] == "ok"


# --------------------------------------------------------------------------------------
# human-in-the-loop ticket gate
# --------------------------------------------------------------------------------------
def _ticket_call(priority: str = "low") -> AIMessage:
    return _tc("create_ticket", {"magnet_id": "MRI-01", "priority": priority,
                                 "summary": "stable but cold head trending.",
                                 "recommended_action": "monitor closely"})


def test_interrupt_then_approval_creates_ticket(tmp_path, monkeypatch):
    from quenchguard import agent as A

    monkeypatch.setattr(A, "REPORTS_DIR", tmp_path)
    client = _client([_ticket_call(),
                      AIMessage(content="Ticket created for MRI-01 after approval.")])

    first = client.post("/api/chat",
                        json={"message": "open a ticket", "thread_id": "t3"})
    events = _events(first.text)
    kinds = [kind for kind, _ in events]
    assert "interrupt" in kinds and "halted" in kinds
    assert "done" not in kinds
    interrupt = next(data for kind, data in events if kind == "interrupt")
    assert interrupt["type"] == "ticket_approval"
    assert interrupt["args"]["priority"] == "low"
    assert not list(tmp_path.glob("tickets/*.md")), "ticket written before approval!"

    second = client.post("/api/chat/resume",
                         json={"thread_id": "t3", "approved": True})
    events2 = _events(second.text)
    assert "done" in [kind for kind, _ in events2]
    text = "".join(data["text"] for kind, data in events2 if kind == "token")
    assert "Ticket created" in text
    files = list(tmp_path.glob("tickets/*.md"))
    assert len(files) == 1 and "monitor closely" in files[0].read_text(encoding="utf-8")


def test_resume_without_pending_approval_is_409():
    """A stale/unknown thread must not silently produce an empty stream."""
    client = _client([])
    response = client.post("/api/chat/resume",
                           json={"thread_id": "never-interrupted", "approved": True})
    assert response.status_code == 409
    assert "no pending human approval" in response.json()["detail"]


def test_repair_thread_closes_dangling_tool_calls():
    """A client-aborted stream leaves unanswered tool_calls; the next turn
    must repair the checkpoint instead of 400ing at OpenAI."""
    from quenchguard.agent import build_graph
    from quenchguard.webui import _repair_thread

    graph = build_graph(ScriptedLLM([]))
    config = {"configurable": {"thread_id": "aborted"}}
    graph.update_state(config, {"messages": [AIMessage(content="", tool_calls=[
        {"name": "doc_search", "args": {"query": "q"}, "id": "call_x",
         "type": "tool_call"}])]})
    repairs = _repair_thread(graph, config)
    assert len(repairs) == 1
    assert repairs[0].tool_call_id == "call_x"
    assert "CANCELLED" in repairs[0].content


def test_repair_thread_blocks_new_message_while_approval_pending(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from quenchguard import agent as A
    from quenchguard.agent import build_graph
    from quenchguard.webui import _repair_thread

    monkeypatch.setattr(A, "REPORTS_DIR", tmp_path)
    graph = build_graph(ScriptedLLM([_ticket_call()]))
    config = {"configurable": {"thread_id": "pending"}}
    graph.invoke({"messages": [("user", "open a ticket")]}, config)
    with pytest.raises(HTTPException) as excinfo:
        _repair_thread(graph, config)
    assert excinfo.value.status_code == 409
    assert "approval is pending" in excinfo.value.detail


def test_interrupt_then_rejection_blocks_ticket(tmp_path, monkeypatch):
    from quenchguard import agent as A

    monkeypatch.setattr(A, "REPORTS_DIR", tmp_path)
    client = _client([_ticket_call("high"),
                      AIMessage(content="Understood — ticket rejected, summarising instead.")])

    client.post("/api/chat", json={"message": "open a ticket", "thread_id": "t4"})
    second = client.post("/api/chat/resume", json={
        "thread_id": "t4", "approved": False, "reason": "insufficient evidence"})
    events = _events(second.text)

    assert not list(tmp_path.glob("tickets/*.md"))
    results = [data for kind, data in events if kind == "tool_result"]
    assert any("REJECTED" in r["content"] and "insufficient evidence" in r["content"]
               for r in results)
    text = "".join(data["text"] for kind, data in events if kind == "token")
    assert "rejected" in text.lower()


# --------------------------------------------------------------------------------------
# visual panels (need the committed artifacts + a data bundle)
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(not (HAVE_ARTIFACTS and HAVE_DATA),
                    reason="serving artifacts / data bundle not built")
def test_fleet_panel_endpoint_and_cache():
    client = _client([])
    response = client.get("/api/fleet")
    assert response.status_code == 200
    body = response.json()
    assert body["fleet"], "fleet is empty"
    assert {"magnet_id", "quench_probability", "alert"} <= set(body["fleet"][0])
    assert 0.0 <= body["threshold"] <= 1.0
    assert body["cached"] is False
    again = client.get("/api/fleet").json()
    assert again["cached"] is True
    assert again["fleet"] == body["fleet"]


@pytest.mark.skipif(not (HAVE_ARTIFACTS and HAVE_DATA),
                    reason="serving artifacts / data bundle not built")
def test_magnet_detail_and_telemetry_endpoints():
    client = _client([])
    detail = client.get("/api/magnet/MRI-01")
    assert detail.status_code == 200
    body = detail.json()
    assert 0.0 <= body["quench_probability"] <= 1.0
    assert len(body["explanation"]) == 3
    assert client.get("/api/magnet/NOT-A-MAGNET").status_code == 404

    telemetry = client.get("/api/magnet/MRI-01/telemetry?hours=24")
    assert telemetry.status_code == 200
    payload = telemetry.json()
    assert set(payload) >= {"magnet_id", "hours", "pressure", "cold_head", "vibration"}
    if payload["pressure"]:
        assert {"t", "v"} <= set(payload["pressure"][0])
