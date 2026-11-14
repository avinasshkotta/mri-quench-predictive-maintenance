"""Tests for the LangGraph maintenance agent (agent.py).

All LLM behaviour is scripted (ScriptedLLM) — no network, no API key needed.
Real-data tool tests skip gracefully when artifacts are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.agent import (AgentState, SYSTEM_PROMPT, ScriptedLLM,
                               build_graph, create_ticket, doc_search,
                               run_agent)

CFG = {"configurable": {"thread_id": "test-1"}}


def _tc(name: str, args: dict, cid: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                              "id": cid, "type": "tool_call"}])


# --------------------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------------------
def test_doc_search_finds_relevant_chunks():
    hits = doc_search("false page budget threshold policy", k=3)
    assert hits, "RAG returned nothing"
    assert hits[0]["score"] >= hits[1]["score"]          # ranked by relevance
    assert all(h["source"].endswith(".md") for h in hits)
    assert any("threshold" in (h["heading"] + h["text"]).lower() for h in hits)


def test_doc_search_empty_query():
    assert doc_search("   ") == []


# --------------------------------------------------------------------------------------
# ticket tool (the thing behind the HITL gate)
# --------------------------------------------------------------------------------------
def test_create_ticket_writes_markdown(tmp_path, monkeypatch):
    from quenchguard import agent as A

    monkeypatch.setattr(A, "REPORTS_DIR", tmp_path)
    out = json.loads(create_ticket("M2", "high",
                                   "Cold head trending up for 3 days.",
                                   "Schedule bearing inspection within 72 h."))
    path = Path(out["path"])
    assert path.exists() and "high" in path.read_text(encoding="utf-8")


def test_create_ticket_rejects_bad_priority():
    out = create_ticket("M2", "yesterday", "s", "a")
    assert "error" in json.loads(out)


# --------------------------------------------------------------------------------------
# graph behaviour with a scripted LLM
# --------------------------------------------------------------------------------------
def _graph(llm):
    return build_graph(ScriptedLLM(llm))


def test_tool_loop_runs_and_answers():
    g = _graph([
        _tc("doc_search", {"query": "retraining policy"}),
        AIMessage(content="Per the model card, retraining triggers on drift."),
    ])
    out = g.invoke({"messages": [("user", "when do we retrain?")]}, CFG)
    assert len(out["messages"]) == 4                      # human, tool call, tool, answer
    assert "retraining" in out["messages"][-1].content
    tool_msg = out["messages"][2]
    assert tool_msg.type == "tool" and "model_card" in tool_msg.content


def test_system_prompt_gates_every_call():
    prompts = []
    g = _graph([AIMessage(content="ok.")])
    orig = ScriptedLLM.invoke

    def spy(self, messages, **kw):
        prompts.append(messages[0])
        return orig(self, messages, **kw)

    ScriptedLLM.invoke = spy
    try:
        g.invoke({"messages": [("user", "hi")]}, CFG)
    finally:
        ScriptedLLM.invoke = orig
    assert prompts and SYSTEM_PROMPT in prompts[0].content


def test_scope_guardrail_script():
    """The scripted refusal path terminates without any tool call."""
    g = _graph([AIMessage(
        content="Refused: that is outside my scope as a maintenance coordinator. "
                "I only handle MRI magnet predictive maintenance.")])
    out = g.invoke({"messages": [("user", "what dosage for an MRI scan?")]}, CFG)
    assert "scope" in out["messages"][-1].content.lower()


# --------------------------------------------------------------------------------------
# HITL ticket gate
# --------------------------------------------------------------------------------------
def test_ticket_interrupt_blocks_until_approved(tmp_path, monkeypatch):
    from quenchguard import agent as A

    monkeypatch.setattr(A, "REPORTS_DIR", tmp_path)
    g = _graph([
        _tc("score_magnet", {"magnet_id": "M1"}),
        _tc("create_ticket", {"magnet_id": "M1", "priority": "high",
                              "summary": "risk rising", "recommended_action": "inspect"}),
        AIMessage(content="Ticket created."),
    ])

    # first invoke must stop at the gate with an interrupt pending
    result = g.invoke({"messages": [("user", "M1 looks risky, open a ticket")]}, CFG)
    pending = result["__interrupt__"]
    assert pending, "expected a pending human approval"
    payload = pending[0].value
    assert payload["type"] == "ticket_approval"
    assert payload["args"]["priority"] == "high"
    assert not list(tmp_path.glob("tickets/*.md")), "ticket written before approval!"

    # resume with approval -> ticket is created, agent summarises
    final = run_agent(g, "", thread_id="test-1", resume={"approved": True})
    last = final["messages"][-1].content
    assert "Ticket created." in last
    gate_msg = [m for m in final["messages"] if m.type == "tool"
                and getattr(m, "name", "") == "create_ticket"][0]
    assert "APPROVED" in gate_msg.content
    files = list(tmp_path.glob("tickets/*.md"))
    assert len(files) == 1 and "inspect" in files[0].read_text(encoding="utf-8")


def test_ticket_rejection_blocks_creation(tmp_path, monkeypatch):
    from quenchguard import agent as A

    monkeypatch.setattr(A, "REPORTS_DIR", tmp_path)
    g = _graph([
        _tc("create_ticket", {"magnet_id": "M3", "priority": "low",
                              "summary": "mild", "recommended_action": "wait"}),
        AIMessage(content="Understood, ticket was rejected; I summarised instead."),
    ])
    result = g.invoke({"messages": [("user", "open a ticket for M3")]}, CFG)
    assert result["__interrupt__"]

    final = run_agent(g, "", thread_id="test-1",
                      resume={"approved": False, "reason": "insufficient evidence"})
    assert not list(tmp_path.glob("tickets/*.md"))
    gate_msg = [m for m in final["messages"] if m.type == "tool"][0]
    assert "REJECTED" in gate_msg.content and "insufficient evidence" in gate_msg.content
    # agent gets a chance to speak after the rejection
    assert "rejected" in final["messages"][-1].content.lower()


def test_thread_isolation(tmp_path):
    """Two threads run independent conversations (checkpointing keying)."""
    g = _graph([AIMessage(content="done.")])
    a = g.invoke({"messages": [("user", "thread A")]},
                 {"configurable": {"thread_id": "A"}})
    b = g.invoke({"messages": [("user", "thread B")]},
                 {"configurable": {"thread_id": "B"}})
    assert a["messages"][0].content == "thread A"
    assert b["messages"][0].content == "thread B"


# --------------------------------------------------------------------------------------
# real-model tools (skipped without artifacts/gold data)
# --------------------------------------------------------------------------------------
def test_score_magnet_real():
    from quenchguard.agent import score_magnet

    try:
        out = json.loads(score_magnet("MRI-01"))
    except FileNotFoundError:
        pytest.skip("model/gold artifacts not built")
    assert 0.0 <= out["quench_probability"] <= 1.0
    assert len(out["explanation"]) == 3


def test_sensor_summary_real():
    from quenchguard.agent import sensor_summary

    try:
        out = json.loads(sensor_summary("MRI-01", hours=24))
    except (FileNotFoundError, KeyError):
        pytest.skip("silver data not built")
    assert "helium_pressure" in out and out["helium_pressure"]["n_readings"] > 0
    assert out["helium_pressure"]["min"] <= out["helium_pressure"]["max"]


# --------------------------------------------------------------------------------------
# demo-bundle fallback (cloud demo serves data/demo when silver/gold are absent)
# --------------------------------------------------------------------------------------
def test_gold_falls_back_to_demo_bundle(monkeypatch, tmp_path):
    import quenchguard.agent as agent_mod

    demo = tmp_path / "data" / "demo"
    demo.mkdir(parents=True)
    pd.DataFrame({"magnet_id": ["MRI-01"],
                  "window_end": [pd.Timestamp("2026-01-01")],
                  "p_mean": [2.4]}).to_parquet(demo / "features.parquet",
                                               index=False)
    monkeypatch.setattr(agent_mod, "REPO_ROOT", tmp_path)
    frame = agent_mod._gold()
    assert len(frame) == 1
    assert agent_mod.data_file("pressure.parquet") is None


def test_data_file_prefers_real_silver_over_bundle():
    import quenchguard.agent as agent_mod

    got = agent_mod.data_file("pressure.parquet")
    if (agent_mod.REPO_ROOT / "data" / "silver" / "pressure.parquet").exists():
        assert got is not None and "silver" in str(got)
    elif (agent_mod.REPO_ROOT / "data" / "demo" / "pressure.parquet").exists():
        assert got is not None and "demo" in str(got)
    else:
        assert got is None


def test_sensor_summary_works_from_bundle_only(monkeypatch, tmp_path):
    import quenchguard.agent as agent_mod

    demo = tmp_path / "data" / "demo"
    demo.mkdir(parents=True)
    hours = pd.date_range("2026-01-01", periods=48, freq="1h")
    pd.DataFrame({"timestamp": hours, "magnet_id": "MRI-01",
                  "pressure_mpa": [2.4 + 0.01 * i for i in range(48)]}).to_parquet(
        demo / "pressure.parquet", index=False)
    monkeypatch.setattr(agent_mod, "REPO_ROOT", tmp_path)
    out = json.loads(agent_mod.sensor_summary("MRI-01", hours=24))
    assert out["magnet_id"] == "MRI-01"
    # the summary window is inclusive at both ends: 25 hourly points
    assert out["helium_pressure"]["n_readings"] == 25
