"""Tests for LLMOps: token ledger, tracing config, eval rows, agent eval harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quenchguard.llmops import (PRICE_TABLE, TokenLedger, configure_tracing,
                                estimate_cost, transcript_to_eval_rows)


# --------------------------------------------------------------------------------------
# cost accounting
# --------------------------------------------------------------------------------------
def test_estimate_cost_uses_price_table():
    cost = estimate_cost("gpt-4o-mini", 1000, 500)
    pin, pout = PRICE_TABLE["gpt-4o-mini"]
    assert cost == pytest.approx((1000 * pin + 500 * pout) / 1e6)


def test_estimate_cost_zero_for_unknown_model():
    assert estimate_cost("totally-made-up", 1000, 1000) == 0.0


def test_ledger_totals_and_save(tmp_path):
    ledger = TokenLedger()
    ledger.record("gpt-4o-mini", 1000, 500, "s1")
    ledger.record("gpt-4o-mini", 2000, 1000, "s2")
    ledger.record("obscure-model", 100, 100, "s2")

    totals = ledger.totals
    assert totals["n_calls"] == 3
    assert totals["prompt_tokens"] == 3100
    assert totals["completion_tokens"] == 1600
    assert totals["cost_usd"] == pytest.approx(
        (1000 * 0.15 + 500 * 0.6 + 2000 * 0.15 + 1000 * 0.6) / 1e6, rel=1e-3)
    assert totals["unpriced_models"] == ["obscure-model"]
    assert totals["by_model"]["gpt-4o-mini"]["n_calls"] == 2

    path = ledger.save(tmp_path / "agent_costs.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["n_calls"] == 3


def test_ledger_scrapes_usage_from_messages():
    real = AIMessage(
        content="x",
        response_metadata={"model_name": "gpt-4o-mini",
                           "token_usage": {"prompt_tokens": 300,
                                           "completion_tokens": 200}})
    scripted = AIMessage(content="y")          # no usage -> ignored
    ledger = TokenLedger()
    cost = ledger.from_messages([real, scripted], scenario="probe")
    assert cost > 0
    assert ledger.totals["n_calls"] == 1
    assert ledger.totals["prompt_tokens"] == 300


# --------------------------------------------------------------------------------------
# tracing (opt-in, env-driven)
# --------------------------------------------------------------------------------------
def test_tracing_disabled_without_key():
    assert configure_tracing(env={}) is False


def test_tracing_enabled_with_key():
    env = {"LANGCHAIN_API_KEY": "lsv2_test"}
    assert configure_tracing(env=env) is True
    assert env["LANGCHAIN_TRACING_V2"] == "true"
    assert env["LANGCHAIN_PROJECT"] == "quenchguard"


def test_tracing_never_mutates_real_env_in_tests(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    configure_tracing(env={"LANGCHAIN_API_KEY": "lsv2_test"})
    assert "LANGCHAIN_TRACING_V2" not in __import__("os").environ


# --------------------------------------------------------------------------------------
# transcript -> eval rows
# --------------------------------------------------------------------------------------
def test_transcript_flattening_pairs_q_with_tool_evidence():
    msgs = [
        HumanMessage(content="how is MRI-01?"),
        AIMessage(content="", tool_calls=[{"name": "score_magnet", "args": {},
                                           "id": "1", "type": "tool_call"}]),
        ToolMessage(content='{"quench_probability": 0.1}', tool_call_id="1",
                    name="score_magnet"),
        AIMessage(content="MRI-01 is at 0.1 probability."),
        HumanMessage(content="and MRI-02?"),
        AIMessage(content="No tools needed: MRI-02 was fine."),
    ]
    rows = transcript_to_eval_rows(msgs)
    assert len(rows) == 2
    assert rows[0]["question"] == "how is MRI-01?"
    assert "quench_probability" in rows[0]["contexts"]
    assert "0.1 probability" in rows[0]["answer"]
    assert rows[1]["contexts"] == ""          # no evidence on the second turn


def test_transcript_flattening_empty_input():
    assert transcript_to_eval_rows([]) == []


# --------------------------------------------------------------------------------------
# offline agent evaluation harness
# --------------------------------------------------------------------------------------
def test_eval_harness_all_scenarios_pass(tmp_path):
    from quenchguard.agent_eval import run_eval

    summary = run_eval(out_dir=tmp_path)
    assert summary["passed"], json.dumps(summary, indent=2)
    assert summary["n_scenarios"] == 4
    assert (tmp_path / "agent_eval.json").exists()


def test_eval_ticket_scenario_never_writes_without_approval(tmp_path):
    """The HITL invariant is asserted twice: structurally (interrupt) and by
    observing the filesystem."""
    from quenchguard.agent_eval import scenario_ticket_rejected, run_scenario

    result = run_scenario(scenario_ticket_rejected(tmp_path))
    assert result["checks"]["hitl_gate_fired"]
    assert result["checks"]["ticket_output_matches_decision"]
    assert not list(tmp_path.glob("tickets/*.md"))
