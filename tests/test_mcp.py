"""Tests for the MCP server (Week 5, Layer 4: the standard connection layer).

The MCP SDK is optional at test time (``importorskip``). The security
invariant is checked explicitly: the server must expose read-only tools only,
never ``create_ticket`` (ticket writes stay behind the HITL gate).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("mcp", reason="mcp SDK not installed")

from quenchguard import mcp_server as M       # noqa: E402

READ_ONLY = {"fleet_risk", "score_magnet", "sensor_summary", "doc_search"}


def test_server_exposes_read_only_tools_only():
    tools = asyncio.run(M.mcp.list_tools())
    names = {tool.name for tool in tools}
    assert READ_ONLY <= names
    assert "create_ticket" not in names, "ticket writes must stay behind HITL"


def test_tool_descriptions_are_present():
    tools = {tool.name: tool for tool in asyncio.run(M.mcp.list_tools())}
    for name in READ_ONLY:
        assert tools[name].description, f"{name} needs a description for the LLM"


def test_doc_search_tool_returns_ranked_citations():
    hits = json.loads(M.doc_search("false page budget threshold policy", k=2))
    assert hits and hits[0]["source"].endswith(".md")
    assert hits[0]["score"] >= hits[-1]["score"]


def test_fleet_risk_tool_against_real_artifacts():
    if not (REPO / "models" / "lstm_detector.pt").exists():
        pytest.skip("trained artifacts not built")
    has_data = ((REPO / "data" / "gold" / "features.parquet").exists()
                or (REPO / "data" / "demo" / "features.parquet").exists())
    if not has_data:
        pytest.skip("data bundle not built")
    body = json.loads(M.fleet_risk())
    assert body["fleet"] and body["n_scored"] == len(body["fleet"])
    assert {"magnet_id", "quench_probability", "alert"} <= set(body["fleet"][0])
