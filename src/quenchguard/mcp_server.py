"""QuenchGuard MCP server — the standard connection layer (Week 5, Layer 4).

Instead of every client writing a bespoke HTTP integration, any
MCP-compatible agent or desktop client discovers and calls these tools
through one common interface:

    python -m quenchguard.mcp_server                    # stdio (default)
    python -m quenchguard.mcp_server --transport streamable-http \
        --host 0.0.0.0 --port 8765                      # remote service

Tools (read-only by design):

* ``fleet_risk``     — champion-LSTM quench probability for every magnet
* ``score_magnet``   — one magnet: probability, threshold, SHAP top-3
* ``sensor_summary`` — raw telemetry statistics for one magnet
* ``doc_search``     — RAG retrieval over the model card / case study / README

Security note (least privilege): ``create_ticket`` is deliberately **not**
exposed here. Ticket creation stays behind the LangGraph human-approval
gate, so no external MCP client can write service tickets.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from .agent import doc_search as _doc_search
from .agent import fleet_risk as _fleet_risk
from .agent import score_magnet as _score_magnet
from .agent import sensor_summary as _sensor_summary

mcp = MCPServer(
    "quenchguard",
    title="QuenchGuard MRI maintenance tools",
    version="1.0.0",
    instructions=(
        "Read-only predictive-maintenance tools for an MRI magnet fleet. "
        "Use fleet_risk to triage the fleet, score_magnet for one magnet's "
        "quench probability and SHAP factors, sensor_summary for raw "
        "telemetry and doc_search for system documentation. Ticket creation "
        "requires a human approver and is not available over MCP."
    ),
)


@mcp.tool()
def fleet_risk() -> str:
    """Current quench risk for every MRI magnet (champion LSTM, latest window).

    Returns JSON: {"fleet": [{"magnet_id", "quench_probability", "alert"}],
    "n_scored": int}, sorted by probability descending.
    """
    return _fleet_risk()


@mcp.tool()
def score_magnet(magnet_id: str) -> str:
    """Score one magnet: quench probability, frozen threshold, SHAP top-3.

    Returns JSON with quench_probability, alert, threshold, horizon_hours,
    baseline_probability and explanation[{feature, shap_value, direction}].
    """
    return _score_magnet(magnet_id)


@mcp.tool()
def sensor_summary(magnet_id: str, hours: float = 24.0) -> str:
    """Recent raw-telemetry statistics (helium pressure, cold head, vibration)."""
    return _sensor_summary(magnet_id, hours)


@mcp.tool()
def doc_search(query: str, k: int = 3) -> str:
    """Retrieve documentation chunks (model card, case study, README) by query.

    Returns a JSON list of {source, heading, text, score}, best match first.
    """
    hits = _doc_search(query, k)
    return json.dumps(hits, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", default="stdio",
                        choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    return 0


__all__ = ["mcp", "fleet_risk", "score_magnet", "sensor_summary", "doc_search",
           "main"]


if __name__ == "__main__":
    raise SystemExit(main())
