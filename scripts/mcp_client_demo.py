#!/usr/bin/env python
"""MCP client demo: discover and call QuenchGuard tools over stdio.

    python scripts/mcp_client_demo.py
    python scripts/mcp_client_demo.py --tool score_magnet --magnet MRI-05

This is the client side of the standard connection layer: no QuenchGuard
imports, no custom HTTP calls — one MCP session and the tools are there.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


async def run(tool: str, magnet: str) -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {**os.environ,
           "PYTHONPATH": f"{REPO_ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "quenchguard.mcp_server"],
        env=env)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("discovered tools:", ", ".join(t.name for t in tools.tools))
            args = {"magnet_id": magnet} if tool != "doc_search" else {"query": magnet}
            result = await session.call_tool(tool, args)
            for block in result.content:
                text = getattr(block, "text", "")
                try:
                    parsed = json.loads(text)
                except ValueError:
                    parsed = text
                print(f"\n{tool} ->")
                print(json.dumps(parsed, indent=2)[:1200]
                      if not isinstance(parsed, str) else parsed[:1200])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", default="fleet_risk",
                    choices=["fleet_risk", "score_magnet", "sensor_summary",
                             "doc_search"])
    ap.add_argument("--magnet", default="MRI-05",
                    help="magnet id (or the query for doc_search)")
    args = ap.parse_args()
    return asyncio.run(run(args.tool, args.magnet))


if __name__ == "__main__":
    sys.exit(main())
