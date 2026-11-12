#!/usr/bin/env python
"""Agent evaluation + cost report.

    python scripts/eval_agent.py             # offline structural eval (no API key)
    python scripts/eval_agent.py --ragas     # + LLM-judged faithfulness (needs OPENAI_API_KEY)

Outputs
-------
* ``reports/agent_eval.json``  — per-scenario structural checks (CI-assertable)
* ``reports/agent_costs.json`` — token/cost ledger summary (populated on live runs)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quenchguard.agent_eval import run_eval              # noqa: E402
from quenchguard.config import REPORTS_DIR               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ragas", action="store_true",
                    help="additionally run RAGAS faithfulness/relevancy "
                         "(requires OPENAI_API_KEY)")
    ap.add_argument("--min-pass", type=int, default=None,
                    help="fail (exit 1) when fewer than N scenarios pass")
    args = ap.parse_args()

    summary = run_eval(out_dir=REPORTS_DIR)
    print(f"agent eval: {summary['n_passed']}/{summary['n_scenarios']} scenarios passed "
          f"({'PASS' if summary['passed'] else 'FAIL'})")
    for s in summary["scenarios"]:
        failed = [k for k, ok in s["checks"].items() if not ok]
        status = "ok " if s["passed"] else "FAIL"
        print(f"  [{status}] {s['id']}"
              + (f" -> {failed}" if failed else ""))

    if args.ragas:
        code = _run_ragas()
        if code == 3:
            print("ragas skipped: OPENAI_API_KEY not set")
        elif code != 0:
            print("ragas run failed", file=sys.stderr)

    if args.min_pass is not None and summary["n_passed"] < args.min_pass:
        print(f"below --min-pass {args.min_pass}", file=sys.stderr)
        return 1
    return 0 if summary["passed"] else 1


def _run_ragas() -> int:
    """Optional LLM-judged metrics; returns 3 when skipped, 1 on failure."""
    import os
    import types

    if not os.environ.get("OPENAI_API_KEY"):
        return 3
    try:
        # ragas 0.4.3 still imports langchain_community.chat_models.vertexai,
        # which was removed in langchain-community 0.4 — shim it (ragas only
        # references the class for typing; we never use VertexAI).
        if "langchain_community.chat_models.vertexai" not in sys.modules:
            mod = types.ModuleType("langchain_community.chat_models.vertexai")
            mod.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules["langchain_community.chat_models.vertexai"] = mod
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from langchain_openai import ChatOpenAI
            from ragas import evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as exc:
        print(f"ragas unavailable ({exc}) — pip install ragas", file=sys.stderr)
        return 3

    from quenchguard.llmops import TokenLedger

    # rebuild a real-LLM transcript for scoring (investigation scenario)
    from quenchguard.agent import build_graph

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    graph = build_graph(llm)
    result = graph.invoke(
        {"messages": [("user",
                       "Which magnet is at the highest quench risk right now and "
                       "what does the model say?")]},
        {"configurable": {"thread_id": "ragas"}})

    ledger = TokenLedger()
    ledger.from_messages(result["messages"], scenario="ragas_probe")
    costs = ledger.save(REPORTS_DIR / "agent_costs.json")
    print(f"token ledger: {ledger.totals['n_calls']} calls, "
          f"${ledger.totals['cost_usd']:.4f} -> {costs.name}")

    from quenchguard.llmops import transcript_to_eval_rows

    rows = transcript_to_eval_rows(result["messages"])
    if not rows or not rows[0]["contexts"]:
        print("no tool evidence to score — skipping ragas")
        return 3
    from datasets import Dataset

    ds = Dataset.from_list([{**r, "contexts": [r["contexts"]], "ground_truth": ""}
                            for r in rows])
    judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy], llm=judge)
    (REPORTS_DIR / "ragas_scores.json").write_text(
        json.dumps({k: float(v) for k, v in scores.items()}, indent=2),
        encoding="utf-8")
    print("ragas:", {k: round(float(v), 3) for k, v in scores.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
