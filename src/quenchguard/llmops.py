"""LLMOps: token/cost ledger, LangSmith tracing config, agent transcripts.

Three orthogonal pieces, all offline-testable:

* :class:`TokenLedger` — token usage + dollar cost per model, scraped from the
  ``response_metadata`` of real ``AIMessage``s (or fed manually in tests).
  Persisted to ``reports/agent_costs.json`` so cost regressions are reviewable.
* :func:`configure_tracing` — opt-in LangSmith tracing via env vars
  (``LANGCHAIN_TRACING_V2`` / ``LANGCHAIN_API_KEY``); never required.
* :func:`transcript_to_eval_rows` — flattens a finished conversation into
  (question, answer, tool_context) rows for RAGAS-style faithfulness scoring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import REPORTS_DIR, REPO_ROOT

# USD per 1M tokens (update here when the model price changes)
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # model:            (input $/M, output $/M)
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (2.500, 10.000),
    "gpt-4.1-mini": (0.400, 1.600),
}
DEFAULT_MODEL = "gpt-4o-mini"


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Dollar cost of one call; unknown models price at 0 (flagged downstream)."""
    if model not in PRICE_TABLE:
        return 0.0
    pin, pout = PRICE_TABLE[model]
    return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000


class TokenLedger:
    """Accumulates token usage + cost across agent runs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, model: str, prompt_tokens: int, completion_tokens: int,
               scenario: str = "") -> float:
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        self.calls.append({
            "model": model, "scenario": scenario,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "cost_usd": round(cost, 6),
            "priced": model in PRICE_TABLE,
        })
        return cost

    def from_messages(self, messages: list[Any], scenario: str = "") -> float:
        """Scrape usage out of finished AI messages (real ChatOpenAI sets
        ``response_metadata['token_usage']``; scripted fakes simply carry none)."""
        total = 0.0
        for msg in messages:
            meta = getattr(msg, "response_metadata", None) or {}
            usage = meta.get("token_usage") or {}
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
            model = meta.get("model_name") or meta.get("model") or DEFAULT_MODEL
            if pt is not None and ct is not None:
                total += self.record(model, int(pt), int(ct), scenario)
        return total

    @property
    def totals(self) -> dict[str, Any]:
        calls = self.calls or []
        unpriced = sorted({c["model"] for c in calls if not c["priced"]})
        return {
            "n_calls": len(calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
            "by_model": {
                m: {"n_calls": sum(1 for c in calls if c["model"] == m),
                    "cost_usd": round(sum(c["cost_usd"] for c in calls
                                          if c["model"] == m), 6)}
                for m in sorted({c["model"] for c in calls})},
            "unpriced_models": unpriced,
        }

    def save(self, path: Path | None = None) -> Path:
        path = path or (REPORTS_DIR / "agent_costs.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.totals, indent=2), encoding="utf-8")
        return path


# --------------------------------------------------------------------------------------
# local environment (.env)
# --------------------------------------------------------------------------------------
def load_env(path: str | Path | None = None) -> None:
    """Load the repo's git-ignored ``.env`` without shadowing real env vars.

    Real environment variables always win (``override=False``), so CI secrets
    are never replaced by a stale local file. No-op when python-dotenv is
    absent or the file does not exist.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:                     # pragma: no cover - optional dep
        return
    load_dotenv(Path(path) if path else REPO_ROOT / ".env", override=False)


# --------------------------------------------------------------------------------------
# tracing (opt-in)
# --------------------------------------------------------------------------------------
def configure_tracing(project: str = "quenchguard",
                      env: dict[str, str] | None = None) -> bool:
    """Enable LangSmith tracing when credentials exist; harmless no-op otherwise.

    Required env (or already set): ``LANGCHAIN_API_KEY`` (also accepts
    ``LANGSMITH_API_KEY``). Sets ``LANGCHAIN_TRACING_V2=true`` and the project
    name. Returns whether tracing is enabled.
    """
    e = dict(os.environ if env is None else env)
    if not e.get("LANGCHAIN_API_KEY") and not e.get("LANGSMITH_API_KEY"):
        return False
    e.setdefault("LANGCHAIN_TRACING_V2", "true")
    e.setdefault("LANGCHAIN_PROJECT", project)
    if env is None:
        os.environ.update({k: v for k, v in e.items()
                           if k.startswith("LANGCHAIN")})
    else:
        env.update({k: v for k, v in e.items() if k.startswith("LANGCHAIN")})
    return True


# --------------------------------------------------------------------------------------
# transcript -> RAGAS-style eval rows
# --------------------------------------------------------------------------------------
def transcript_to_eval_rows(messages: list[Any]) -> list[dict[str, str]]:
    """Flatten one conversation into eval rows.

    Each user question is paired with the assistant's final answer and the
    tool evidence gathered in between (RAGAS "contexts"). Deterministic and
    offline — the RAGAS metrics themselves need an LLM and run separately.
    """
    rows: list[dict[str, str]] = []
    question, evidence, pending_answer = None, [], ""
    for msg in messages:
        kind = getattr(msg, "type", "")
        content = str(getattr(msg, "content", ""))
        if kind == "human":
            if question is not None and (evidence or pending_answer):
                rows.append(_row(question, pending_answer, evidence))
            question, evidence, pending_answer = content.strip(), [], ""
        elif kind == "tool":
            evidence.append(content)
        elif kind == "ai" and content and not (getattr(msg, "tool_calls", None)):
            pending_answer = content
    if question is not None and (evidence or pending_answer):
        rows.append(_row(question, pending_answer, evidence))
    return rows


def _row(question: str, answer: str, evidence: list[str]) -> dict[str, str]:
    return {
        "question": question,
        "answer": answer,
        "contexts": "\n---\n".join(evidence) if evidence else "",
    }


__all__ = ["TokenLedger", "estimate_cost", "configure_tracing", "load_env",
           "transcript_to_eval_rows", "PRICE_TABLE"]
