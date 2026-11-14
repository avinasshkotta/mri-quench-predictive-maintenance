"""Smoke tests for the Streamlit demo app (scripts/demo_app.py).

Uses Streamlit's AppTest harness to render the app headlessly. Requires
trained artifacts (conditional, like the serving tests); the agent tab is
exercised only for its no-key warning path — the chat itself is UI-driven.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("streamlit", reason="streamlit not installed")

from streamlit.testing.v1 import AppTest          # noqa: E402

APP = REPO / "scripts" / "demo_app.py"


def test_app_renders_fleet_and_governance_without_exception():
    if not (REPO / "models" / "lstm_detector.pt").exists():
        pytest.skip("trained artifacts absent (CI) — run_pipeline.py first")
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4
    # fleet metrics rendered (six magnets)
    assert at.metric, "no st.metric elements rendered"


def test_app_warns_when_openai_key_is_missing(monkeypatch):
    if not (REPO / "models" / "lstm_detector.pt").exists():
        pytest.skip("trained artifacts absent (CI)")
    # an empty value beats .env loading (override=False) and must trigger
    # the friendly warning instead of building the graph
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    warnings_text = "\n".join(w.value for w in at.warning)
    assert "OPENAI_API_KEY" in warnings_text
