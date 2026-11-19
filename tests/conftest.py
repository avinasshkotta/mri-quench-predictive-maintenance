"""Shared test configuration.

The prediction audit log (EU AI Act Art. 12) is disabled by default so tests
never append to the repo's ``reports/prediction_log.jsonl``. Tests that
exercise the log set ``PREDICTION_LOG_PATH`` / ``PREDICTION_LOG`` explicitly.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_prediction_log(monkeypatch):
    monkeypatch.setenv("PREDICTION_LOG", "off")
