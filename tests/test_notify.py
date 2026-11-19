"""Ticket email alerts: SMTP notifier is opt-in, fail-safe and test-covered."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quenchguard import notify as N        # noqa: E402
from quenchguard.agent import create_ticket  # noqa: E402

ENV_KEYS = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
            "SMTP_FROM", "SMTP_USE_TLS", "TICKET_ALERT_TO")


@pytest.fixture(autouse=True)
def _clean_email_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class FakeSMTP:
    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.started_tls = False
        self.logged_in = None
        self.sent = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, user, password):
        self.logged_in = (user, password)

    def send_message(self, msg, from_addr=None, to_addrs=None):
        self.sent.append((msg, from_addr, to_addrs))


def test_unconfigured_is_a_noop(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("SMTP must not be touched when unconfigured")

    monkeypatch.setattr(N.smtplib, "SMTP", boom)
    assert N.email_configured() is False
    assert N.notify_ticket("MRI-01", "low", "s", "a") is False


def test_configured_sends_alert_email(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("SMTP_USER", "user")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setenv("TICKET_ALERT_TO", "a@example.com, b@example.com")
    monkeypatch.setattr(N.smtplib, "SMTP", FakeSMTP)
    FakeSMTP.instances.clear()

    ok = N.notify_ticket("MRI-02", "high", "cold head trending up",
                         "inspect bearings", "/tmp/ticket.md",
                         "2026-01-01T00:00:00Z")
    assert ok is True

    smtp = FakeSMTP.instances[-1]
    assert (smtp.host, smtp.port) == ("smtp.example.com", 2525)
    assert smtp.started_tls is True
    assert smtp.logged_in == ("user", "secret")

    message, from_addr, to_addrs = smtp.sent[-1]
    assert "MRI-02" in message["Subject"]
    assert "HIGH" in message["Subject"]
    body = message.get_content()
    assert "inspect bearings" in body
    assert "Art. 14" in body
    assert to_addrs == ["a@example.com", "b@example.com"]


def test_smtp_failure_warns_and_returns_false(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("TICKET_ALERT_TO", "a@example.com")

    class BrokenSMTP(FakeSMTP):
        def send_message(self, *_a, **_k):
            raise OSError("connection refused")

    monkeypatch.setattr(N.smtplib, "SMTP", BrokenSMTP)
    with pytest.warns(RuntimeWarning, match="email delivery failed"):
        assert N.notify_ticket("MRI-01", "low", "s", "a") is False


def test_create_ticket_emails_after_write(tmp_path, monkeypatch):
    from quenchguard import agent as A

    monkeypatch.setattr(A, "REPORTS_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(N, "notify_ticket",
                        lambda *a, **k: calls.append((a, k)) or True)

    out = json.loads(create_ticket("MRI-03", "medium", "summary", "act"))
    assert out["created"] is True
    assert Path(out["path"]).exists()
    assert calls and calls[0][0][0] == "MRI-03"
    assert calls[0][0][1] == "medium"


def test_create_ticket_survives_notifier_crash(tmp_path, monkeypatch):
    from quenchguard import agent as A

    monkeypatch.setattr(A, "REPORTS_DIR", tmp_path)

    def boom(*_a, **_k):
        raise RuntimeError("smtp exploded")

    monkeypatch.setattr(N, "notify_ticket", boom)
    out = json.loads(create_ticket("MRI-04", "low", "s", "a"))
    assert out["created"] is True
    assert Path(out["path"]).exists()


def test_status_reports_email_configuration(monkeypatch):
    assert N.status()["ticket_email_configured"] is False
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("TICKET_ALERT_TO", "a@example.com")
    state = N.status()
    assert state["ticket_email_configured"] is True
    assert state["ticket_email_recipients"] == 1
