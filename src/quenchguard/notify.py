"""Email alerts for approved service tickets (stdlib SMTP, fail-safe).

When the LangGraph human-approval gate approves a ``create_ticket`` call, the
ticket file is written and an alert email is sent to the maintenance inbox.
Configuration is entirely environment-driven; without it the notifier is a
no-op, so local runs and tests never touch a mail server:

* ``SMTP_HOST``          required to enable email (e.g. smtp.gmail.com)
* ``SMTP_PORT``          default 587
* ``SMTP_USER`` / ``SMTP_PASSWORD``  optional login (start-TLS by default)
* ``SMTP_USE_TLS``       ``0`` disables STARTTLS (default on)
* ``SMTP_FROM``          default = ``SMTP_USER``
* ``TICKET_ALERT_TO``    required to enable email; comma-separated recipients

Delivery failures are warnings, never exceptions: a ticket that was written
must never be lost because the mail server is down (EU AI Act Art. 12 keeps
the file as the audit record).
"""

from __future__ import annotations

import os
import smtplib
import warnings
from email.message import EmailMessage
from typing import Any, Iterable


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def recipients() -> list[str]:
    raw = _env("TICKET_ALERT_TO")
    return [addr.strip() for addr in raw.split(",") if addr.strip()]


def email_configured() -> bool:
    """True when SMTP is usable.

    Requires a host and at least one recipient, and — when an SMTP user is
    configured — a password (Gmail/SES reject unauthenticated relays, so
    reporting "configured" without credentials would be misleading).
    """
    if not _env("SMTP_HOST") or not recipients():
        return False
    if _env("SMTP_USER") and not _env("SMTP_PASSWORD"):
        return False
    return True


def ticket_email(magnet_id: str, priority: str, summary: str,
                 recommended_action: str, path: str | None = None,
                 created: str | None = None) -> tuple[str, str]:
    """Compose (subject, plain-text body) for one approved ticket."""
    subject = f"[QuenchGuard] {str(priority).upper()} ticket — {magnet_id}"
    lines = [
        "A service ticket was approved by a human operator.",
        "",
        f"Magnet      : {magnet_id}",
        f"Priority    : {priority}",
        f"Created     : {created or 'n/a'}",
        f"Ticket file : {path or 'n/a'}",
        "",
        "Summary",
        "-------",
        summary.strip() or "(none)",
        "",
        "Recommended action",
        "------------------",
        recommended_action.strip() or "(none)",
        "",
        "This message was sent by QuenchGuard after the human-in-the-loop",
        "approval gate (EU AI Act Art. 14). The ticket file is the audit record.",
    ]
    return subject, "\n".join(lines)


def send_ticket_email(subject: str, body: str,
                      to: Iterable[str] | None = None) -> bool:
    """Send one plain-text email. Returns False (with a warning) on failure."""
    if not email_configured():
        return False
    host = _env("SMTP_HOST")
    port = int(_env("SMTP_PORT") or "587")
    user = _env("SMTP_USER")
    password = _env("SMTP_PASSWORD")
    sender = _env("SMTP_FROM") or user or "quenchguard@localhost"
    targets = list(to or recipients())
    if not targets:
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(targets)
    message.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            use_tls = _env("SMTP_USE_TLS").lower() not in {"0", "false", "off", "no"}
            if use_tls:
                smtp.starttls()
            if user and password:
                smtp.login(user, password)
            smtp.send_message(message, from_addr=sender, to_addrs=targets)
        return True
    except Exception as exc:                       # noqa: BLE001 — never raise
        warnings.warn(f"ticket email delivery failed ({type(exc).__name__}: {exc})",
                      RuntimeWarning, stacklevel=2)
        return False


def notify_ticket(magnet_id: str, priority: str, summary: str,
                  recommended_action: str, path: str | None = None,
                  created: str | None = None) -> bool:
    """Send the alert for an approved ticket; no-op when email is unconfigured."""
    if not email_configured():
        return False
    subject, body = ticket_email(magnet_id, priority, summary,
                                 recommended_action, path, created)
    return send_ticket_email(subject, body)


def status() -> dict[str, Any]:
    """Non-secret email configuration summary for /api/status."""
    return {
        "ticket_email_configured": email_configured(),
        "ticket_email_recipients": len(recipients()),
    }


__all__ = ["email_configured", "ticket_email", "send_ticket_email",
           "notify_ticket", "recipients", "status"]
