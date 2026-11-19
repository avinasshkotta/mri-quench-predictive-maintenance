"use client";

import { useState } from "react";
import { priorityTone } from "@/lib/format";
import type { HitlRequest } from "@/lib/types";

interface HitlCardProps {
  hitl: HitlRequest;
  busy: boolean;
  onApprove: (hitl: HitlRequest) => void;
  onReject: (hitl: HitlRequest, reason: string | null) => void;
}

export default function HitlCard({ hitl, busy, onApprove, onReject }: HitlCardProps) {
  const [reason, setReason] = useState("");
  const approved = hitl.status === "approved";
  const rejected = hitl.status === "rejected";
  const decided = approved || rejected;
  const tone = priorityTone(hitl.args.priority);
  const priority = hitl.args.priority ?? "—";
  const magnet = hitl.args.magnet_id ?? "—";
  const summary = hitl.args.summary ?? "—";
  const action = hitl.args.recommended_action ?? "—";

  return (
    <section
      className={`hitl-card status-${hitl.status}`}
      aria-label={decided ? "Ticket decision" : "Human approval required"}
    >
      <div className="hitl-head">
        <span className="hitl-icon" aria-hidden>
          {approved ? "✅" : rejected ? "🚫" : "⏸"}
        </span>
        <div className="hitl-head-text">
          <h3>
            {approved
              ? "Ticket approved"
              : rejected
                ? "Ticket rejected"
                : "Human approval required"}
          </h3>
          {!decided && (
            <p className="hitl-question">
              {hitl.question ?? "A human operator must approve this service ticket."}
            </p>
          )}
        </div>
      </div>

      <div className="hitl-fields">
        <div className="hitl-field">
          <span className="kicker">Magnet</span>
          <span className="mono hitl-magnet">{magnet}</span>
        </div>
        <div className="hitl-field">
          <span className="kicker">Priority</span>
          <span className={`priority-badge tone-${tone}`}>{priority}</span>
        </div>
        <div className="hitl-field wide">
          <span className="kicker">Summary</span>
          <p>{summary}</p>
        </div>
        <div className="hitl-field wide">
          <span className="kicker">Recommended action</span>
          <p>{action}</p>
        </div>
      </div>

      {!decided && (
        <>
          <label className="hitl-reason">
            <span className="kicker">Rejection reason (optional)</span>
            <input
              type="text"
              value={reason}
              onChange={(event) => setReason(event.target.value)}
              placeholder="Why is this ticket being rejected?"
              disabled={busy}
              maxLength={500}
            />
          </label>
          <div className="hitl-actions">
            <button
              type="button"
              className="btn btn-approve"
              onClick={() => onApprove(hitl)}
              disabled={busy}
            >
              ✅ Approve and create ticket
            </button>
            <button
              type="button"
              className="btn btn-reject"
              onClick={() => onReject(hitl, reason.trim() ? reason.trim() : null)}
              disabled={busy}
            >
              🚫 Reject
            </button>
          </div>
        </>
      )}

      {decided && (
        <div className={`hitl-status ${hitl.status}`}>
          <span aria-hidden>{approved ? "✅" : "🚫"}</span>
          <span>
            {approved ? "Approved by human operator" : "Rejected by human operator"}
            {rejected && hitl.reason ? ` — ${hitl.reason}` : ""}
          </span>
          {busy && <span className="mini-spinner" aria-hidden />}
          {busy && <span className="sr-only">Resuming the assistant</span>}
        </div>
      )}

      {!decided && busy && <p className="hitl-working">Resuming agent…</p>}
    </section>
  );
}
