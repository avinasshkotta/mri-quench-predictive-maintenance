"use client";

import { useState } from "react";
import Markdown, { type Components } from "react-markdown";
import { formatScore } from "@/lib/format";
import type { ChatMessage, Citation, HitlRequest } from "@/lib/types";
import HitlCard from "./HitlCard";
import ToolTrace from "./ToolTrace";

interface MessageBubbleProps {
  message: ChatMessage;
  busy: boolean;
  onResume: (
    messageId: string,
    hitl: HitlRequest,
    approved: boolean,
    reason: string | null,
  ) => void;
}

const markdownComponents: Components = {
  a: ({ node, ...props }) => {
    void node;
    return <a {...props} target="_blank" rel="noopener noreferrer" />;
  },
};

function Citations({ citations }: { citations: Citation[] }) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const openCitation = openIndex !== null ? citations[openIndex] : undefined;

  return (
    <div className="citations">
      <div className="trace-label">Sources</div>
      <div className="chip-row">
        {citations.map((citation, index) => {
          const isOpen = openIndex === index;
          return (
            <button
              key={`${citation.source}-${citation.heading ?? ""}-${index}`}
              type="button"
              className={`citation-chip ${isOpen ? "is-open" : ""}`}
              onClick={() => setOpenIndex(isOpen ? null : index)}
              aria-expanded={isOpen}
              title={`${citation.source}${citation.heading ? ` — ${citation.heading}` : ""}`}
            >
              <span aria-hidden>📄</span>
              <span className="citation-source">{citation.source}</span>
              {citation.heading && <span className="citation-heading">· {citation.heading}</span>}
              {typeof citation.score === "number" && (
                <span className="citation-score">{formatScore(citation.score)}</span>
              )}
            </button>
          );
        })}
      </div>

      {openCitation && (
        <div className="citation-detail">
          <div className="citation-detail-head">
            <span className="mono">{openCitation.source}</span>
            {openCitation.heading && <strong>{openCitation.heading}</strong>}
            {typeof openCitation.score === "number" && (
              <span className="chip-tiny">relevance {formatScore(openCitation.score)}</span>
            )}
          </div>
          <p>
            Retrieved from the QuenchGuard documentation index
            {openCitation.heading ? ` — section “${openCitation.heading}”` : ""}.
          </p>
        </div>
      )}
    </div>
  );
}

export default function MessageBubble({ message, busy, onResume }: MessageBubbleProps) {
  if (message.role === "user") {
    return (
      <div className="message-row is-user">
        <div className="user-bubble">{message.content}</div>
      </div>
    );
  }

  const streaming = Boolean(message.streaming);
  const hasBody =
    Boolean(message.content) ||
    streaming ||
    message.tools.length > 0 ||
    message.citations.length > 0 ||
    message.hitl.length > 0 ||
    Boolean(message.error);

  return (
    <article className={`message-row is-assistant ${streaming ? "is-streaming" : ""}`} aria-busy={streaming}>
      <div className="assistant-head">
        <span className={`agent-avatar ${streaming ? "is-live" : ""}`} aria-hidden>
          QG
        </span>
        <span className="agent-name">QuenchGuard agent</span>
        {streaming && (
          <span className="stream-state">
            <span className="pulse-dot" aria-hidden />
            working
          </span>
        )}
      </div>

      <div className="assistant-body">
        {message.content ? (
          <div className={`md ${streaming ? "md-streaming" : ""}`}>
            <Markdown components={markdownComponents}>{message.content}</Markdown>
          </div>
        ) : streaming ? (
          <div className="thinking" role="status" aria-label="Assistant is thinking">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        ) : null}

        {message.error && (
          <p className="msg-error" role="alert">
            <span aria-hidden>⚠</span> {message.error}
          </p>
        )}

        {message.stopped && !streaming && <p className="msg-stopped">■ Generation stopped</p>}

        {message.tools.length > 0 && <ToolTrace tools={message.tools} />}

        {message.citations.length > 0 && <Citations citations={message.citations} />}

        {message.hitl.map((hitl) => (
          <HitlCard
            key={hitl.call_id}
            hitl={hitl}
            busy={busy}
            onApprove={(request) => onResume(message.id, request, true, null)}
            onReject={(request, reason) => onResume(message.id, request, false, reason)}
          />
        ))}

        {!hasBody && <p className="muted-sm">No response content.</p>}
      </div>
    </article>
  );
}
