"use client";

import { useState } from "react";
import { argPreview, truncate } from "@/lib/format";
import type { ToolCall } from "@/lib/types";

const RESULT_PREVIEW_CHARS = 300;

function ToolRow({ tool }: { tool: ToolCall }) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const content = tool.content ?? "";
  const preview = truncate(content, RESULT_PREVIEW_CHARS);
  const args = argPreview(tool.args);
  const state = tool.done
    ? "is-done"
    : tool.interrupted
      ? "is-stopped"
      : tool.waiting
        ? "is-waiting"
        : "is-pending";

  return (
    <div className={`tool-row ${state}`}>
      <div className="tool-line">
        <span className="tool-status" aria-hidden>
          {tool.done ? "✓" : tool.interrupted ? "■" : tool.waiting ? "⏸" : <span className="mini-spinner" />}
        </span>
        <span className="tool-name">{tool.name}</span>
        {args && (
          <span className="tool-args" title={argPreview(tool.args, 2000)}>
            ({args})
          </span>
        )}
        {tool.interrupted && <span className="chip-tiny">stopped</span>}
        {tool.waiting && <span className="chip-tiny">awaiting approval</span>}
        {!tool.interrupted && !tool.waiting && (
          <span className="sr-only">{tool.done ? "completed" : "running"}</span>
        )}
      </div>

      {tool.done && content && (
        <div className="tool-result">
          <div className="tool-result-actions">
            <button
              type="button"
              className="link-btn"
              onClick={() => setOpen((value) => !value)}
              aria-expanded={open}
            >
              {open ? "Hide result" : "Show result"}
            </button>
            {tool.truncated && <span className="chip-tiny">truncated</span>}
          </div>
          {open && (
            <>
              <pre className="tool-pre">{expanded ? content : preview}</pre>
              {content.length > RESULT_PREVIEW_CHARS && (
                <button
                  type="button"
                  className="link-btn"
                  onClick={() => setExpanded((value) => !value)}
                >
                  {expanded
                    ? "Show less"
                    : `Show all ${content.length.toLocaleString()} characters`}
                </button>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function ToolTrace({ tools }: { tools: ToolCall[] }) {
  return (
    <div className="tool-trace">
      <div className="trace-label">Tool activity</div>
      {tools.map((tool) => (
        <ToolRow key={tool.id} tool={tool} />
      ))}
    </div>
  );
}
