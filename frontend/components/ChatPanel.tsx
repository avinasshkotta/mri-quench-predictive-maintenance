"use client";

import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { isAbortError, streamSSE } from "@/lib/api";
import { createId, errorMessage } from "@/lib/format";
import {
  isRecord,
  readBoolean,
  readNumber,
  readString,
  type ChatMessage,
  type Citation,
  type HitlRequest,
  type HitlStatus,
  type ToolCall,
} from "@/lib/types";
import MessageBubble from "./MessageBubble";

const SUGGESTIONS = [
  "Which magnet is at highest risk right now?",
  "Explain the risk factors for MRI-05",
  "What does the model card say about limitations?",
  "Draft a service ticket for MRI-01",
];

interface ChatPanelProps {
  llmConfigured: boolean | null;
  apiOnline: boolean | null;
  model: string | null;
  onMagnetHint: (magnetId: string) => void;
}

function finishTool(
  tools: ToolCall[],
  name: string,
  content: string,
  truncated: boolean,
  fallback: ToolCall,
): ToolCall[] {
  for (let index = tools.length - 1; index >= 0; index -= 1) {
    const tool = tools[index];
    if (tool && tool.name === name && !tool.done) {
      const next = tools.slice();
      next[index] = {
        ...tool,
        content,
        truncated,
        done: true,
        interrupted: false,
        waiting: false,
      };
      return next;
    }
  }
  return [...tools, fallback];
}

function isDuplicateCitation(citations: Citation[], next: Citation): boolean {
  return citations.some(
    (citation) => citation.source === next.source && citation.heading === next.heading,
  );
}

function contentToString(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export default function ChatPanel({
  llmConfigured,
  apiOnline,
  model,
  onMagnetHint,
}: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [banner, setBanner] = useState<string | null>(null);

  const threadIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamingRef = useRef(false);
  const listRef = useRef<HTMLDivElement | null>(null);
  const pinnedRef = useRef(true);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const canChat = llmConfigured !== false && apiOnline !== false;

  const updateMessage = useCallback(
    (id: string, updater: (message: ChatMessage) => ChatMessage) => {
      setMessages((prev) => prev.map((message) => (message.id === id ? updater(message) : message)));
    },
    [],
  );

  const handleEvent = useCallback(
    (messageId: string, type: string, data: unknown) => {
      const payload = isRecord(data) ? data : {};

      switch (type) {
        case "meta": {
          const threadId = readString(payload.thread_id);
          if (threadId) threadIdRef.current = threadId;
          break;
        }

        case "token": {
          const text = readString(payload.text);
          if (!text) break;
          updateMessage(messageId, (message) => ({
            ...message,
            content: message.content + text,
          }));
          break;
        }

        case "tool_start": {
          const name = readString(payload.name);
          if (!name) break;
          const tool: ToolCall = {
            id: createId(),
            name,
            args: payload.args,
            done: false,
          };
          updateMessage(messageId, (message) => ({
            ...message,
            tools: [...message.tools, tool],
          }));

          const args = isRecord(payload.args) ? payload.args : null;
          const magnetId = args ? readString(args.magnet_id) : undefined;
          if (magnetId && (name === "score_magnet" || name === "sensor_summary")) {
            onMagnetHint(magnetId);
          }
          break;
        }

        case "tool_result": {
          const name = readString(payload.name);
          if (!name) break;
          const content = contentToString(payload.content);
          const truncated = readBoolean(payload.truncated) === true;
          const fallback: ToolCall = {
            id: createId(),
            name,
            args: undefined,
            content,
            truncated,
            done: true,
          };
          updateMessage(messageId, (message) => ({
            ...message,
            tools: finishTool(message.tools, name, content, truncated, fallback),
          }));
          break;
        }

        case "citation": {
          const source = readString(payload.source);
          if (!source) break;
          const citation: Citation = {
            source,
            heading: readString(payload.heading),
            score: readNumber(payload.score),
          };
          updateMessage(messageId, (message) =>
            isDuplicateCitation(message.citations, citation)
              ? message
              : { ...message, citations: [...message.citations, citation] },
          );
          break;
        }

        case "interrupt": {
          const callId = readString(payload.call_id) ?? createId();
          const rawArgs = isRecord(payload.args) ? payload.args : {};
          const hitl: HitlRequest = {
            type: readString(payload.type) ?? "ticket_approval",
            call_id: callId,
            args: {
              magnet_id: readString(rawArgs.magnet_id),
              priority: readString(rawArgs.priority),
              summary: readString(rawArgs.summary),
              recommended_action: readString(rawArgs.recommended_action),
            },
            question: readString(payload.question),
            status: "pending",
            reason: null,
          };
          updateMessage(messageId, (message) =>
            message.hitl.some((item) => item.call_id === callId)
              ? message
              : {
                  ...message,
                  hitl: [...message.hitl, hitl],
                  tools: message.tools.map((tool) =>
                    !tool.done && tool.name === "create_ticket"
                      ? { ...tool, waiting: true }
                      : tool,
                  ),
                },
          );
          break;
        }

        case "error": {
          const detail = readString(payload.message) ?? "The assistant reported an error.";
          updateMessage(messageId, (message) => ({ ...message, error: detail }));
          break;
        }

        case "halted":
        case "done":
          break;

        default:
          break;
      }
    },
    [onMagnetHint, updateMessage],
  );

  const startStream = useCallback(
    async (path: string, body: unknown, messageId: string, isResume: boolean) => {
      if (streamingRef.current) return;
      streamingRef.current = true;
      setStreaming(true);
      const controller = new AbortController();
      abortRef.current = controller;
      updateMessage(messageId, (message) => ({
        ...message,
        streaming: true,
        stopped: false,
        error: null,
        tools: message.tools.map((tool) =>
          tool.waiting ? { ...tool, waiting: false } : tool,
        ),
      }));

      try {
        await streamSSE(
          path,
          body,
          (type, data) => handleEvent(messageId, type, data),
          controller.signal,
        );
      } catch (error) {
        if (isAbortError(error)) {
          updateMessage(messageId, (message) => ({
            ...message,
            stopped: true,
            streaming: false,
            tools: message.tools.map((tool) =>
              tool.done ? tool : { ...tool, interrupted: true, waiting: false },
            ),
          }));
        } else {
          const detail = errorMessage(error);
          updateMessage(messageId, (message) => ({
            ...message,
            error: detail,
            streaming: false,
            tools: message.tools.map((tool) =>
              tool.done ? tool : { ...tool, interrupted: true, waiting: false },
            ),
          }));
          setBanner(detail);
          if (!isResume) {
            // A pre-stream failure leaves an empty assistant bubble — drop it.
            setMessages((prev) =>
              prev.filter((message) => {
                const empty =
                  !message.content &&
                  message.tools.length === 0 &&
                  message.citations.length === 0 &&
                  message.hitl.length === 0;
                return !(message.id === messageId && empty);
              }),
            );
          }
        }
      } finally {
        streamingRef.current = false;
        abortRef.current = null;
        setStreaming(false);
        updateMessage(messageId, (message) => ({ ...message, streaming: false }));
      }
    },
    [handleEvent, updateMessage],
  );

  const send = useCallback(
    (raw: string) => {
      const text = raw.trim();
      if (!text || streamingRef.current || !canChat) return;

      const threadId = threadIdRef.current ?? createId();
      threadIdRef.current = threadId;

      const userMessage: ChatMessage = {
        id: createId(),
        role: "user",
        content: text,
        tools: [],
        citations: [],
        hitl: [],
      };
      const assistantMessage: ChatMessage = {
        id: createId(),
        role: "assistant",
        content: "",
        streaming: true,
        tools: [],
        citations: [],
        hitl: [],
        error: null,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setInput("");
      setBanner(null);
      pinnedRef.current = true;
      void startStream(
        "/api/chat",
        { message: text, thread_id: threadId },
        assistantMessage.id,
        false,
      );
    },
    [canChat, startStream],
  );

  const handleResume = useCallback(
    (messageId: string, hitl: HitlRequest, approved: boolean, reason: string | null) => {
      if (streamingRef.current) return;
      const threadId = threadIdRef.current;
      if (!threadId) {
        setBanner("This conversation lost its thread id. Start a new message to continue.");
        return;
      }
      const decisionReason = approved ? null : reason;
      const status: HitlStatus = approved ? "approved" : "rejected";
      setBanner(null);
      updateMessage(messageId, (message) => ({
        ...message,
        error: null,
        hitl: message.hitl.map((item) =>
          item.call_id === hitl.call_id
            ? { ...item, status, reason: decisionReason }
            : item,
        ),
      }));
      void startStream(
        "/api/chat/resume",
        { thread_id: threadId, approved, reason: decisionReason },
        messageId,
        true,
      );
    },
    [startStream, updateMessage],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const resetChat = useCallback(() => {
    if (streamingRef.current) return;
    threadIdRef.current = null;
    setMessages([]);
    setInput("");
    setBanner(null);
    pinnedRef.current = true;
  }, []);

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      send(input);
    }
  };

  const handleScroll = () => {
    const el = listRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  useEffect(() => {
    const el = listRef.current;
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const placeholder = !canChat
    ? llmConfigured === false
      ? "Assistant unavailable — no LLM key configured"
      : "Backend offline — reconnect to chat"
    : "Ask about magnet health, quench risk or service tickets…";

  return (
    <section className="chat-card" aria-label="Maintenance chat">
      <header className="chat-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <svg
              viewBox="0 0 24 24"
              width="21"
              height="21"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M2.5 12h4l2.1-5.4 3 10.8 2.1-5.4h7.8" />
            </svg>
          </span>
          <div className="brand-text">
            <h1>QuenchGuard</h1>
            <p className="subtitle">MRI quench early-warning assistant</p>
          </div>
        </div>
        <div className="chat-header-actions">
          {model && (
            <span className="model-chip" title="Model served by the API">
              {model}
            </span>
          )}
          {messages.length > 0 && (
            <button
              type="button"
              className="ghost-btn"
              onClick={resetChat}
              disabled={streaming}
            >
              New chat
            </button>
          )}
        </div>
      </header>

      {llmConfigured === false && (
        <div className="notice notice-warn" role="status">
          <span className="notice-icon" aria-hidden>
            ⚠
          </span>
          <span>
            <strong>Assistant unavailable.</strong> The server has no LLM API key configured, so
            chat is disabled. Live fleet risk and telemetry still work.
          </span>
        </div>
      )}

      {banner && (
        <div className="notice notice-error" role="alert">
          <span className="notice-icon" aria-hidden>
            ⚠
          </span>
          <span>{banner}</span>
          <button
            type="button"
            className="notice-close"
            onClick={() => setBanner(null)}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}

      <div
        className={`messages ${messages.length === 0 ? "is-empty" : ""}`}
        ref={listRef}
        onScroll={handleScroll}
        role="log"
        aria-label="Conversation"
      >
        {messages.length === 0 ? (
          <div className="empty-state">
            <span className="kicker">Maintenance copilot</span>
            <h2>Ask the fleet.</h2>
            <p>
              QuenchGuard reasons over live quench-risk scores, raw sensor telemetry and the model
              documentation. Tool calls and RAG citations appear inline, and every service ticket
              waits for human approval.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((question) => (
                <button
                  key={question}
                  type="button"
                  className="suggestion"
                  onClick={() => send(question)}
                  disabled={!canChat}
                >
                  <span className="suggestion-arrow" aria-hidden>
                    →
                  </span>
                  <span>{question}</span>
                </button>
              ))}
            </div>
            {!canChat && (
              <p className="empty-note">
                Chat is disabled until the backend and LLM are available.
              </p>
            )}
          </div>
        ) : (
          messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              busy={streaming}
              onResume={handleResume}
            />
          ))
        )}
      </div>

      <div className="composer">
        <div className="composer-box">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            rows={1}
            disabled={!canChat || streaming}
            aria-label="Message QuenchGuard"
          />
          {streaming ? (
            <button type="button" className="btn btn-stop" onClick={stop}>
              <span className="stop-square" aria-hidden />
              Stop
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-send"
              onClick={() => send(input)}
              disabled={!canChat || !input.trim()}
            >
              Send
              <span aria-hidden>↑</span>
            </button>
          )}
        </div>
        <div className="composer-hint">
          <span>
            Enter to send <span className="hint-sep">·</span> Shift+Enter for newline
          </span>
          <span>{streaming ? "Streaming response…" : model ? `Model ${model}` : "Live tool trace"}</span>
        </div>
      </div>
    </section>
  );
}
