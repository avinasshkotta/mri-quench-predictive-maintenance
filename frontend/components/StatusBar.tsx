"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getStatus } from "@/lib/api";
import { errorMessage } from "@/lib/format";
import type { StatusResponse } from "@/lib/types";

type Tone = "good" | "bad" | "warn" | "info" | "idle";

interface StatusBarProps {
  onStatusChange: (status: StatusResponse | null, reachable: boolean) => void;
}

function Pill({ tone, label, title }: { tone: Tone; label: string; title?: string }) {
  return (
    <span className={`pill tone-${tone}`} title={title}>
      <span className="pill-dot" aria-hidden />
      {label}
    </span>
  );
}

export default function StatusBar({ onStatusChange }: StatusBarProps) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);
  const [checking, setChecking] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const seqRef = useRef(0);

  const check = useCallback(async () => {
    const seq = seqRef.current + 1;
    seqRef.current = seq;
    setChecking(true);
    try {
      const data = await getStatus();
      if (seq !== seqRef.current) return;
      setStatus(data);
      setReachable(true);
      setError(null);
      onStatusChange(data, true);
    } catch (err) {
      if (seq !== seqRef.current) return;
      setStatus(null);
      setReachable(false);
      setError(errorMessage(err));
      onStatusChange(null, false);
    } finally {
      if (seq === seqRef.current) setChecking(false);
    }
  }, [onStatusChange]);

  useEffect(() => {
    void check();
    const timer = window.setInterval(() => {
      void check();
    }, 30_000);
    return () => {
      window.clearInterval(timer);
      seqRef.current += 1;
    };
  }, [check]);

  const offline = reachable === false;
  const llmTone: Tone = status ? (status.llm_configured ? "good" : "warn") : "idle";
  const artifactTone: Tone = status ? (status.artifacts_loaded ? "good" : "warn") : "idle";
  const dataTone: Tone =
    status?.data_source === "demo" ? "warn" : status?.data_source === "none" ? "bad" : "info";

  return (
    <header className="status-bar">
      <div className="status-inner">
        <div className="status-bar-row">
          <div className="status-left">
            <span className="status-title">Fleet operations</span>
            <span
              className={`live-dot ${offline ? "is-off" : reachable ? "is-on" : "is-idle"}`}
              aria-hidden
            />
            <span className={`status-mode ${offline ? "is-off" : ""}`}>
              {offline ? "offline" : reachable ? "live" : "connecting"}
            </span>
          </div>

          <div className="status-pills">
            <Pill
              tone={offline ? "bad" : reachable ? "good" : "idle"}
              label={offline ? "API offline" : reachable ? "API online" : "API …"}
              title={error ?? undefined}
            />
            <Pill
              tone={llmTone}
              label={
                status
                  ? status.llm_configured
                    ? `LLM ready${status.model ? ` · ${status.model}` : ""}`
                    : "LLM not configured"
                  : "LLM …"
              }
            />
            <Pill
              tone={artifactTone}
              label={
                status
                  ? status.artifacts_loaded
                    ? "Artifacts loaded"
                    : "Artifacts missing"
                  : "Artifacts …"
              }
            />
            <Pill
              tone={status ? dataTone : "idle"}
              label={status ? `Data: ${status.data_source}` : "Data …"}
              title={status ? `Data source: ${status.data_source}` : undefined}
            />
            <button
              type="button"
              className="icon-btn status-refresh"
              onClick={() => void check()}
              disabled={checking}
              aria-label="Refresh system status"
              title="Refresh system status"
            >
              <span className={checking ? "spin" : undefined} aria-hidden>
                ↻
              </span>
            </button>
          </div>
        </div>

        {offline && (
          <div className="offline-banner" role="alert">
            <span className="offline-icon" aria-hidden>
              ⚠
            </span>
            <span>
              Backend offline — start it with <code>python scripts/serve.py</code>
            </span>
            {error && <span className="offline-detail">{error}</span>}
            <button
              type="button"
              className="banner-retry"
              onClick={() => void check()}
              disabled={checking}
            >
              {checking ? "Retrying…" : "Retry"}
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
