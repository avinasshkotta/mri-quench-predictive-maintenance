"use client";

import { useCallback, useEffect, useState } from "react";
import { getFleet, isAbortError } from "@/lib/api";
import { errorMessage, formatClock, formatProbability, formatRate } from "@/lib/format";
import type { FleetEntry, FleetResponse } from "@/lib/types";

interface FleetPanelProps {
  selectedId: string | null;
  onSelect: (magnetId: string) => void;
}

function byRisk(a: FleetEntry, b: FleetEntry): number {
  return b.quench_probability - a.quench_probability;
}

export default function FleetPanel({ selectedId, onSelect }: FleetPanelProps) {
  const [data, setData] = useState<FleetResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshedAt, setRefreshedAt] = useState<Date | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const next = await getFleet(signal);
      if (signal?.aborted) return;
      setData(next);
      setRefreshedAt(new Date());
    } catch (err) {
      if (isAbortError(err) || signal?.aborted) return;
      setError(errorMessage(err));
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const fleet = data && Array.isArray(data.fleet) ? [...data.fleet].sort(byRisk) : [];

  // Focus the riskiest magnet once the first fleet snapshot arrives.
  useEffect(() => {
    if (!data || selectedId || !Array.isArray(data.fleet) || data.fleet.length === 0) return;
    const top = [...data.fleet].sort(byRisk)[0];
    if (top) onSelect(top.magnet_id);
  }, [data, selectedId, onSelect]);

  const threshold = data ? formatRate(data.threshold) : "—";

  return (
    <section className="panel fleet-panel" aria-label="Fleet risk">
      <div className="panel-head">
        <div>
          <span className="kicker">Live risk</span>
          <h2 className="panel-title">Fleet overview</h2>
        </div>
        <button
          type="button"
          className="icon-btn"
          onClick={() => void load()}
          disabled={loading}
          aria-label="Refresh fleet risk"
          title="Refresh fleet risk"
        >
          <span className={loading ? "spin" : undefined} aria-hidden>
            ↻
          </span>
        </button>
      </div>

      <div className="panel-meta">
        <span>
          quench probability <span className="meta-dot">·</span> alert ≥{" "}
          <span className="mono">{threshold}</span>
        </span>
        <span className="meta-right">
          {data?.cached && <span className="chip-tiny">cached</span>}
          refreshed {refreshedAt ? formatClock(refreshedAt) : "—"}
        </span>
      </div>

      <div className="fleet-body">
        {loading && !data && (
          <div className="fleet-list" aria-label="Loading fleet risk">
            <div className="skeleton-row" />
            <div className="skeleton-row" />
            <div className="skeleton-row" />
          </div>
        )}

        {error && (
          <div className="inline-error fleet-error" role="alert">
            <span>{error}</span>
            <button type="button" className="link-btn" onClick={() => void load()}>
              Retry
            </button>
          </div>
        )}

        {!error && data && fleet.length === 0 && (
          <p className="panel-empty">No magnets reported by the fleet endpoint.</p>
        )}

        {fleet.length > 0 && (
          <>
            <div className="fleet-cols" aria-hidden>
              <span>Magnet</span>
              <span>Quench p</span>
            </div>
            <div className="fleet-list">
              {fleet.map((entry) => {
                const selected = selectedId === entry.magnet_id;
                return (
                  <button
                    key={entry.magnet_id}
                    type="button"
                    className={`fleet-card ${selected ? "is-selected" : ""} ${
                      entry.alert ? "is-alert" : ""
                    }`}
                    onClick={() => onSelect(entry.magnet_id)}
                    aria-pressed={selected}
                    title={`${entry.magnet_id} — quench probability ${formatProbability(
                      entry.quench_probability,
                    )}`}
                  >
                    <span className="fleet-left">
                      <span className="fleet-id">{entry.magnet_id}</span>
                      {entry.alert && <span className="badge badge-alert">ALERT</span>}
                    </span>
                    <span className="fleet-prob">{formatProbability(entry.quench_probability)}</span>
                  </button>
                );
              })}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
