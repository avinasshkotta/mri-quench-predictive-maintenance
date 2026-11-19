"use client";

import { useCallback, useEffect, useState } from "react";
import { getMagnet, getTelemetry, isAbortError } from "@/lib/api";
import {
  errorMessage,
  formatHours,
  formatProbability,
  formatRate,
  formatSigned,
} from "@/lib/format";
import type { MagnetDetailResponse, ShapFactor, TelemetryResponse } from "@/lib/types";
import Sparkline from "./Sparkline";

interface MagnetDetailProps {
  magnetId: string | null;
  refreshKey: number;
}

const TELEMETRY_HOURS = 72;

function topFactors(factors: ShapFactor[] | null | undefined): ShapFactor[] {
  if (!Array.isArray(factors)) return [];
  return [...factors]
    .filter((factor) => Number.isFinite(factor.shap_value))
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 3);
}

function ShapTable({ factors }: { factors: ShapFactor[] }) {
  if (factors.length === 0) {
    return <p className="panel-empty">No factor explanation available.</p>;
  }
  const maxAbs = Math.max(...factors.map((factor) => Math.abs(factor.shap_value)), 0.000001);

  return (
    <table className="shap-table">
      <thead>
        <tr>
          <th>Feature</th>
          <th aria-label="Relative impact">Impact</th>
          <th className="shap-value-head">SHAP</th>
        </tr>
      </thead>
      <tbody>
        {factors.map((factor) => {
          const positive = factor.shap_value >= 0;
          return (
            <tr key={factor.feature}>
              <td className="shap-feature" title={factor.feature}>
                <span className="feature-name">{factor.feature}</span>
                <span className="shap-dir">{factor.direction}</span>
              </td>
              <td className="shap-impact">
                <span className="shap-bar-track">
                  <span
                    className={`shap-bar ${positive ? "pos" : "neg"}`}
                    style={{
                      width: `${Math.max(8, (Math.abs(factor.shap_value) / maxAbs) * 100)}%`,
                    }}
                  />
                </span>
              </td>
              <td className={`shap-value ${positive ? "pos" : "neg"}`}>
                {formatSigned(factor.shap_value)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function MagnetDetail({ magnetId, refreshKey }: MagnetDetailProps) {
  const [detail, setDetail] = useState<MagnetDetailResponse | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [telemetryError, setTelemetryError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    if (!magnetId) {
      setDetail(null);
      setTelemetry(null);
      setError(null);
      setTelemetryError(null);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    setLoading(true);
    setDetail(null);
    setTelemetry(null);
    setError(null);
    setTelemetryError(null);

    void (async () => {
      const [detailResult, telemetryResult] = await Promise.allSettled([
        getMagnet(magnetId, controller.signal),
        getTelemetry(magnetId, TELEMETRY_HOURS, controller.signal),
      ]);
      if (controller.signal.aborted) return;

      if (detailResult.status === "fulfilled") {
        setDetail(detailResult.value);
      } else if (!isAbortError(detailResult.reason)) {
        setError(errorMessage(detailResult.reason));
      }

      if (telemetryResult.status === "fulfilled") {
        setTelemetry(telemetryResult.value);
      } else if (!isAbortError(telemetryResult.reason)) {
        setTelemetryError(errorMessage(telemetryResult.reason));
      }

      setLoading(false);
    })();

    return () => controller.abort();
  }, [magnetId, refreshKey, nonce]);

  const alert = detail
    ? typeof detail.alert === "boolean"
      ? detail.alert
      : detail.quench_probability >= detail.threshold
    : false;
  const factors = topFactors(detail?.explanation);

  return (
    <section className="panel detail-panel" aria-label="Selected magnet detail">
      <div className="panel-head detail-panel-head">
        <div>
          <span className="kicker">Magnet detail</span>
          <h2 className="panel-title detail-id">{magnetId ?? "No selection"}</h2>
        </div>
        {magnetId && (
          <button
            type="button"
            className="icon-btn"
            onClick={reload}
            disabled={loading}
            aria-label="Refresh magnet detail"
            title="Refresh magnet detail"
          >
            <span className={loading ? "spin" : undefined} aria-hidden>
              ↻
            </span>
          </button>
        )}
      </div>

      {!magnetId && (
        <p className="panel-empty detail-empty">
          Select a magnet from the fleet list — or ask the assistant about one — to inspect its risk
          factors and sensor telemetry.
        </p>
      )}

      {magnetId && loading && !detail && (
        <div className="detail-skeleton" aria-label="Loading magnet detail">
          <div className="skeleton-block" />
          <div className="skeleton-grid">
            <div className="skeleton-cell" />
            <div className="skeleton-cell" />
            <div className="skeleton-cell" />
            <div className="skeleton-cell" />
          </div>
          <div className="skeleton-block short" />
        </div>
      )}

      {magnetId && error && (
        <div className="detail-body">
          <div className="inline-error" role="alert">
            <span>{error}</span>
            <button type="button" className="link-btn" onClick={reload}>
              Retry
            </button>
          </div>
        </div>
      )}

      {detail && (
        <>
          <div className="detail-metric">
            <div>
              <span className="kicker">Quench probability</span>
              <div className="big-prob">{formatProbability(detail.quench_probability)}</div>
            </div>
            <div className="metric-flags">
              <span className={`badge ${alert ? "badge-alert" : "badge-ok"}`}>
                {alert ? "ALERT" : "NORMAL"}
              </span>
              <span className="muted-xs">alert ≥ {formatRate(detail.threshold)}</span>
            </div>
          </div>

          <div className="detail-body">
            <div className="kv-grid">
              <div className="kv">
                <span className="kicker">Baseline</span>
                <span className="value">{formatProbability(detail.baseline_probability)}</span>
              </div>
              <div className="kv">
                <span className="kicker">Threshold</span>
                <span className="value">{formatRate(detail.threshold)}</span>
              </div>
              <div className="kv">
                <span className="kicker">Horizon</span>
                <span className="value">{formatHours(detail.horizon_hours)}</span>
              </div>
              <div className="kv">
                <span className="kicker">Min lead</span>
                <span className="value">{formatHours(detail.min_lead_hours)}</span>
              </div>
            </div>
          </div>

          <div className="detail-section">
            <div className="detail-section-head">
              <span className="kicker">Top risk factors</span>
              {detail.explanation_model && (
                <span className="model-tag" title="Explanation model">
                  {detail.explanation_model}
                </span>
              )}
            </div>
            <ShapTable factors={factors} />
          </div>

          <div className="detail-section">
            <div className="detail-section-head">
              <span className="kicker">Sensor telemetry</span>
              <span className="muted-xs">
                last {telemetry?.hours ?? TELEMETRY_HOURS} h
              </span>
            </div>

            {telemetryError && (
              <div className="inline-error telemetry-error" role="alert">
                <span>{telemetryError}</span>
                <button type="button" className="link-btn" onClick={reload}>
                  Retry
                </button>
              </div>
            )}

            <div className="spark-stack">
              <Sparkline label="Helium pressure (MPa)" points={telemetry?.pressure ?? []} />
              <Sparkline
                label="Cold head (K)"
                points={telemetry?.cold_head ?? []}
                color="#22d3ee"
              />
              <Sparkline
                label="Vibration RMS"
                points={telemetry?.vibration ?? []}
                color="#a78bfa"
              />
            </div>
          </div>
        </>
      )}
    </section>
  );
}
