"use client";

import { useId } from "react";
import { formatMetric, formatTimestamp } from "@/lib/format";
import type { TelemetryPoint } from "@/lib/types";

const WIDTH = 280;
const HEIGHT = 60;
const PAD_X = 4;
const PAD_Y = 7;
const MAX_POINTS = 240;

interface SparklineProps {
  label: string;
  points: TelemetryPoint[];
  color?: string;
}

function cleanPoints(points: TelemetryPoint[]): TelemetryPoint[] {
  if (!Array.isArray(points)) return [];
  return points.filter((point) => point != null && Number.isFinite(point.v));
}

function downsample(points: TelemetryPoint[], max: number): TelemetryPoint[] {
  if (points.length <= max) return points;
  const step = (points.length - 1) / (max - 1);
  const output: TelemetryPoint[] = [];
  for (let index = 0; index < max; index += 1) {
    const point = points[Math.round(index * step)];
    if (point) output.push(point);
  }
  return output;
}

export default function Sparkline({ label, points, color = "#2dd4bf" }: SparklineProps) {
  const gradientId = `spark-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  const data = downsample(cleanPoints(points), MAX_POINTS);

  if (data.length === 0) {
    return (
      <div className="spark is-empty">
        <div className="spark-head">
          <span className="spark-label">{label}</span>
          <span className="spark-latest muted">—</span>
        </div>
        <div className="spark-placeholder">No telemetry</div>
      </div>
    );
  }

  const values = data.map((point) => point.v);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;
  const innerW = WIDTH - PAD_X * 2;
  const innerH = HEIGHT - PAD_Y * 2;

  const coords = data.map((point, index) => {
    const x =
      data.length === 1
        ? PAD_X + innerW / 2
        : PAD_X + (index / (data.length - 1)) * innerW;
    const ratio = span === 0 ? 0.5 : (point.v - min) / span;
    const y = HEIGHT - PAD_Y - ratio * innerH;
    return { x, y };
  });

  const first = coords[0];
  const last = coords[coords.length - 1];
  if (!first || !last) return null;

  const line = coords.map((coord) => `${coord.x.toFixed(2)},${coord.y.toFixed(2)}`).join(" ");
  const area = [
    `M ${first.x.toFixed(2)} ${(HEIGHT - PAD_Y).toFixed(2)}`,
    ...coords.map((coord) => `L ${coord.x.toFixed(2)} ${coord.y.toFixed(2)}`),
    `L ${last.x.toFixed(2)} ${(HEIGHT - PAD_Y).toFixed(2)}`,
    "Z",
  ].join(" ");

  const latest = data[data.length - 1];
  const latestValue = latest ? latest.v : null;
  const latestTime = latest ? latest.t : null;

  return (
    <div className="spark">
      <div className="spark-head">
        <span className="spark-label">{label}</span>
        <span className="spark-latest" style={{ color }}>
          {formatMetric(latestValue)}
        </span>
      </div>

      <svg
        className="spark-svg"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${label}: ${data.length} samples, min ${formatMetric(min)}, max ${formatMetric(max)}`}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.26" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>

        <line
          x1={PAD_X}
          x2={WIDTH - PAD_X}
          y1={PAD_Y + innerH / 3}
          y2={PAD_Y + innerH / 3}
          className="spark-grid"
        />
        <line
          x1={PAD_X}
          x2={WIDTH - PAD_X}
          y1={PAD_Y + (innerH * 2) / 3}
          y2={PAD_Y + (innerH * 2) / 3}
          className="spark-grid"
        />

        <path d={area} fill={`url(#${gradientId})`} />
        <polyline
          points={line}
          fill="none"
          stroke={color}
          strokeWidth="1.7"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
        {data.length === 1 && <circle cx={first.x} cy={first.y} r="2.4" fill={color} />}
        <circle cx={last.x} cy={last.y} r="2.2" fill={color} />
      </svg>

      <div className="spark-foot">
        <span>min {formatMetric(min)}</span>
        <span className="spark-when">{latestTime ? formatTimestamp(latestTime) : "—"}</span>
        <span>max {formatMetric(max)}</span>
      </div>
    </div>
  );
}
