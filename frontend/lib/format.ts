/** Small presentation helpers shared across components. */

/** 6-decimal probability, matching the API payload precision. */
export function formatProbability(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(6);
}

/** Alert threshold, e.g. 0.9 -> "0.90". */
export function formatRate(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(2);
}

/** Signed SHAP value, e.g. -0.1741 -> "-0.1741". */
export function formatSigned(value: number | null | undefined, digits = 4): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

/** Relevance scores on citation chips. */
export function formatScore(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(2);
}

/** Adaptive decimal count for raw sensor readings. */
export function formatMetric(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 10) return value.toFixed(2);
  if (abs >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

/** Local wall-clock time for "refreshed at" labels. */
export function formatClock(date: Date | null | undefined): string {
  if (!date || Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/** Compact telemetry timestamp, e.g. "01-01 14:30". */
export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString([], {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export function formatHours(hours: number | null | undefined): string {
  if (typeof hours !== "number" || !Number.isFinite(hours)) return "—";
  return `${hours} h`;
}

/** Compact one-line preview of tool arguments. */
export function argPreview(args: unknown, max = 160): string {
  if (args === null || args === undefined) return "";
  let text: string;
  if (typeof args === "string") {
    text = args;
  } else {
    try {
      text = JSON.stringify(args);
    } catch {
      text = String(args);
    }
  }
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

export function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

/** UUID with a Math.random fallback for non-secure contexts. */
export function createId(): string {
  const cryptoObj = typeof globalThis === "object" ? globalThis.crypto : undefined;
  if (cryptoObj && typeof cryptoObj.randomUUID === "function") {
    try {
      return cryptoObj.randomUUID();
    } catch {
      /* fall through */
    }
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (char) => {
    const random = Math.floor(Math.random() * 16);
    const value = char === "x" ? random : (random & 0x3) | 0x8;
    return value.toString(16);
  });
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error) return error;
  return "Something went wrong. Please try again.";
}

export type PriorityTone = "low" | "medium" | "high" | "critical" | "unknown";

export function priorityTone(priority: string | null | undefined): PriorityTone {
  const normalized = (priority ?? "").toLowerCase();
  if (normalized === "low" || normalized === "medium" || normalized === "high" || normalized === "critical") {
    return normalized;
  }
  return "unknown";
}
