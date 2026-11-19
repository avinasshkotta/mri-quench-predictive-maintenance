import type {
  FleetResponse,
  MagnetDetailResponse,
  StatusResponse,
  TelemetryResponse,
} from "@/lib/types";
import { isRecord, readString } from "@/lib/types";

/**
 * API origin resolution, in priority order:
 * 1. `NEXT_PUBLIC_API_BASE` at build time (split hosting: UI on Vercel, API
 *    on a container host).
 * 2. Local development: Next dev server on :3000, FastAPI on :8000.
 * 3. Production single container: FastAPI serves the static export from the
 *    same origin.
 */
const CONFIGURED_API_BASE = process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "");

export const API_BASE =
  CONFIGURED_API_BASE ??
  (typeof window !== "undefined" &&
  (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
    ? "http://127.0.0.1:8000"
    : "");

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const NETWORK_ERROR =
  "Could not reach the backend. Check that it is running on http://127.0.0.1:8000.";

export function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name?: unknown }).name === "AbortError"
  );
}

async function readErrorDetail(res: Response): Promise<string> {
  let detail = res.statusText || `Request failed (${res.status})`;
  try {
    const body: unknown = await res.json();
    if (isRecord(body)) {
      const parsed = readString(body.detail);
      if (parsed) detail = parsed;
    }
  } catch {
    /* non-JSON error body — keep the status text */
  }
  return detail;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, init);
  } catch (error) {
    if (isAbortError(error)) throw error;
    throw new ApiError(NETWORK_ERROR, 0);
  }
  if (!res.ok) {
    throw new ApiError(await readErrorDetail(res), res.status);
  }
  return (await res.json()) as T;
}

export function getStatus(signal?: AbortSignal): Promise<StatusResponse> {
  return request<StatusResponse>("/api/status", { signal, cache: "no-store" });
}

export function getFleet(signal?: AbortSignal): Promise<FleetResponse> {
  return request<FleetResponse>("/api/fleet", { signal, cache: "no-store" });
}

export function getMagnet(magnetId: string, signal?: AbortSignal): Promise<MagnetDetailResponse> {
  return request<MagnetDetailResponse>(`/api/magnet/${encodeURIComponent(magnetId)}`, {
    signal,
    cache: "no-store",
  });
}

export function getTelemetry(
  magnetId: string,
  hours = 72,
  signal?: AbortSignal,
): Promise<TelemetryResponse> {
  const query = new URLSearchParams({ hours: String(hours) });
  return request<TelemetryResponse>(
    `/api/magnet/${encodeURIComponent(magnetId)}/telemetry?${query.toString()}`,
    { signal, cache: "no-store" },
  );
}

/**
 * Read a text/event-stream response and dispatch each SSE block.
 *
 * Handles CRLF line endings, multi-line `data:` fields and a final block that
 * the server closes without a trailing blank line.
 */
export async function streamSSE(
  path: string,
  body: unknown,
  onEvent: (type: string, data: unknown) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (isAbortError(error)) throw error;
    throw new ApiError(NETWORK_ERROR, 0);
  }

  if (!res.ok || !res.body) {
    throw new ApiError(await readErrorDetail(res), res.status);
  }

  const dispatch = (block: string): void => {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length === 0) return;
    const raw = dataLines.join("\n");
    try {
      onEvent(event, JSON.parse(raw));
    } catch {
      onEvent(event, raw);
    }
  };

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let idx = buf.indexOf("\n\n");
    while (idx !== -1) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      if (block.trim()) dispatch(block);
      idx = buf.indexOf("\n\n");
    }
  }

  buf += decoder.decode().replace(/\r\n/g, "\n");
  if (buf.trim()) dispatch(buf);
}
