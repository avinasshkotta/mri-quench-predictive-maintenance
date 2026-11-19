/** Types mirroring the QuenchGuard FastAPI contract. */

export type DataSource = "silver" | "demo" | "none";

export interface StatusResponse {
  status: string;
  llm_configured: boolean;
  model: string | null;
  artifacts_loaded: boolean;
  data_source: DataSource;
}

export interface FleetEntry {
  magnet_id: string;
  quench_probability: number;
  alert: boolean;
}

export interface FleetResponse {
  fleet: FleetEntry[];
  threshold: number;
  cached: boolean;
}

export interface ShapFactor {
  feature: string;
  shap_value: number;
  direction: string;
}

export interface MagnetDetailResponse {
  magnet_id: string;
  quench_probability: number;
  alert: boolean;
  threshold: number;
  horizon_hours?: number | null;
  min_lead_hours?: number | null;
  baseline_probability?: number | null;
  explanation?: ShapFactor[] | null;
  explanation_model?: string | null;
}

export interface TelemetryPoint {
  t: string;
  v: number;
}

export interface TelemetryResponse {
  magnet_id: string;
  hours?: number;
  pressure?: TelemetryPoint[] | null;
  cold_head?: TelemetryPoint[] | null;
  vibration?: TelemetryPoint[] | null;
}

/* ------------------------------------------------------------------ */
/* Conversation model                                                  */
/* ------------------------------------------------------------------ */

export type ChatRole = "user" | "assistant";
export type HitlStatus = "pending" | "approved" | "rejected";
export type Priority = "low" | "medium" | "high" | "critical";

export interface TicketArgs {
  magnet_id?: string;
  priority?: string;
  summary?: string;
  recommended_action?: string;
}
export interface HitlRequest {
  type: string;
  call_id: string;
  args: TicketArgs;
  question?: string;
  status: HitlStatus;
  /** Reason supplied by the operator when a ticket is rejected. */
  reason?: string | null;
}

export interface ToolCall {
  id: string;
  name: string;
  args: unknown;
  content?: string;
  truncated?: boolean;
  done: boolean;
  /** The stream ended (stop / error) before this tool reported a result. */
  interrupted?: boolean;
  /** A HITL gate is paused on this tool (e.g. create_ticket). */
  waiting?: boolean;
}

export interface Citation {
  source: string;
  heading?: string;
  score?: number;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  streaming?: boolean;
  stopped?: boolean;
  tools: ToolCall[];
  citations: Citation[];
  hitl: HitlRequest[];
  error?: string | null;
}

/* ------------------------------------------------------------------ */
/* SSE payloads (validated at the boundary with the helpers below)     */
/* ------------------------------------------------------------------ */

export interface MetaPayload {
  thread_id: string;
  model?: string;
}

export interface TokenPayload {
  text: string;
}

export interface ToolStartPayload {
  name: string;
  args?: unknown;
}

export interface ToolResultPayload {
  name: string;
  content?: string;
  truncated?: boolean;
}

export interface CitationPayload {
  source: string;
  heading?: string;
  score?: number;
}

export interface InterruptPayload {
  type: string;
  call_id: string;
  args?: unknown;
  question?: string;
}

export interface ErrorPayload {
  message: string;
}

/* ------------------------------------------------------------------ */
/* Narrowing helpers                                                   */
/* ------------------------------------------------------------------ */

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function readString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

export function readNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function readBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}
