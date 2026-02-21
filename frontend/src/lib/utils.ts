import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import type { TraceEvent, Report } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export interface QueryMetrics {
  latencyMs: number;
  llmCalls: number;
  totalTokens: number;
  toolCalls: number;
  sourceCount: number;
  avgSourceQuality: number;
  factCheckPassRate: number;
  verifiedCount: number;
  totalClaims: number;
}

export function computeQueryMetrics(
  events: TraceEvent[],
  report: Report | null,
  apiMetrics?: {
    fact_check_pass_rate?: number;
    verification_count?: number;
    claims_count?: number;
    total_latency_ms?: number;
    total_tokens?: number;
  }
): QueryMetrics {
  // Latency: from first event to last event + last latency
  let latencyMs = apiMetrics?.total_latency_ms ?? 0;
  if (!latencyMs && events.length >= 2) {
    const firstTs = new Date(events[0].timestamp).getTime();
    const lastEv = events[events.length - 1];
    const lastTs = new Date(lastEv.timestamp).getTime();
    latencyMs = lastTs - firstTs + (lastEv.latency_ms ?? 0);
  }

  // LLM calls and tokens
  const llmEvents = events.filter((e) => (e.tokens_used ?? 0) > 0);
  const llmCalls = llmEvents.length;
  const totalTokens =
    apiMetrics?.total_tokens ??
    events.reduce((a, e) => a + (e.tokens_used ?? 0), 0);

  // Tool calls
  const toolCalls = events.filter((e) => e.type === "tool_called").length;

  // Sources from report citations
  const citations = report?.citations ?? [];
  const sourceCount = citations.length;
  const avgSourceQuality =
    sourceCount > 0
      ? citations.reduce((a, c) => a + c.source_quality_score, 0) / sourceCount
      : 0;

  // Fact-check pass rate
  const factCheckPassRate = apiMetrics?.fact_check_pass_rate ?? 0;
  const verifiedCount = apiMetrics?.verification_count ?? 0;
  const totalClaims = apiMetrics?.claims_count ?? 0;

  return {
    latencyMs,
    llmCalls,
    totalTokens,
    toolCalls,
    sourceCount,
    avgSourceQuality,
    factCheckPassRate,
    verifiedCount,
    totalClaims,
  };
}

export function metricsToItems(m: QueryMetrics) {
  const latency =
    m.latencyMs < 1000
      ? `${Math.round(m.latencyMs)}ms`
      : `${(m.latencyMs / 1000).toFixed(1)}s`;

  const passRateStr =
    m.factCheckPassRate > 0
      ? `${Math.round(m.factCheckPassRate * 100)}%`
      : m.totalClaims > 0
      ? `${m.verifiedCount}/${m.totalClaims}`
      : "—";

  const qualityStr =
    m.avgSourceQuality > 0 ? `${(m.avgSourceQuality * 10).toFixed(1)}/10` : "—";

  return [
    { key: "latency", value: latency, sub: "total" },
    {
      key: "llm_calls",
      value: String(m.llmCalls),
      sub: `${(m.totalTokens / 1000).toFixed(1)}k tok`,
    },
    {
      key: "tool_calls",
      value: String(m.toolCalls),
      sub: `${m.sourceCount} sources`,
    },
    {
      key: "accuracy",
      value: passRateStr,
      sub: qualityStr !== "—" ? `avg ${qualityStr}` : "fact-check",
    },
  ];
}

