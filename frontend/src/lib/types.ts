// Agent types for the trace system

export type AgentName = "planner" | "researcher" | "fact_checker" | "synthesizer" | "system";

export type EventType =
  | "agent_started"
  | "agent_completed"  // Individual agent finished its work
  | "tool_called"
  | "tool_result"
  | "claim_extracted"
  | "claim_verified"
  | "report_generating"
  | "checkpoint"
  | "hitl_checkpoint"  // HITL checkpoint (workflow paused)
  | "complete"  // Workflow-level completion with report
  | "error"
  | "heartbeat"  // Keep-alive heartbeat
  | "connected";  // Initial connection

export interface TraceEvent {
  id: string;
  agent: AgentName;
  action: string;
  detail: string;
  timestamp: string;
  latency_ms?: number;
  tokens_used?: number;
  tool?: string;
  type: EventType;
  data?: any; // Additional event data (e.g., report data on complete event)
}

export interface QuerySubmission {
  query: string;
  stream_id: string;
}

export type StreamStatus = "idle" | "running" | "hitl" | "synth" | "complete" | "error";

export interface Citation {
  number: number;
  title: string;
  url: string;
  domain?: string;
  source_quality_score: number;
  accessed_date: string;
}

export interface ReportSection {
  heading: string;
  content: string;
}

export interface Report {
  title: string;
  summary: string;
  sections: ReportSection[];
  confidence_summary: string;
  citations: Citation[];
}

export interface VerificationClaim {
  claim: string;
  ok: boolean;
}

// Detailed verification result (matching backend schema)
export interface VerificationResult {
  claim: {
    statement: string;
    source_url: string;
    source_title: string;
    confidence: "high" | "medium" | "low";
    category: string;
  };
  verdict: "supported" | "contradicted" | "unverifiable";
  confidence: number;
  supporting_sources: string[];
  contradicting_sources: string[];
  reasoning: string;
}

// HITL checkpoint data
export interface HITLCheckpoint {
  summary: string;
  claims: VerificationResult[];
  source_quality: number;
}

// Report response from API
export interface ReportResponse {
  stream_id: string;
  status: string;
  report: Report | null;
  raw_report: string | null;
  metrics: {
    claims_count?: number;
    verification_count?: number;
    citations_count?: number;
    trace_events_count?: number;
    iteration_count?: number;
    fact_check_pass_rate?: number;
    total_latency_ms?: number;
    total_tokens?: number;
  };
  error: string | null;
}

// Feedback action types
export type FeedbackAction = "approve" | "dig_deeper" | "correct" | "abort";

// Design system colors for component styling
export const DS = {
  colors: {
    bg: "#0C0C0C",
    surface: "#141414",
    surfaceHover: "#1A1A1A",
    border: "#282828",
    borderDotted: "#333333",
    text: "#E0E0E0",
    textMid: "#999999",
    textDim: "#5A5A5A",
    white: "#FFFFFF",
    green: "#5AE05A",
    yellow: "#D4C55A",
    red: "#D45A5A",
  },
} as const;
