// Agent types for the trace system

export type AgentName = "planner" | "researcher" | "fact_checker" | "synthesizer" | "system";

export type EventType =
  | "agent_started"
  | "tool_called"
  | "tool_result"
  | "claim_extracted"
  | "claim_verified"
  | "report_generating"
  | "checkpoint"
  | "complete"
  | "error";

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
