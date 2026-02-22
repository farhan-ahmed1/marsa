import { describe, it, expect } from "vitest";
import { cn, computeQueryMetrics, metricsToItems } from "@/lib/utils";
import type { TraceEvent, Report } from "@/lib/types";

describe("cn", () => {
  it("merges class names", () => {
    expect(cn("px-2", "py-1")).toBe("px-2 py-1");
  });

  it("deduplicates conflicting tailwind classes", () => {
    expect(cn("px-2", "px-4")).toBe("px-4");
  });

  it("handles conditional classes", () => {
    expect(cn("base", false && "hidden", "extra")).toBe("base extra");
  });
});

describe("computeQueryMetrics", () => {
  const baseEvent: TraceEvent = {
    id: "e1",
    type: "agent_started",
    agent: "planner",
    action: "start",
    detail: "",
    timestamp: "2025-01-01T00:00:00.000Z",
  };

  it("returns zeros for empty events and no report", () => {
    const m = computeQueryMetrics([], null);
    expect(m.latencyMs).toBe(0);
    expect(m.llmCalls).toBe(0);
    expect(m.totalTokens).toBe(0);
    expect(m.toolCalls).toBe(0);
    expect(m.sourceCount).toBe(0);
    expect(m.avgSourceQuality).toBe(0);
    expect(m.factCheckPassRate).toBe(0);
  });

  it("calculates latency from events when no apiMetrics", () => {
    const events: TraceEvent[] = [
      { ...baseEvent, timestamp: "2025-01-01T00:00:00.000Z" },
      { ...baseEvent, id: "e2", timestamp: "2025-01-01T00:00:05.000Z", latency_ms: 200 },
    ];
    const m = computeQueryMetrics(events, null);
    expect(m.latencyMs).toBe(5200); // 5000ms gap + 200ms last latency
  });

  it("uses apiMetrics for latency when provided", () => {
    const events: TraceEvent[] = [baseEvent];
    const m = computeQueryMetrics(events, null, { total_latency_ms: 9000 });
    expect(m.latencyMs).toBe(9000);
  });

  it("counts LLM calls and tokens", () => {
    const events: TraceEvent[] = [
      { ...baseEvent, tokens_used: 500 },
      { ...baseEvent, id: "e2", tokens_used: 300 },
      { ...baseEvent, id: "e3" }, // no tokens
    ];
    const m = computeQueryMetrics(events, null);
    expect(m.llmCalls).toBe(2);
    expect(m.totalTokens).toBe(800);
  });

  it("counts tool calls", () => {
    const events: TraceEvent[] = [
      { ...baseEvent, type: "tool_called" },
      { ...baseEvent, id: "e2", type: "tool_called" },
      { ...baseEvent, id: "e3", type: "tool_result" },
    ];
    const m = computeQueryMetrics(events, null);
    expect(m.toolCalls).toBe(2);
  });

  it("computes source metrics from report citations", () => {
    const report: Report = {
      title: "Test",
      summary: "Test report",
      sections: [],
      confidence_summary: "",
      citations: [
        { number: 1, title: "A", url: "https://a.com", source_quality_score: 0.8, accessed_date: "2025-01-01" },
        { number: 2, title: "B", url: "https://b.com", source_quality_score: 0.6, accessed_date: "2025-01-01" },
      ],
    };
    const m = computeQueryMetrics([], report);
    expect(m.sourceCount).toBe(2);
    expect(m.avgSourceQuality).toBeCloseTo(0.7);
  });

  it("uses apiMetrics for fact-check pass rate", () => {
    const m = computeQueryMetrics([], null, {
      fact_check_pass_rate: 0.85,
      verification_count: 10,
      claims_count: 12,
    });
    expect(m.factCheckPassRate).toBe(0.85);
    expect(m.verifiedCount).toBe(10);
    expect(m.totalClaims).toBe(12);
  });
});

describe("metricsToItems", () => {
  it("formats millisecond latency", () => {
    const items = metricsToItems({
      latencyMs: 500,
      llmCalls: 3,
      totalTokens: 2500,
      toolCalls: 5,
      sourceCount: 4,
      avgSourceQuality: 0.75,
      factCheckPassRate: 0.9,
      verifiedCount: 9,
      totalClaims: 10,
    });
    expect(items[0].value).toBe("500ms");
    expect(items[1].value).toBe("3");
    expect(items[1].sub).toBe("2.5k tok");
    expect(items[2].value).toBe("5");
    expect(items[2].sub).toBe("4 sources");
    expect(items[3].value).toBe("90%");
    expect(items[3].sub).toBe("avg 7.5/10");
  });

  it("formats second-level latency", () => {
    const items = metricsToItems({
      latencyMs: 12500,
      llmCalls: 0,
      totalTokens: 0,
      toolCalls: 0,
      sourceCount: 0,
      avgSourceQuality: 0,
      factCheckPassRate: 0,
      verifiedCount: 0,
      totalClaims: 0,
    });
    expect(items[0].value).toBe("12.5s");
  });

  it("shows dash when no fact check data", () => {
    const items = metricsToItems({
      latencyMs: 0,
      llmCalls: 0,
      totalTokens: 0,
      toolCalls: 0,
      sourceCount: 0,
      avgSourceQuality: 0,
      factCheckPassRate: 0,
      verifiedCount: 0,
      totalClaims: 0,
    });
    // with no pass rate and no claims, it shows "—"
    expect(items[3].value).toBe("\u2014");
  });

  it("shows verified/total when pass rate is 0 but claims exist", () => {
    const items = metricsToItems({
      latencyMs: 0,
      llmCalls: 0,
      totalTokens: 0,
      toolCalls: 0,
      sourceCount: 0,
      avgSourceQuality: 0,
      factCheckPassRate: 0,
      verifiedCount: 5,
      totalClaims: 10,
    });
    expect(items[3].value).toBe("5/10");
  });
});
