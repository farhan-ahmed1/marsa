import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useAgentStream } from "@/hooks/useAgentStream";

// Mock EventSource
class MockEventSource {
  static instances: MockEventSource[] = [];
  url: string;
  onopen: ((e: Event) => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  readyState = 0;
  close = vi.fn();

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }
}

beforeEach(() => {
  MockEventSource.instances = [];
  vi.stubGlobal("EventSource", MockEventSource);
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function sendSSE(es: MockEventSource, data: Record<string, unknown>) {
  es.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
}

describe("useAgentStream", () => {
  it("starts idle with no streamId", () => {
    const { result } = renderHook(() => useAgentStream(null));
    expect(result.current.status).toBe("idle");
    expect(result.current.events).toEqual([]);
    expect(result.current.error).toBeNull();
    expect(MockEventSource.instances).toHaveLength(0);
  });

  it("opens EventSource when streamId is provided", () => {
    const { result } = renderHook(() => useAgentStream("test-stream"));
    expect(MockEventSource.instances).toHaveLength(1);
    expect(MockEventSource.instances[0].url).toContain("/api/query/test-stream/stream");
    expect(result.current.status).toBe("running");
  });

  it("accumulates events from SSE messages", () => {
    const { result } = renderHook(() => useAgentStream("s1"));
    const es = MockEventSource.instances[0];

    act(() => {
      sendSSE(es, {
        type: "agent_started",
        data: { agent: "planner", action: "planning" },
        timestamp: "2025-01-01T00:00:00Z",
      });
    });

    expect(result.current.events).toHaveLength(1);
    expect(result.current.events[0].agent).toBe("planner");
    expect(result.current.events[0].type).toBe("agent_started");
  });

  it("skips heartbeat and connected events", () => {
    const { result } = renderHook(() => useAgentStream("s1"));
    const es = MockEventSource.instances[0];

    act(() => {
      sendSSE(es, { type: "heartbeat" });
      sendSSE(es, { type: "connected" });
    });

    expect(result.current.events).toHaveLength(0);
  });

  it("sets status to complete and closes on complete event", () => {
    const { result } = renderHook(() => useAgentStream("s1"));
    const es = MockEventSource.instances[0];

    act(() => {
      sendSSE(es, { type: "complete", data: { agent: "system" } });
    });

    expect(result.current.status).toBe("complete");
    expect(es.close).toHaveBeenCalled();
  });

  it("sets error status on error event", () => {
    const { result } = renderHook(() => useAgentStream("s1"));
    const es = MockEventSource.instances[0];

    act(() => {
      sendSSE(es, {
        type: "error",
        data: { agent: "system", detail: "Something broke" },
      });
    });

    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("Something broke");
    expect(es.close).toHaveBeenCalled();
  });

  it("handles HITL checkpoint event", () => {
    const { result } = renderHook(() => useAgentStream("s1"));
    const es = MockEventSource.instances[0];

    act(() => {
      sendSSE(es, {
        type: "hitl_checkpoint",
        data: {
          agent: "system",
          summary: "Review claims",
          claims: [{ claim: { statement: "test" }, verdict: "supported" }],
          source_quality: 0.8,
        },
      });
    });

    expect(result.current.status).toBe("hitl");
    expect(result.current.hitlCheckpoint).not.toBeNull();
    expect(result.current.hitlCheckpoint?.summary).toBe("Review claims");
    expect(es.close).toHaveBeenCalled();
  });

  it("sets synth status for synthesizer events", () => {
    const { result } = renderHook(() => useAgentStream("s1"));
    const es = MockEventSource.instances[0];

    act(() => {
      sendSSE(es, {
        type: "agent_started",
        data: { agent: "synthesizer", action: "generating" },
      });
    });

    expect(result.current.status).toBe("synth");
  });

  it("sets error on EventSource connection error", () => {
    const { result } = renderHook(() => useAgentStream("s1"));
    const es = MockEventSource.instances[0];

    act(() => {
      es.onerror?.({} as Event);
    });

    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("Connection to agent stream failed");
    expect(es.close).toHaveBeenCalled();
  });

  it("resets state with reset()", () => {
    const { result } = renderHook(() => useAgentStream("s1"));
    const es = MockEventSource.instances[0];

    act(() => {
      sendSSE(es, { type: "agent_started", data: { agent: "planner" } });
    });
    expect(result.current.events).toHaveLength(1);

    act(() => {
      result.current.reset();
    });

    expect(result.current.events).toHaveLength(0);
    expect(result.current.status).toBe("idle");
    expect(result.current.error).toBeNull();
  });

  it("reconnect() opens a new EventSource connection", () => {
    const { result } = renderHook(() => useAgentStream("s1"));
    expect(MockEventSource.instances).toHaveLength(1);

    act(() => {
      result.current.reconnect();
    });

    // A reconnect increments connectionId, which triggers a new useEffect cycle
    // The old EventSource is closed in cleanup, and a new one is created
    expect(MockEventSource.instances.length).toBeGreaterThanOrEqual(2);
    expect(result.current.status).toBe("running");
  });

  it("closes EventSource on unmount", () => {
    const { unmount } = renderHook(() => useAgentStream("s1"));
    const es = MockEventSource.instances[0];
    unmount();
    expect(es.close).toHaveBeenCalled();
  });

  it("sets isTimedOut after EVENT_TIMEOUT_MS", () => {
    const { result } = renderHook(() => useAgentStream("s1"));

    act(() => {
      vi.advanceTimersByTime(60001);
    });

    expect(result.current.isTimedOut).toBe(true);
  });

  it("normalizes nested data fields to top level", () => {
    const { result } = renderHook(() => useAgentStream("s1"));
    const es = MockEventSource.instances[0];

    act(() => {
      sendSSE(es, {
        type: "tool_result",
        data: {
          agent: "researcher",
          action: "search",
          detail: "Found 5 results",
          latency_ms: 150,
          tokens_used: 200,
          tool: "tavily_search",
        },
        timestamp: "2025-01-01T00:00:01Z",
      });
    });

    const event = result.current.events[0];
    expect(event.agent).toBe("researcher");
    expect(event.action).toBe("search");
    expect(event.detail).toBe("Found 5 results");
    expect(event.latency_ms).toBe(150);
    expect(event.tokens_used).toBe(200);
    expect(event.tool).toBe("tavily_search");
  });
});
