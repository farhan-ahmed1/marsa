"use client";

import { useState, useEffect, useCallback } from "react";
import type { TraceEvent, StreamStatus } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface UseAgentStreamReturn {
  events: TraceEvent[];
  status: StreamStatus;
  error: string | null;
  reset: () => void;
}

export function useAgentStream(streamId: string | null): UseAgentStreamReturn {
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setEvents([]);
    setStatus("idle");
    setError(null);
  }, []);

  useEffect(() => {
    if (!streamId) {
      return;
    }

    setStatus("running");
    setError(null);

    const eventSource = new EventSource(
      `${API_BASE}/api/query/${streamId}/stream`
    );

    eventSource.onopen = () => {
      setStatus("running");
    };

    eventSource.onmessage = (e) => {
      try {
        const event: TraceEvent = JSON.parse(e.data);
        setEvents((prev) => [...prev, event]);

        if (event.type === "complete") {
          setStatus("complete");
          eventSource.close();
        } else if (event.type === "error") {
          setStatus("error");
          setError(event.detail || "Unknown error occurred");
          eventSource.close();
        }
      } catch (parseError) {
        console.error("Failed to parse SSE event:", parseError);
      }
    };

    eventSource.onerror = (e) => {
      console.error("SSE connection error:", e);
      setStatus("error");
      setError("Connection to agent stream failed");
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [streamId]);

  return { events, status, error, reset };
}
