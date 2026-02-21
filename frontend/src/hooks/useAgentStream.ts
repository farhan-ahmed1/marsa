"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { TraceEvent, StreamStatus, VerificationResult, Report } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Timeout warning after 60 seconds of no events
const EVENT_TIMEOUT_MS = 60000;

let eventIdCounter = 0;

/**
 * Normalize an SSE payload into a flat TraceEvent.
 *
 * The backend sends events shaped as { type, data: { agent, action, detail, ... }, timestamp }.
 * The frontend TraceEvent expects those fields at the top level. This function
 * hoists nested data fields so downstream code can always read event.agent, etc.
 */
function normalizeEvent(raw: Record<string, any>): TraceEvent {
  const data = raw.data && typeof raw.data === "object" ? raw.data : {};
  return {
    id: raw.id || `evt-${++eventIdCounter}`,
    type: raw.type || data.type || "tool_result",
    agent: raw.agent || data.agent || "system",
    action: raw.action || data.action || "",
    detail: raw.detail || data.detail || "",
    timestamp: raw.timestamp || new Date().toISOString(),
    latency_ms: raw.latency_ms ?? data.latency_ms,
    tokens_used: raw.tokens_used ?? data.tokens_used,
    tool: raw.tool || data.tool,
    data: data,
  };
}

interface HITLCheckpointData {
  summary: string;
  claims: VerificationResult[];
  sourceQuality: number;
}

interface UseAgentStreamReturn {
  events: TraceEvent[];
  status: StreamStatus;
  error: string | null;
  isTimedOut: boolean;
  hitlCheckpoint: HITLCheckpointData | null;
  report: Report | null;
  reset: () => void;
  reconnect: () => void;
}

export function useAgentStream(streamId: string | null): UseAgentStreamReturn {
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [isTimedOut, setIsTimedOut] = useState(false);
  const [hitlCheckpoint, setHitlCheckpoint] = useState<HITLCheckpointData | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  // Incrementing connectionId forces the SSE useEffect to re-run and open
  // a new EventSource even when streamId has not changed (e.g. after HITL feedback).
  const [connectionId, setConnectionId] = useState(0);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastEventTimeRef = useRef<number>(Date.now());

  const reset = useCallback(() => {
    setEvents([]);
    setStatus("idle");
    setError(null);
    setIsTimedOut(false);
    setHitlCheckpoint(null);
    setReport(null);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  /**
   * Reconnect to the SSE stream for the same streamId.
   * Used after HITL feedback submission so the frontend can
   * receive events from the resumed workflow.
   */
  const reconnect = useCallback(() => {
    setStatus("running");
    setError(null);
    setIsTimedOut(false);
    setHitlCheckpoint(null);
    setConnectionId((prev) => prev + 1);
  }, []);

  // Reset timeout on new events
  const resetTimeout = useCallback(() => {
    lastEventTimeRef.current = Date.now();
    setIsTimedOut(false);
    
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    timeoutRef.current = setTimeout(() => {
      setIsTimedOut(true);
    }, EVENT_TIMEOUT_MS);
  }, []);

  useEffect(() => {
    if (!streamId) {
      return;
    }

    setStatus("running");
    setError(null);
    setIsTimedOut(false);
    resetTimeout();

    const eventSource = new EventSource(
      `${API_BASE}/api/query/${streamId}/stream`
    );

    eventSource.onopen = () => {
      setStatus("running");
      resetTimeout();
    };

    eventSource.onmessage = (e) => {
      try {
        const raw = JSON.parse(e.data);
        const event = normalizeEvent(raw);
        
        // Skip heartbeat events - they're just keep-alive signals
        if (event.type === "heartbeat" || event.type === "connected") {
          resetTimeout();
          return;
        }
        
        setEvents((prev) => [...prev, event]);
        resetTimeout();

        // Handle different event types
        if (event.type === "complete") {
          setStatus("complete");
          eventSource.close();
          if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
          }
        } else if (event.type === "error") {
          setStatus("error");
          setError(event.detail || "Unknown error occurred");
          eventSource.close();
          if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
          }
        } else if (event.type === "checkpoint" || event.type === "hitl_checkpoint") {
          setStatus("hitl");
          // Close the connection - workflow is paused for feedback
          eventSource.close();
          if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
          }
          // Parse HITL checkpoint data from event
          try {
            const checkpointData = event.data || (event.detail ? JSON.parse(event.detail) : null);
            if (checkpointData) {
              setHitlCheckpoint({
                summary: checkpointData.summary || checkpointData.message || "",
                claims: checkpointData.claims || [],
                sourceQuality: checkpointData.source_quality || 0,
              });
            }
          } catch {
            // Checkpoint detail might not be JSON, that's okay
            setHitlCheckpoint({
              summary: event.detail || "Review required",
              claims: [],
              sourceQuality: 0,
            });
          }
        } else if (event.agent === "synthesizer") {
          setStatus("synth");
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
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };

    return () => {
      eventSource.close();
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [streamId, connectionId, resetTimeout]);

  return { events, status, error, isTimedOut, hitlCheckpoint, report, reset, reconnect };
}
