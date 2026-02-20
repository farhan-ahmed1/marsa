"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { TraceEvent, StreamStatus, VerificationResult, Report } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Timeout warning after 60 seconds of no events
const EVENT_TIMEOUT_MS = 60000;

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
}

export function useAgentStream(streamId: string | null): UseAgentStreamReturn {
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [isTimedOut, setIsTimedOut] = useState(false);
  const [hitlCheckpoint, setHitlCheckpoint] = useState<HITLCheckpointData | null>(null);
  const [report, setReport] = useState<Report | null>(null);
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
        const event: TraceEvent = JSON.parse(e.data);
        
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
          // Try to parse report from event metadata
          if (event.detail && typeof event.detail === "object") {
            // Report might be in metadata
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
  }, [streamId, resetTimeout]);

  return { events, status, error, isTimedOut, hitlCheckpoint, report, reset };
}
