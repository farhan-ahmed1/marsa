"use client";

import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { QueryInput } from "@/components/QueryInput";
import { AgentTrace } from "@/components/AgentTrace";
import { Pipeline } from "@/components/Pipeline";
import { ReportView } from "@/components/ReportView";
import { HITLFeedback } from "@/components/HITLFeedback";
import { Timeline } from "@/components/Timeline";
import { Metrics } from "@/components/Metrics";
import { ToastProvider, useToast } from "@/components/Toast";
import { useAgentStream } from "@/hooks/useAgentStream";
import { getReport, submitFeedback } from "@/lib/api";
import { computeQueryMetrics, metricsToItems } from "@/lib/utils";
import type { AgentName, StreamStatus, TraceEvent, FeedbackAction, ReportResponse } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ActiveTab = "overview" | "report";

function HomePageContent() {
  const [streamId, setStreamId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>("overview");
  const [currentQuery, setCurrentQuery] = useState<string>("");
  const [phase, setPhase] = useState<StreamStatus>("idle");
  const [currentAgent, setCurrentAgent] = useState<AgentName | null>(null);
  const [showReport, setShowReport] = useState(false);
  const [reportData, setReportData] = useState<ReportResponse | null>(null);
  const [isLoadingReport, setIsLoadingReport] = useState(false);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const traceRef = useRef<HTMLDivElement>(null);
  const { addToast } = useToast();

  const { events, status, error, isTimedOut, hitlCheckpoint, reset, reconnect } = useAgentStream(streamId);

  const fetchReport = useCallback(async (sid: string) => {
    setIsLoadingReport(true);
    try {
      const data = await getReport(sid, { maxRetries: 20, retryDelay: 500 });
      setReportData(data);
    } catch (err) {
      console.error("Failed to fetch report:", err);
      addToast("Failed to fetch report. Please try again.", "error");
    } finally {
      setIsLoadingReport(false);
    }
  }, [addToast]);

  // Show error toasts
  useEffect(() => {
    if (error) {
      addToast(error, "error");
    }
  }, [error, addToast]);

  // Show timeout warning
  useEffect(() => {
    if (isTimedOut && (phase === "running" || phase === "synth")) {
      addToast("Taking longer than expected. The agents are still working...", "warning", 10000);
    }
  }, [isTimedOut, phase, addToast]);

  // Track events and update state
  useEffect(() => {
    if (events.length > 0) {
      const lastEvent = events[events.length - 1];
      setCurrentAgent(lastEvent.agent as AgentName);
      if (lastEvent.type === "checkpoint" || lastEvent.type === "hitl_checkpoint") {
        setPhase("hitl");
      } else if (lastEvent.type === "complete") {
        // Workflow-level complete event (has report data)
        const eventData = lastEvent.data as any;
        console.log("=== WORKFLOW COMPLETE EVENT ===");
        console.log("Event data:", JSON.stringify(eventData, null, 2));
        console.log("Has report?", !!eventData?.report);
        
        setPhase("complete");
        setShowReport(true);
        setActiveTab("report");
        
        if (eventData && eventData.report) {
          console.log("Setting report from event");
          setReportData({
            stream_id: streamId || "",
            status: eventData.status || "completed",
            report: eventData.report,
            raw_report: eventData.raw_report,
            metrics: eventData.metrics || {},
            error: null,
          });
        } else {
          console.log("No report in event, falling back to fetch");
          if (streamId) {
            fetchReport(streamId);
          }
        }
      } else if (lastEvent.type === "agent_completed") {
        // Individual agent finished - not the workflow completion
        console.log(`Agent ${lastEvent.agent} completed: ${lastEvent.data?.detail || lastEvent.detail}`);
      } else if (lastEvent.agent === "synthesizer") {
        setPhase("synth");
      }
    }
  }, [events, streamId, fetchReport]);

  useEffect(() => {
    if (status === "running" && phase === "idle") setPhase("running");
    else if (status === "complete") setPhase("complete");
    else if (status === "error") setPhase("error");
    // Only transition to hitl if we are NOT already running (e.g. after
    // submitting feedback and reconnecting).  Without this guard the
    // stale "hitl" status from the hook would immediately revert
    // the phase back before the reconnected stream delivers new events.
    else if (status === "hitl" && phase !== "running" && phase !== "synth") setPhase("hitl");
  }, [status, phase]);

  const handleSubmit = useCallback(async (query: string) => {
    reset();
    setCurrentQuery(query);
    setActiveTab("overview");
    setPhase("running");
    setShowReport(false);
    setCurrentAgent(null);
    setReportData(null);
    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, enable_hitl: true }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setStreamId(data.stream_id);
    } catch (err) {
      console.error("Failed to submit query:", err);
      setPhase("error");
      addToast("Failed to submit query. Please check your connection.", "error");
    }
  }, [reset, addToast]);

  const handleFeedback = useCallback(async (action: FeedbackAction, detail?: string) => {
    if (!streamId) return;
    
    setIsSubmittingFeedback(true);
    try {
      await submitFeedback(streamId, action, detail);
      addToast(
        action === "approve" 
          ? "Continuing to synthesis..." 
          : action === "abort" 
          ? "Research aborted" 
          : "Feedback submitted, continuing research...",
        action === "abort" ? "warning" : "success"
      );
      
      if (action === "abort") {
        setPhase("idle");
        reset();
      } else {
        // Reconnect to the SSE stream so we receive events from the
        // resumed workflow. This resets the hook status to "running"
        // and opens a fresh EventSource to the same stream endpoint.
        setPhase("running");
        reconnect();
      }
    } catch (err) {
      console.error("Failed to submit feedback:", err);
      addToast("Failed to submit feedback. Please try again.", "error");
    } finally {
      setIsSubmittingFeedback(false);
    }
  }, [streamId, addToast, reset, reconnect]);

  const pipelineVisible = phase !== "idle";

  return (
    <div className="min-h-screen bg-terminal-black text-terminal-white">
      {/* Header */}
      <header className="flex items-center justify-between px-6 h-14 border-b border-terminal-border bg-terminal-black sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 bg-accent rounded-md flex items-center justify-center text-white text-sm font-bold">M</div>
          <span className="text-base font-semibold text-terminal-pure tracking-tight">MARSA</span>
          <span className="text-xs text-terminal-dim bg-terminal-surface px-2 py-0.5 rounded border border-terminal-border">v1.0</span>
        </div>
      </header>

      {/* Pipeline progress bar */}
      <Pipeline currentAgent={currentAgent} status={phase} />

      {/* Main content */}
      <div className="flex" style={{ height: `calc(100vh - ${pipelineVisible ? 112 : 56}px)` }}>
        {/* Left panel: Query + Trace */}
        <div className="w-[400px] flex-shrink-0 border-r border-terminal-border flex flex-col bg-terminal-surface/30">
          <QueryInput onSubmit={handleSubmit} status={phase} />
          <div ref={traceRef} className="flex-1 overflow-y-auto">
            <AgentTrace events={events} status={phase} />
          </div>
        </div>

        {/* Right panel: Tabs + Content */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Tab bar */}
          <div className="flex items-center gap-1 px-6 border-b border-terminal-border bg-terminal-black">
            <button
              onClick={() => setActiveTab("overview")}
              className={`px-4 py-3.5 text-sm font-medium transition-all border-b-2 -mb-px ${
                activeTab === "overview"
                  ? "text-terminal-pure border-accent"
                  : "text-terminal-dim border-transparent hover:text-terminal-white"
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab("report")}
              className={`px-4 py-3.5 text-sm font-medium transition-all border-b-2 -mb-px flex items-center gap-2 ${
                activeTab === "report"
                  ? "text-terminal-pure border-accent"
                  : "text-terminal-dim border-transparent hover:text-terminal-white"
              }`}
            >
              Report
              {showReport && (
                <span className="w-2 h-2 rounded-full bg-semantic-pass" />
              )}
            </button>
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-y-auto">
            {activeTab === "overview" ? (
              <OverviewContent
                phase={phase}
                currentAgent={currentAgent}
                events={events}
                isTimedOut={isTimedOut}
                hitlCheckpoint={hitlCheckpoint}
                streamId={streamId}
                onFeedback={handleFeedback}
                isSubmittingFeedback={isSubmittingFeedback}
                reportData={reportData}
              />
            ) : (
              <ReportView
                report={reportData?.report || null}
                rawReport={reportData?.raw_report}
                isLoading={isLoadingReport}
                metrics={
                  reportData
                    ? metricsToItems(
                        computeQueryMetrics(events, reportData.report ?? null, reportData.metrics)
                      )
                    : undefined
                }
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function HomePage() {
  return (
    <ToastProvider>
      <HomePageContent />
    </ToastProvider>
  );
}

interface OverviewContentProps {
  phase: StreamStatus;
  currentAgent: AgentName | null;
  events: TraceEvent[];
  isTimedOut: boolean;
  hitlCheckpoint: { summary: string; claims: import("@/lib/types").VerificationResult[]; sourceQuality: number } | null;
  streamId: string | null;
  onFeedback: (action: FeedbackAction, detail?: string) => Promise<void>;
  isSubmittingFeedback: boolean;
  reportData?: ReportResponse | null;
}

function OverviewContent({ 
  phase, 
  currentAgent, 
  events, 
  isTimedOut,
  hitlCheckpoint,
  streamId,
  onFeedback,
  isSubmittingFeedback,
  reportData,
}: OverviewContentProps) {
  // --- Idle ---
  if (phase === "idle") {
    return (
      <div className="flex flex-col items-center justify-center h-full px-8 py-16 animate-fade-in">
        <div className="w-16 h-16 rounded-2xl bg-accent/10 border border-accent/20 flex items-center justify-center mb-6">
          <span className="text-accent text-2xl font-bold">M</span>
        </div>
        <h1 className="text-2xl font-semibold text-terminal-pure tracking-tight mb-2">
          Multi-Agent Research
        </h1>
        <p className="text-terminal-mid text-sm text-center max-w-sm leading-relaxed mb-8">
          Specialized agents collaborate to produce verified, well-sourced reports from complex queries.
        </p>
        <div className="grid grid-cols-2 gap-3 w-full max-w-sm">
          {[
            { step: "Plan", desc: "Decomposes your query", color: "text-blue-400", num: "1" },
            { step: "Research", desc: "Searches web & docs", color: "text-emerald-400", num: "2" },
            { step: "Verify", desc: "Fact-checks all claims", color: "text-amber-400", num: "3" },
            { step: "Synthesize", desc: "Writes the final report", color: "text-violet-400", num: "4" },
          ].map((item) => (
            <div key={item.step} className="flex items-start gap-3 p-3 rounded-lg bg-terminal-surface border border-terminal-border">
              <span className={`text-xs font-mono font-semibold mt-0.5 ${item.color}`}>{item.num}</span>
              <div>
                <div className={`text-sm font-medium ${item.color}`}>{item.step}</div>
                <div className="text-xs text-terminal-dim mt-0.5">{item.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // --- Running / Synthesizing ---
  if (phase === "running" || phase === "synth") {
    const totalTokens = events.reduce((a, e) => a + (e.tokens_used || 0), 0);
    const agentLabel: Record<string, string> = {
      planner: "Planning your query...",
      researcher: "Researching sources...",
      fact_checker: "Verifying claims...",
      synthesizer: "Writing report...",
      system: "Waiting for review...",
    };
    const agentColor: Record<string, string> = {
      planner: "text-blue-400",
      researcher: "text-emerald-400",
      fact_checker: "text-amber-400",
      synthesizer: "text-violet-400",
      system: "text-slate-400",
    };

    return (
      <div className="flex flex-col items-center justify-center h-full px-8 animate-fade-in">
        {/* Spinner */}
        <div className="relative w-14 h-14 mb-6">
          <div className="w-14 h-14 rounded-full border-2 border-terminal-border" />
          <div className="absolute inset-0 w-14 h-14 rounded-full border-2 border-transparent border-t-accent animate-spin" />
        </div>

        <p className={`text-base font-medium mb-1 ${currentAgent ? agentColor[currentAgent] ?? "text-terminal-white" : "text-terminal-white"}`}>
          {currentAgent ? agentLabel[currentAgent] ?? "Processing..." : "Initializing..."}
        </p>
        <p className="text-sm text-terminal-dim">
          {events.length} events &nbsp;·&nbsp; {totalTokens.toLocaleString()} tokens used
        </p>

        {isTimedOut && (
          <div className="mt-6 px-4 py-3 rounded-lg border border-semantic-unknownBorder bg-semantic-unknownSubtle animate-fade-in">
            <p className="text-sm text-semantic-unknown">Taking longer than expected. Still working...</p>
          </div>
        )}
      </div>
    );
  }

  // --- HITL checkpoint ---
  if (phase === "hitl") {
    return (
      <div className="p-6 animate-fade-in">
        <HITLFeedback
          streamId={streamId || ""}
          summary={hitlCheckpoint?.summary || "Research phase complete. Please review the findings."}
          claims={hitlCheckpoint?.claims || []}
          sourceQuality={hitlCheckpoint?.sourceQuality || 0}
          onSubmitFeedback={onFeedback}
          isSubmitting={isSubmittingFeedback}
        />
      </div>
    );
  }

  // --- Complete ---
  if (phase === "complete") {
    // Compute real agent stats from events
    const AGENT_ORDER_MAP: Record<string, { name: string; color: string; barColor: string }> = {
      planner:      { name: "Planner",      color: "text-blue-400",   barColor: "bg-blue-400" },
      researcher:   { name: "Researcher",   color: "text-emerald-400", barColor: "bg-emerald-400" },
      fact_checker: { name: "Fact Checker", color: "text-amber-400",  barColor: "bg-amber-400" },
      synthesizer:  { name: "Synthesizer",  color: "text-violet-400", barColor: "bg-violet-400" },
    };

    // Compute per-agent stats from real trace events
    const byAgent: Record<string, TraceEvent[]> = {};
    for (const e of events) {
      if (!byAgent[e.agent]) byAgent[e.agent] = [];
      byAgent[e.agent].push(e);
    }

    const firstTs = events.length > 0 ? new Date(events[0].timestamp).getTime() : 0;
    const lastEv = events[events.length - 1];
    const totalElapsedMs = events.length > 0
      ? new Date(lastEv.timestamp).getTime() - firstTs + (lastEv.latency_ms ?? 0)
      : 0;

    const agentStats = (["planner", "researcher", "fact_checker", "synthesizer"] as const)
      .filter((agent) => byAgent[agent])
      .map((agent) => {
        const evs = byAgent[agent];
        const spanMs =
          new Date(evs[evs.length - 1].timestamp).getTime() -
          new Date(evs[0].timestamp).getTime() +
          (evs[evs.length - 1].latency_ms ?? 300);
        const tokens = evs.reduce((a, e) => a + (e.tokens_used ?? 0), 0);
        const pct = totalElapsedMs > 0 ? Math.round((spanMs / totalElapsedMs) * 100) : 0;
        const { name, color, barColor } = AGENT_ORDER_MAP[agent];
        const time = spanMs < 1000 ? `${Math.round(spanMs)}ms` : `${(spanMs / 1000).toFixed(1)}s`;
        return { name, time, tokens, color, barColor, pct: Math.max(pct, 4) };
      });

    // Compute real metrics
    const queryMetrics = computeQueryMetrics(events, reportData?.report ?? null, reportData?.metrics);
    const metricItems = metricsToItems(queryMetrics);

    return (
      <div className="p-6 animate-fade-in space-y-6">
        {/* Success banner */}
        <div className="flex items-center gap-3 p-4 rounded-lg bg-semantic-passSubtle border border-semantic-passBorder">
          <div className="w-8 h-8 rounded-full bg-semantic-passSubtle border border-semantic-passBorder flex items-center justify-center flex-shrink-0">
            <svg className="w-4 h-4 text-semantic-pass" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-semantic-pass">Research complete</p>
            <p className="text-xs text-terminal-mid mt-0.5">Switch to the <span className="text-terminal-white font-medium">Report</span> tab to view results</p>
          </div>
        </div>

        {/* Query metrics strip */}
        <Metrics show metrics={metricItems} />

        {/* Gantt timeline */}
        <div>
          <h3 className="text-sm font-semibold text-terminal-white mb-3">Execution Timeline</h3>
          <Timeline events={events} />
        </div>

        {/* Per-agent execution breakdown */}
        <div>
          <h3 className="text-sm font-semibold text-terminal-white mb-3">Agent Breakdown</h3>
          <div className="space-y-3.5">
            {agentStats.map((item, i) => (
              <div key={i} className="animate-slide-in" style={{ animationDelay: `${i * 0.06}s` }}>
                <div className="flex justify-between items-center mb-1.5">
                  <span className={`text-sm font-medium ${item.color}`}>{item.name}</span>
                  <div className="flex items-center gap-3 text-xs text-terminal-dim font-mono">
                    <span>{item.time}</span>
                    <span>{item.tokens.toLocaleString()} tok</span>
                    <span className="text-terminal-dim/60">{item.pct}%</span>
                  </div>
                </div>
                <div className="h-1.5 rounded-full bg-terminal-surface overflow-hidden">
                  <div
                    className={`h-full rounded-full ${item.barColor} transition-all duration-700`}
                    style={{ width: `${item.pct}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // --- Error ---
  if (phase === "error") {
    return (
      <div className="flex flex-col items-center justify-center h-full px-8 animate-fade-in">
        <div className="w-14 h-14 rounded-full bg-semantic-failSubtle border border-semantic-failBorder flex items-center justify-center mb-6">
          <svg className="w-6 h-6 text-semantic-fail" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
        <p className="text-base font-medium text-semantic-fail mb-2">Something went wrong</p>
        <p className="text-sm text-terminal-dim text-center max-w-xs">
          An error occurred during processing. Check the console for details or try again.
        </p>
      </div>
    );
  }

  return null;
}
