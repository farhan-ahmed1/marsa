"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { QueryInput } from "@/components/QueryInput";
import { AgentTrace } from "@/components/AgentTrace";
import { Pipeline } from "@/components/Pipeline";
import { Metrics } from "@/components/Metrics";
import { ReportView } from "@/components/ReportView";
import { HITLFeedback } from "@/components/HITLFeedback";
import { ToastProvider, useToast } from "@/components/Toast";
import { useAgentStream } from "@/hooks/useAgentStream";
import { getReport, submitFeedback } from "@/lib/api";
import type { AgentName, StreamStatus, TraceEvent, Report, FeedbackAction, ReportResponse } from "@/lib/types";

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

  const { events, status, error, isTimedOut, hitlCheckpoint, reset } = useAgentStream(streamId);

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
    else if (status === "hitl") setPhase("hitl");
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
        setPhase("running");
      }
    } catch (err) {
      console.error("Failed to submit feedback:", err);
      addToast("Failed to submit feedback. Please try again.", "error");
    } finally {
      setIsSubmittingFeedback(false);
    }
  }, [streamId, addToast, reset]);

  const handlePreviewReport = useCallback(() => {
    // Only show preview if there's no active query/report
    if (phase !== "idle" && phase !== "error") {
      addToast("Cannot preview while a query is running or completed", "warning");
      return;
    }

    // Create mock report data for demo purposes
    const mockReport: Report = {
      title: "Rust vs Go for Distributed Systems",
      summary: "Both Rust and Go are strong contenders for distributed systems development. Rust offers superior memory safety and raw performance [1], while Go provides faster development velocity and a mature concurrency model [2].",
      sections: [
        {
          heading: "Performance Characteristics",
          content: "Rust achieves near-C performance with zero-cost abstractions [1]. The ownership system eliminates garbage collection pauses, making it ideal for latency-sensitive applications. Go, while slightly slower in raw benchmarks, offers excellent performance for network-bound services [3].",
        },
        {
          heading: "Concurrency Model",
          content: "Go's goroutines and channels provide an intuitive concurrency model that scales well [2]. Rust's async/await with tokio offers more control but requires understanding of lifetimes and pinning [4].",
        },
        {
          heading: "Developer Experience",
          content: "Go compiles faster and has a gentler learning curve. Rust's compiler is strict but catches many bugs at compile time [5].",
        },
      ],
      confidence_summary: "High confidence in performance claims based on multiple benchmark sources. Medium confidence in developer experience comparisons due to subjective nature.",
      citations: [
        { number: 1, title: "Rust Performance Guide", url: "https://rust-lang.org/perf", source_quality_score: 0.9, accessed_date: "2026-02-17" },
        { number: 2, title: "Go Concurrency Patterns", url: "https://go.dev/blog/pipelines", source_quality_score: 0.85, accessed_date: "2026-02-17" },
        { number: 3, title: "TechEmpower Benchmarks", url: "https://techempower.com/benchmarks", source_quality_score: 0.8, accessed_date: "2026-02-17" },
        { number: 4, title: "Tokio Tutorial", url: "https://tokio.rs/tutorial", source_quality_score: 0.85, accessed_date: "2026-02-17" },
        { number: 5, title: "Developer Survey 2026", url: "https://stackoverflow.com/survey", source_quality_score: 0.75, accessed_date: "2026-02-17" },
      ],
    };
    
    setReportData({
      stream_id: "preview",
      status: "completed",
      report: mockReport,
      raw_report: null,
      metrics: {
        claims_count: 8,
        verification_count: 8,
        citations_count: 5,
        fact_check_pass_rate: 0.875,
      },
      error: null,
    });
    setPhase("complete");
    setCurrentAgent(null);
    setShowReport(true);
    setActiveTab("report");
    setCurrentQuery("Compare Rust vs Go for building distributed systems");
  }, [phase, addToast]);

  const pipelineHeight = phase !== "idle" ? 42 : 0;

  return (
    <div className="min-h-screen bg-terminal-black text-terminal-white font-mono">
      <header className="flex items-center justify-between px-5 h-12 border-b border-dashed border-terminal-borderDotted bg-terminal-black sticky top-0 z-50">
        <div className="flex items-center gap-2.5">
          <span className="w-6 h-6 border-[1.5px] border-terminal-white rounded-sm flex items-center justify-center text-[0.65rem] font-semibold">M</span>
          <span className="text-[0.82rem] font-medium text-terminal-white tracking-tight">marsa</span>
          <span className="text-[0.62rem] text-terminal-dim px-1.5 py-0.5 border border-dashed border-terminal-borderDotted rounded-sm">v1.0.0</span>
        </div>
        <button 
          onClick={handlePreviewReport} 
          disabled={phase !== "idle" && phase !== "error"}
          className={`px-2.5 py-1 border border-dashed border-terminal-borderDotted rounded-sm bg-transparent text-[0.7rem] font-mono transition-all duration-200 ${
            phase !== "idle" && phase !== "error" 
              ? "text-terminal-borderDotted cursor-not-allowed opacity-40" 
              : "text-terminal-dim cursor-pointer hover:border-terminal-mid hover:text-terminal-white"
          }`}
          title={phase !== "idle" && phase !== "error" ? "Preview only available when idle" : "Show demo report"}
        >
          preview report
        </button>
      </header>
      <Pipeline currentAgent={currentAgent} status={phase} />
      <div className="flex max-w-[1280px] w-full mx-auto">
        <div className="flex-shrink-0 w-[420px] border-r border-dashed border-terminal-borderDotted flex flex-col" style={{ height: `calc(100vh - 48px - ${pipelineHeight}px)` }}>
          <QueryInput onSubmit={handleSubmit} status={phase} />
          <div ref={traceRef} className="flex-1 overflow-y-auto px-4 py-3.5">
            <AgentTrace events={events} status={phase} />
          </div>
        </div>
        <div className="flex-1 flex flex-col" style={{ height: `calc(100vh - 48px - ${pipelineHeight}px)` }}>
          <div className="flex border-b border-dashed border-terminal-borderDotted px-5">
            <button onClick={() => setActiveTab("overview")} className={`px-3.5 py-3 bg-transparent border-none text-[0.74rem] font-mono cursor-pointer transition-all duration-200 ${activeTab === "overview" ? "text-terminal-white border-b border-terminal-white" : "text-terminal-dim border-b border-transparent"}`}>overview</button>
            <button onClick={() => setActiveTab("report")} className={`px-3.5 py-3 bg-transparent border-none text-[0.74rem] font-mono cursor-pointer transition-all duration-200 flex items-center gap-1.5 ${activeTab === "report" ? "text-terminal-white border-b border-terminal-white" : "text-terminal-dim border-b border-transparent"}`}>report{showReport && <span className="w-1.5 h-1.5 rounded-full bg-semantic-pass" />}</button>
          </div>
          <div className="flex-1 overflow-y-auto px-6 py-4 scrollbar-terminal">
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
              />
            ) : (
              <ReportView 
                report={reportData?.report || null} 
                rawReport={reportData?.raw_report}
                isLoading={isLoadingReport}
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
}: OverviewContentProps) {
  if (phase === "idle") {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-center animate-fade-in">
        <div className="border border-dashed border-terminal-borderDotted rounded-full w-[72px] h-[72px] flex items-center justify-center mb-4">
          <span className="font-mono text-[1.3rem] text-terminal-dim font-light">M</span>
        </div>
        <div className="font-sans text-[1.15rem] font-semibold text-terminal-white mb-1.5 tracking-tight">multi-agent research</div>
        <div className="font-mono text-[0.76rem] text-terminal-dim max-w-[340px] leading-relaxed">specialized agents collaborate to produce verified, well-sourced reports from complex queries.</div>
        <div className="flex gap-0 mt-6 border border-dashed border-terminal-borderDotted rounded-sm overflow-hidden">
          {["plan", "research", "verify", "synthesize"].map((s, i) => (
            <div key={s} className={`px-3.5 py-2 font-mono text-[0.68rem] text-terminal-dim animate-slide-in ${i < 3 ? "border-r border-dashed border-terminal-borderDotted" : ""}`} style={{ animationDelay: `${0.15 + i * 0.07}s` }}>
              <span className="text-terminal-mid mr-1.5">{i + 1}.</span>{s}
            </div>
          ))}
        </div>
      </div>
    );
  }
  
  if (phase === "running" || phase === "synth") {
    const totalTokens = events.reduce((a, e) => a + (e.tokens_used || 0), 0);
    return (
      <div className="flex flex-col items-center justify-center h-[50vh] animate-fade-in">
        <div className="w-12 h-12 border border-dashed border-terminal-mid rounded-full flex items-center justify-center mb-4 animate-spin-slow">
          <span className="w-1.5 h-1.5 rounded-full bg-terminal-white shadow-[0_0_6px_rgba(255,255,255,0.3)]" />
        </div>
        <div className="font-mono text-[0.82rem] text-terminal-white mb-1">
          {currentAgent === "planner" && "planning..."}
          {currentAgent === "researcher" && "researching..."}
          {currentAgent === "fact_checker" && "verifying..."}
          {currentAgent === "synthesizer" && "synthesizing..."}
          {currentAgent === "system" && "awaiting review..."}
          {!currentAgent && "initializing..."}
        </div>
        <div className="font-mono text-[0.68rem] text-terminal-dim">{events.length} events | {totalTokens.toLocaleString()} tokens</div>
        {isTimedOut && (
          <div className="mt-4 px-3 py-2 border border-dashed border-semantic-unknown/40 rounded-sm bg-semantic-unknown/10 animate-fade-in">
            <span className="font-mono text-[0.7rem] text-semantic-unknown">taking longer than expected...</span>
          </div>
        )}
      </div>
    );
  }
  
  if (phase === "hitl") {
    // Show the HITL feedback interface
    return (
      <HITLFeedback
        streamId={streamId || ""}
        summary={hitlCheckpoint?.summary || "Research phase complete. Please review the findings."}
        claims={hitlCheckpoint?.claims || []}
        sourceQuality={hitlCheckpoint?.sourceQuality || 0}
        onSubmitFeedback={onFeedback}
        isSubmitting={isSubmittingFeedback}
      />
    );
  }
  
  if (phase === "complete") {
    return (
      <div className="animate-slide-in">
        <Metrics show={true} />
        <div className="flex items-center gap-2.5 px-3.5 py-3 border border-semantic-pass/30 rounded-sm mb-5 bg-semantic-pass/5">
          <span className="text-semantic-pass text-[0.82rem]">OK</span>
          <div>
            <div className="font-mono text-[0.78rem] text-terminal-white">research complete</div>
            <div className="font-mono text-[0.68rem] text-terminal-dim">switch to <span className="text-terminal-white">report</span> tab to view results</div>
          </div>
        </div>
        <div className="font-mono text-[0.62rem] text-terminal-dim uppercase tracking-wider mb-3">{"// execution_timeline"}</div>
        {[{ a: "planner", d: "1.4s", t: 560, w: 10 }, { a: "researcher", d: "5.8s", t: 1847, w: 42 }, { a: "fact_checker", d: "3.2s", t: 2348, w: 23 }, { a: "synthesizer", d: "3.8s", t: 3653, w: 27 }].map((item, i) => (
          <div key={i} className="mb-2.5 animate-slide-in" style={{ animationDelay: `${0.08 + i * 0.07}s` }}>
            <div className="flex justify-between mb-1">
              <span className="font-mono text-[0.72rem] text-terminal-white">{item.a}</span>
              <span className="font-mono text-[0.62rem] text-terminal-dim">{item.d} | {item.t.toLocaleString()} tok</span>
            </div>
            <div className="h-1 rounded-sm bg-terminal-surface border border-terminal-border overflow-hidden">
              <div className="h-full rounded-sm bg-terminal-dim animate-grow" style={{ width: `${item.w}%`, animationDelay: `${0.2 + i * 0.12}s` }} />
            </div>
          </div>
        ))}
      </div>
    );
  }
  
  if (phase === "error") {
    return (
      <div className="flex flex-col items-center justify-center h-[50vh] animate-fade-in">
        <div className="w-12 h-12 border border-dashed border-semantic-fail rounded-full flex items-center justify-center mb-4">
          <span className="text-semantic-fail text-[1rem]">!!</span>
        </div>
        <div className="font-mono text-[0.82rem] text-semantic-fail mb-1">{"error occurred"}</div>
        <div className="font-mono text-[0.68rem] text-terminal-dim text-center max-w-[300px]">
          An error occurred during processing. Check the console for details or try again.
        </div>
      </div>
    );
  }
  
  return null;
}
