"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { QueryInput } from "@/components/QueryInput";
import { AgentTrace } from "@/components/AgentTrace";
import { Pipeline } from "@/components/Pipeline";
import { Metrics } from "@/components/Metrics";
import { useAgentStream } from "@/hooks/useAgentStream";
import type { AgentName, StreamStatus, TraceEvent } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ActiveTab = "overview" | "report";

export default function HomePage() {
  const [streamId, setStreamId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>("overview");
  const [currentQuery, setCurrentQuery] = useState<string>("");
  const [phase, setPhase] = useState<StreamStatus>("idle");
  const [currentAgent, setCurrentAgent] = useState<AgentName | null>(null);
  const [showReport, setShowReport] = useState(false);
  const traceRef = useRef<HTMLDivElement>(null);

  const { events, status, error, reset } = useAgentStream(streamId);

  useEffect(() => {
    if (events.length > 0) {
      const lastEvent = events[events.length - 1];
      setCurrentAgent(lastEvent.agent as AgentName);
      if (lastEvent.type === "checkpoint") setPhase("hitl");
      else if (lastEvent.type === "complete") {
        setPhase("complete");
        setShowReport(true);
        setActiveTab("report");
      } else if (lastEvent.agent === "synthesizer") setPhase("synth");
    }
  }, [events]);

  useEffect(() => {
    if (status === "running" && phase === "idle") setPhase("running");
    else if (status === "complete") setPhase("complete");
    else if (status === "error") setPhase("error");
  }, [status, phase]);

  const handleSubmit = useCallback(async (query: string) => {
    reset();
    setCurrentQuery(query);
    setActiveTab("overview");
    setPhase("running");
    setShowReport(false);
    setCurrentAgent(null);
    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setStreamId(data.stream_id);
    } catch (err) {
      console.error("Failed to submit query:", err);
      setPhase("error");
    }
  }, [reset]);

  const handlePreviewReport = () => {
    setPhase("complete");
    setCurrentAgent(null);
    setShowReport(true);
    setActiveTab("report");
    setCurrentQuery("Compare Rust vs Go for building distributed systems");
  };

  const pipelineHeight = phase !== "idle" ? 42 : 0;

  return (
    <div className="min-h-screen bg-terminal-black text-terminal-white font-mono">
      <header className="flex items-center justify-between px-5 h-12 border-b border-dashed border-terminal-borderDotted bg-terminal-black sticky top-0 z-50">
        <div className="flex items-center gap-2.5">
          <span className="w-6 h-6 border-[1.5px] border-terminal-white rounded-sm flex items-center justify-center text-[0.65rem] font-semibold">M</span>
          <span className="text-[0.82rem] font-medium text-terminal-white tracking-tight">marsa</span>
          <span className="text-[0.62rem] text-terminal-dim px-1.5 py-0.5 border border-dashed border-terminal-borderDotted rounded-sm">v1.0.0</span>
        </div>
        <button onClick={handlePreviewReport} className="px-2.5 py-1 border border-dashed border-terminal-borderDotted rounded-sm bg-transparent text-terminal-dim text-[0.7rem] font-mono cursor-pointer transition-all duration-200 hover:border-terminal-mid hover:text-terminal-white">preview report</button>
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
          <div className="flex-1 overflow-y-auto px-6 py-4">
            {activeTab === "overview" ? <OverviewContent phase={phase} currentAgent={currentAgent} events={events} /> : <ReportContent showReport={showReport} />}
          </div>
        </div>
      </div>
    </div>
  );
}

function OverviewContent({ phase, currentAgent, events }: { phase: StreamStatus; currentAgent: AgentName | null; events: TraceEvent[] }) {
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
      </div>
    );
  }
  if (phase === "hitl") {
    return (
      <div className="flex flex-col items-center justify-center h-[50vh] animate-fade-in">
        <div className="w-12 h-12 border border-dashed border-terminal-mid rounded-full flex items-center justify-center mb-4 animate-pulse-dot">
          <span className="text-[0.8rem]">||</span>
        </div>
        <div className="font-mono text-[0.82rem] text-terminal-white mb-1">checkpoint: human review</div>
        <div className="font-mono text-[0.72rem] text-terminal-dim text-center max-w-[260px]">review findings in the left panel before synthesis.</div>
      </div>
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
            <div className="font-mono text-[0.68rem] text-terminal-dim">10 sources | switch to <span className="text-terminal-white">report</span> tab</div>
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
  return <div className="flex flex-col items-center justify-center h-[50vh] text-semantic-fail"><span className="font-mono text-sm">{"// error occurred"}</span></div>;
}

function ReportContent({ showReport }: { showReport: boolean }) {
  if (!showReport) {
    return <div className="flex flex-col items-center justify-center h-[60vh] opacity-30"><span className="font-mono text-[0.74rem]">{"// report"}</span><span className="font-mono text-[0.66rem] mt-1">pending synthesis...</span></div>;
  }
  return (
    <div className="animate-slide-in">
      <div className="flex justify-between items-center mb-5 pb-3 border-b border-dashed border-terminal-borderDotted">
        <span className="font-mono text-[0.68rem] text-terminal-dim uppercase tracking-wider">{"// generated report"}</span>
        <button className="px-2.5 py-1 rounded-sm border border-dashed border-terminal-borderDotted bg-transparent text-terminal-dim text-[0.7rem] font-mono cursor-pointer transition-all duration-200 hover:text-semantic-pass">copy</button>
      </div>
      <h2 className="font-sans text-[1.35rem] font-semibold text-terminal-pure leading-tight mb-3.5 tracking-tight">Rust vs Go for Distributed Systems</h2>
      <div className="px-3.5 py-3 rounded-sm border-l-2 border-terminal-dim bg-white/[0.02] font-sans text-[0.82rem] text-terminal-mid leading-relaxed mb-7">Both Rust and Go are strong contenders for distributed systems development. Rust offers superior memory safety and raw performance, while Go provides faster development velocity and a mature concurrency model.</div>
      <div className="font-mono text-[0.68rem] text-terminal-dim uppercase tracking-wider mb-3">{"// report content fully implemented in Day 13"}</div>
    </div>
  );
}
