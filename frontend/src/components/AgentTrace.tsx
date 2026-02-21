"use client";

import { useRef, useEffect } from "react";
import type { TraceEvent, StreamStatus } from "@/lib/types";

interface AgentTraceProps {
  events: TraceEvent[];
  status: StreamStatus;
}

// Agent color configuration
const AGENT_CONFIG: Record<string, { label: string; dot: string; badge: string }> = {
  planner:      { label: "Planner",      dot: "bg-blue-400",    badge: "text-blue-400 bg-blue-400/10 border-blue-400/30" },
  researcher:   { label: "Researcher",   dot: "bg-emerald-400", badge: "text-emerald-400 bg-emerald-400/10 border-emerald-400/30" },
  fact_checker: { label: "Fact Checker", dot: "bg-amber-400",   badge: "text-amber-400 bg-amber-400/10 border-amber-400/30" },
  synthesizer:  { label: "Synthesizer",  dot: "bg-violet-400",  badge: "text-violet-400 bg-violet-400/10 border-violet-400/30" },
  system:       { label: "System",       dot: "bg-slate-400",   badge: "text-slate-400 bg-slate-400/10 border-slate-400/30" },
};

function getAgentConfig(agent: string) {
  return AGENT_CONFIG[agent] ?? { label: agent, dot: "bg-terminal-dim", badge: "text-terminal-mid bg-terminal-surface border-terminal-border" };
}

// Dot connector
function TimelineDot({ active, last, color }: { active: boolean; last: boolean; color: string }) {
  return (
    <div className="flex flex-col items-center flex-shrink-0">
      <span className={`w-2 h-2 rounded-full mt-1 transition-all duration-300 ${active ? `${color} shadow-sm` : "bg-terminal-border"}`} />
      {!last && <span className="w-px flex-1 min-h-[20px] bg-terminal-border mt-1" />}
    </div>
  );
}

function TraceRow({ event, isLast, isActive }: { event: TraceEvent; isLast: boolean; isActive: boolean }) {
  const config = getAgentConfig(event.agent);

  const formatTime = (ts: string) => {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  };

  return (
    <div className="flex gap-3 animate-slide-in">
      <TimelineDot active={isActive} last={isLast} color={config.dot} />
      <div className="flex-1 pb-4 min-w-0">
        {/* Header row */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium border ${config.badge}`}>
            {config.label}
          </span>
          <span className="text-terminal-mid text-xs">→</span>
          <span className="text-terminal-white text-sm font-medium font-mono">{event.action}</span>
          {event.tool && (
            <span className="text-terminal-dim text-xs font-mono">({event.tool})</span>
          )}
          {isActive && (
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
          )}
        </div>

        {/* Detail text */}
        {event.detail && (
          <p className="text-terminal-mid text-xs mt-1 leading-relaxed line-clamp-2 font-mono">
            {event.detail}
          </p>
        )}

        {/* Metadata row */}
        <div className="flex items-center gap-3 mt-1.5 text-terminal-dim text-xs font-mono">
          <span>{formatTime(event.timestamp)}</span>
          {event.latency_ms !== undefined && (
            <span>{event.latency_ms < 1000 ? `${event.latency_ms}ms` : `${(event.latency_ms / 1000).toFixed(2)}s`}</span>
          )}
          {event.tokens_used !== undefined && (
            <span>{event.tokens_used.toLocaleString()} tok</span>
          )}
        </div>
      </div>
    </div>
  );
}

export function AgentTrace({ events, status }: AgentTraceProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [events]);

  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-terminal-dim text-sm">
        No activity yet
      </div>
    );
  }

  const isRunning = status === "running" || status === "synth";

  return (
    <div ref={containerRef} className="px-4 py-4 overflow-y-auto h-full">
      <p className="text-xs font-medium text-terminal-dim uppercase tracking-wider mb-4">
        Agent Activity
      </p>
      {events.map((event, i) => (
        <TraceRow
          key={event.id || i}
          event={event}
          isLast={i === events.length - 1}
          isActive={isRunning && i === events.length - 1}
        />
      ))}
    </div>
  );
}
