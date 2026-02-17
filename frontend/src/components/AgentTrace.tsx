"use client";

import { useRef, useEffect } from "react";
import type { TraceEvent, StreamStatus } from "@/lib/types";

interface AgentTraceProps {
  events: TraceEvent[];
  status: StreamStatus;
}

// Blinking cursor component
function Cursor() {
  return (
    <span className="inline-block w-[7px] h-[14px] bg-terminal-white ml-0.5 align-middle animate-blink" />
  );
}

// Dot indicator for timeline
function Dot({ on = false, last = false }: { on?: boolean; last?: boolean }) {
  return (
    <div className="flex flex-col items-center">
      <span
        className={`w-1.5 h-1.5 rounded-full transition-all duration-300 ${
          on ? "bg-terminal-white shadow-[0_0_6px_rgba(255,255,255,0.3)]" : "bg-terminal-dim"
        }`}
      />
      {!last && <span className="w-px flex-1 min-h-[16px] bg-terminal-border" />}
    </div>
  );
}

// Tag component for agent/action labels
function Tag({ children, active = false }: { children: React.ReactNode; active?: boolean }) {
  return (
    <span
      className={`inline-block px-1.5 py-0.5 border border-dashed rounded-sm text-[0.66rem] font-mono transition-all duration-200 ${
        active
          ? "border-terminal-white text-terminal-white"
          : "border-terminal-border text-terminal-mid"
      }`}
    >
      {children}
    </span>
  );
}

// Single trace row
function TraceRow({
  event,
  isLast,
  isActive,
}: {
  event: TraceEvent;
  isLast: boolean;
  isActive: boolean;
}) {
  const formatTime = (ts: string) => {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  };

  return (
    <div className="flex items-start gap-2.5 animate-slide-in">
      <div className="pt-1">
        <Dot on={isActive} last={isLast} />
      </div>
      <div className="flex-1 pb-3">
        <div className="flex items-center gap-1.5 flex-wrap mb-0.5">
          <Tag active={isActive}>{event.agent}</Tag>
          <span className="text-terminal-dim text-[0.68rem]">-&gt;</span>
          <span className="text-terminal-white text-[0.72rem] font-mono">{event.action}</span>
          {event.tool && (
            <span className="text-terminal-dim text-[0.66rem] font-mono">({event.tool})</span>
          )}
          {isActive && <Cursor />}
        </div>
        {event.detail && (
          <div className="text-terminal-mid text-[0.68rem] font-mono line-clamp-2 mt-0.5">{event.detail}</div>
        )}
        <div className="flex items-center gap-2.5 mt-1 text-terminal-dim text-[0.6rem] font-mono">
          <span>{formatTime(event.timestamp)}</span>
          {event.latency_ms !== undefined && <span>{event.latency_ms < 1000 ? `${event.latency_ms}ms` : `${(event.latency_ms / 1000).toFixed(2)}s`}</span>}
          {event.tokens_used !== undefined && <span>{event.tokens_used} tok</span>}
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
      <div className="flex items-center justify-center h-32 text-terminal-dim text-[0.72rem] font-mono">
        <span>{"// awaiting agent activity"}</span>
      </div>
    );
  }

  const isRunning = status === "running" || status === "synth";

  return (
    <div ref={containerRef} className="overflow-y-auto scrollbar-terminal">
      <div className="font-mono text-[0.62rem] text-terminal-dim uppercase tracking-wider mb-3">
        {"// agent_trace"}
      </div>
      {events.map((event, i) => (
        <TraceRow
          key={event.id || i}
          event={event}
          isLast={i === events.length - 1}
          isActive={isRunning && i === events.length - 1}
        />
      ))}
      {status === "complete" && (
        <div className="flex items-center gap-2 pt-2 mt-2 border-t border-dashed border-terminal-borderDotted">
          <span className="w-1.5 h-1.5 rounded-full bg-semantic-pass" />
          <span className="text-terminal-mid text-[0.68rem] font-mono">pipeline complete</span>
        </div>
      )}
      {status === "hitl" && (
        <div className="flex items-center gap-2 pt-2 mt-2 border-t border-dashed border-terminal-borderDotted">
          <span className="w-1.5 h-1.5 rounded-full bg-semantic-unknown animate-pulse" />
          <span className="text-terminal-mid text-[0.68rem] font-mono">awaiting human review</span>
        </div>
      )}
      {status === "error" && (
        <div className="flex items-center gap-2 pt-2 mt-2 border-t border-dashed border-semantic-fail">
          <span className="w-1.5 h-1.5 rounded-full bg-semantic-fail" />
          <span className="text-semantic-fail text-[0.68rem] font-mono">error occurred</span>
        </div>
      )}
    </div>
  );
}
