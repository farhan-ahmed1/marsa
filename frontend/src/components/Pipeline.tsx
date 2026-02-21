"use client";

import type { AgentName, StreamStatus } from "@/lib/types";

interface PipelineProps {
  currentAgent: AgentName | null;
  status: StreamStatus;
}

const STEPS: { id: AgentName; label: string; color: string; activeBg: string; doneBg: string }[] = [
  { id: "planner",      label: "Plan",     color: "text-blue-400",    activeBg: "bg-blue-400/20 border-blue-400/40",    doneBg: "bg-blue-500 border-blue-500" },
  { id: "researcher",   label: "Research", color: "text-emerald-400", activeBg: "bg-emerald-400/20 border-emerald-400/40", doneBg: "bg-emerald-500 border-emerald-500" },
  { id: "fact_checker", label: "Verify",   color: "text-amber-400",   activeBg: "bg-amber-400/20 border-amber-400/40",   doneBg: "bg-amber-500 border-amber-500" },
  { id: "synthesizer",  label: "Synthesize",color: "text-violet-400", activeBg: "bg-violet-400/20 border-violet-400/40", doneBg: "bg-violet-500 border-violet-500" },
];

export function Pipeline({ currentAgent, status }: PipelineProps) {
  if (status === "idle") return null;

  const currentIndex = currentAgent ? STEPS.findIndex((s) => s.id === currentAgent) : -1;

  return (
    <div className="flex items-center px-6 py-2 border-b border-terminal-border bg-terminal-surface/50 animate-fade-in">
      {STEPS.map((step, i) => {
        const done = i < currentIndex || status === "complete";
        const active = step.id === currentAgent;

        return (
          <div key={step.id} className="flex items-center flex-1">
            <div className="flex items-center gap-2">
              {/* Circle */}
              <div
                className={`w-6 h-6 rounded-full border flex items-center justify-center transition-all duration-300 ${
                  done
                    ? step.doneBg
                    : active
                    ? step.activeBg
                    : "border-terminal-border bg-transparent"
                }`}
              >
                {done ? (
                  <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                ) : active ? (
                  <span className={`w-2 h-2 rounded-full ${step.color.replace("text-", "bg-")} animate-pulse`} />
                ) : (
                  <span className="w-1.5 h-1.5 rounded-full bg-terminal-border" />
                )}
              </div>

              {/* Label */}
              <span
                className={`text-xs font-medium transition-all duration-300 ${
                  done || active ? step.color : "text-terminal-dim"
                }`}
              >
                {step.label}
              </span>
            </div>

            {/* Connector */}
            {i < STEPS.length - 1 && (
              <div className={`flex-1 mx-3 h-px transition-all duration-500 ${done ? "bg-terminal-mid" : "bg-terminal-border"}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}
