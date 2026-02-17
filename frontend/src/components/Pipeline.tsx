"use client";

import type { AgentName, StreamStatus } from "@/lib/types";

interface PipelineProps {
  currentAgent: AgentName | null;
  status: StreamStatus;
}

const STEPS: AgentName[] = ["planner", "researcher", "fact_checker", "synthesizer"];

function Dot({ on = false }: { on?: boolean }) {
  return (
    <span
      className={`inline-block w-1.5 h-1.5 rounded-full transition-all duration-300 ${
        on ? "bg-terminal-white shadow-[0_0_6px_rgba(255,255,255,0.3)]" : "bg-terminal-dim"
      }`}
    />
  );
}

export function Pipeline({ currentAgent, status }: PipelineProps) {
  if (status === "idle") return null;

  const currentIndex = currentAgent ? STEPS.indexOf(currentAgent) : -1;

  return (
    <div className="flex items-center px-6 py-2.5 border-b border-dashed border-terminal-borderDotted animate-fade-in">
      {STEPS.map((step, i) => {
        const done = i < currentIndex || status === "complete";
        const active = step === currentAgent;

        return (
          <div key={step} className="flex items-center flex-1">
            <div className="flex items-center gap-1.5">
              {/* Step circle */}
              <div
                className={`w-[22px] h-[22px] rounded-full flex items-center justify-center transition-all duration-300 ${
                  done
                    ? "border-[1.5px] border-terminal-white bg-terminal-white"
                    : active
                    ? "border-[1.5px] border-terminal-white shadow-[0_0_10px_rgba(255,255,255,0.12)]"
                    : "border border-dashed border-terminal-borderDotted"
                }`}
              >
                {done ? (
                  <span className="text-terminal-black text-[0.6rem] font-bold">
                    {"\u2713"}
                  </span>
                ) : active ? (
                  <Dot on />
                ) : (
                  <span className="w-[3px] h-[3px] rounded-full bg-terminal-dim" />
                )}
              </div>

              {/* Step label */}
              <span
                className={`font-mono text-[0.68rem] transition-all duration-300 ${
                  done || active ? "text-terminal-white" : "text-terminal-dim"
                }`}
              >
                {step.replace("_", " ")}
              </span>
            </div>

            {/* Connector line */}
            {i < STEPS.length - 1 && (
              <div
                className={`flex-1 mx-2.5 transition-all duration-400 ${
                  done
                    ? "border-t border-terminal-dim"
                    : "border-t border-dashed border-terminal-borderDotted"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
