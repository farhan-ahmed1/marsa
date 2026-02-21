"use client";

import { useMemo, useState, useRef, useEffect } from "react";
import type { TraceEvent } from "@/lib/types";

// --- Config ---

const AGENT_ORDER = ["planner", "researcher", "fact_checker", "synthesizer"] as const;

const AGENT_CONFIG: Record<string, { label: string; bar: string; textColor: string }> = {
  planner:      { label: "Planner",      bar: "#60a5fa", textColor: "#93c5fd" },
  researcher:   { label: "Researcher",   bar: "#34d399", textColor: "#6ee7b7" },
  fact_checker: { label: "Fact Checker", bar: "#fbbf24", textColor: "#fcd34d" },
  synthesizer:  { label: "Synthesizer",  bar: "#a78bfa", textColor: "#c4b5fd" },
};

// --- Types ---

interface AgentSpan {
  agent: string;
  startMs: number;
  endMs: number;
  events: TraceEvent[];
  totalTokens: number;
  toolCalls: number;
  llmCalls: number;
}

interface ComputedTimeline {
  spans: AgentSpan[];
  totalMs: number;
  startTs: number;
}

// --- Helpers ---

function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function computeTimeline(events: TraceEvent[]): ComputedTimeline {
  if (events.length === 0) return { spans: [], totalMs: 0, startTs: 0 };

  const startTs = new Date(events[0].timestamp).getTime();

  // Group events by agent
  const byAgent: Record<string, TraceEvent[]> = {};
  for (const e of events) {
    if (!byAgent[e.agent]) byAgent[e.agent] = [];
    byAgent[e.agent].push(e);
  }

  const spans: AgentSpan[] = [];
  let totalMs = 0;

  for (const agent of AGENT_ORDER) {
    const evs = byAgent[agent];
    if (!evs || evs.length === 0) continue;

    const startMs = new Date(evs[0].timestamp).getTime() - startTs;
    const lastEv = evs[evs.length - 1];
    const endMs =
      new Date(lastEv.timestamp).getTime() -
      startTs +
      (lastEv.latency_ms ?? 400);

    const totalTokens = evs.reduce((a, e) => a + (e.tokens_used ?? 0), 0);
    const toolCalls = evs.filter((e) => e.type === "tool_called").length;
    const llmCalls = evs.filter((e) => (e.tokens_used ?? 0) > 0).length;

    spans.push({ agent, startMs, endMs, events: evs, totalTokens, toolCalls, llmCalls });
    totalMs = Math.max(totalMs, endMs);
  }

  return { spans, totalMs: Math.max(totalMs, 500), startTs };
}

// --- Sub-components ---

interface DetailPanelProps {
  event: TraceEvent;
  onClose: () => void;
}

function DetailPanel({ event, onClose }: DetailPanelProps) {
  const cfg = AGENT_CONFIG[event.agent] ?? { label: event.agent, textColor: "#999" };
  return (
    <div className="mt-3 p-3 rounded border border-terminal-border bg-terminal-surface text-xs font-mono animate-slide-in">
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold" style={{ color: cfg.textColor }}>
          {cfg.label} &rarr; {event.action}
        </span>
        <button
          onClick={onClose}
          className="text-terminal-dim hover:text-terminal-white transition-colors"
        >
          &times;
        </button>
      </div>
      {event.detail && (
        <p className="text-terminal-mid mb-1.5 leading-relaxed">{event.detail}</p>
      )}
      <div className="flex items-center gap-4 text-terminal-dim">
        {event.latency_ms !== undefined && (
          <span>{formatMs(event.latency_ms)}</span>
        )}
        {event.tokens_used !== undefined && event.tokens_used > 0 && (
          <span>{event.tokens_used.toLocaleString()} tok</span>
        )}
        {event.tool && <span>tool: {event.tool}</span>}
        <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
      </div>
    </div>
  );
}

// --- Main Component ---

interface TimelineProps {
  events: TraceEvent[];
}

export function Timeline({ events }: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(600);
  const [selectedEvent, setSelectedEvent] = useState<TraceEvent | null>(null);

  // Observe container width for responsive SVG
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setContainerWidth(w);
    });
    ro.observe(el);
    setContainerWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const timeline = useMemo(() => computeTimeline(events), [events]);

  if (timeline.spans.length === 0) {
    return (
      <div className="flex items-center justify-center h-20 text-terminal-dim text-xs">
        No timeline data yet
      </div>
    );
  }

  // Layout constants
  const LABEL_W = 100;
  const AXIS_H = 22;
  const ROW_H = 36;
  const BAR_H = 14;
  const BAR_Y_OFFSET = (ROW_H - BAR_H) / 2;
  const MARKER_R = 3.5;
  const svgHeight = timeline.spans.length * ROW_H + AXIS_H;
  const chartW = containerWidth - LABEL_W;
  const scaleX = (ms: number) => (ms / timeline.totalMs) * chartW;

  // Tick marks for time axis
  const tickCount = Math.min(6, Math.floor(chartW / 60));
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) =>
    (timeline.totalMs / tickCount) * i
  );

  return (
    <div ref={containerRef} className="w-full">
      <svg
        width={containerWidth}
        height={svgHeight}
        className="overflow-visible"
      >
        {/* Row backgrounds */}
        {timeline.spans.map((span, i) => (
          <rect
            key={`bg-${span.agent}`}
            x={0}
            y={i * ROW_H}
            width={containerWidth}
            height={ROW_H}
            fill={i % 2 === 0 ? "rgba(255,255,255,0.015)" : "transparent"}
          />
        ))}

        {/* Agent labels */}
        {timeline.spans.map((span, i) => {
          const cfg = AGENT_CONFIG[span.agent] ?? { label: span.agent, textColor: "#999" };
          return (
            <text
              key={`label-${span.agent}`}
              x={0}
              y={i * ROW_H + ROW_H / 2 + 4}
              fill={cfg.textColor}
              fontSize={10}
              fontFamily="monospace"
              fontWeight="600"
            >
              {cfg.label}
            </text>
          );
        })}

        {/* Chart area - bars and markers */}
        <g transform={`translate(${LABEL_W}, 0)`}>
          {/* Vertical grid lines */}
          {ticks.map((t, i) => (
            <line
              key={`grid-${i}`}
              x1={scaleX(t)}
              x2={scaleX(t)}
              y1={0}
              y2={timeline.spans.length * ROW_H}
              stroke="rgba(255,255,255,0.06)"
              strokeWidth={1}
            />
          ))}

          {timeline.spans.map((span, i) => {
            const cfg = AGENT_CONFIG[span.agent] ?? { label: span.agent, bar: "#888", textColor: "#999" };
            const barX = scaleX(span.startMs);
            const barW = Math.max(scaleX(span.endMs) - scaleX(span.startMs), 4);
            const barY = i * ROW_H + BAR_Y_OFFSET;
            const markerY = i * ROW_H + ROW_H / 2;
            const durationMs = span.endMs - span.startMs;

            return (
              <g key={`span-${span.agent}`}>
                {/* Agent bar */}
                <rect
                  x={barX}
                  y={barY}
                  width={barW}
                  height={BAR_H}
                  rx={3}
                  fill={cfg.bar}
                  opacity={0.25}
                />
                {/* Bar outline */}
                <rect
                  x={barX}
                  y={barY}
                  width={barW}
                  height={BAR_H}
                  rx={3}
                  fill="none"
                  stroke={cfg.bar}
                  strokeWidth={1}
                  opacity={0.6}
                />

                {/* Duration label inside/after bar */}
                {barW > 34 && (
                  <text
                    x={barX + barW / 2}
                    y={barY + BAR_H / 2 + 3.5}
                    fill={cfg.textColor}
                    fontSize={8.5}
                    fontFamily="monospace"
                    textAnchor="middle"
                    opacity={0.9}
                  >
                    {formatMs(durationMs)}
                    {span.totalTokens > 0 ? ` · ${(span.totalTokens / 1000).toFixed(1)}k tok` : ""}
                  </text>
                )}

                {/* Event markers (tool calls and LLM calls) */}
                {span.events.map((ev, j) => {
                  const isToolCall = ev.type === "tool_called";
                  const hasTokens = (ev.tokens_used ?? 0) > 0;
                  if (!isToolCall && !hasTokens) return null;

                  const evMs = new Date(ev.timestamp).getTime() - timeline.startTs;
                  const mx = scaleX(Math.max(evMs, span.startMs + (span.endMs - span.startMs) * 0.1));
                  const isSelected = selectedEvent === ev;
                  const markerColor = isToolCall ? "#fb923c" : "#e2e8f0";
                  const r = isSelected ? 5 : MARKER_R;

                  return (
                    <g key={`marker-${span.agent}-${j}`}>
                      {/* Vertical tick line on bar */}
                      <line
                        x1={mx}
                        x2={mx}
                        y1={barY}
                        y2={barY + BAR_H}
                        stroke={markerColor}
                        strokeWidth={1}
                        opacity={0.5}
                      />
                      {/* Marker circle above bar */}
                      <circle
                        cx={mx}
                        cy={markerY - BAR_H / 2 - 3}
                        r={r}
                        fill={markerColor}
                        opacity={0.9}
                        className="cursor-pointer"
                        onClick={() =>
                          setSelectedEvent(isSelected ? null : ev)
                        }
                      />
                    </g>
                  );
                })}
              </g>
            );
          })}

          {/* Time axis */}
          <g transform={`translate(0, ${timeline.spans.length * ROW_H})`}>
            <line
              x1={0}
              x2={chartW}
              y1={0}
              y2={0}
              stroke="rgba(255,255,255,0.12)"
              strokeWidth={1}
            />
            {ticks.map((t, i) => (
              <g key={`tick-${i}`} transform={`translate(${scaleX(t)}, 0)`}>
                <line x1={0} x2={0} y1={0} y2={4} stroke="rgba(255,255,255,0.2)" />
                <text
                  x={0}
                  y={14}
                  fill="rgba(255,255,255,0.35)"
                  fontSize={9}
                  fontFamily="monospace"
                  textAnchor={i === 0 ? "start" : i === tickCount ? "end" : "middle"}
                >
                  {formatMs(t)}
                </text>
              </g>
            ))}
          </g>
        </g>
      </svg>

      {/* Detail panel for selected event */}
      {selectedEvent && (
        <DetailPanel
          event={selectedEvent}
          onClose={() => setSelectedEvent(null)}
        />
      )}

      {/* Legend */}
      <div className="flex items-center gap-4 mt-2 text-xs font-mono text-terminal-dim">
        <div className="flex items-center gap-1.5">
          <svg width="10" height="10">
            <circle cx="5" cy="5" r="3.5" fill="#fb923c" opacity={0.9} />
          </svg>
          <span>tool call</span>
        </div>
        <div className="flex items-center gap-1.5">
          <svg width="10" height="10">
            <circle cx="5" cy="5" r="3.5" fill="#e2e8f0" opacity={0.9} />
          </svg>
          <span>llm call</span>
        </div>
        <div className="flex items-center gap-1.5 ml-auto">
          <span>total: {formatMs(timeline.totalMs)}</span>
        </div>
      </div>
    </div>
  );
}
