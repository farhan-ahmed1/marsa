"use client";

interface MetricItem {
  key: string;
  value: string;
  sub: string;
}

interface MetricsProps {
  show: boolean;
  metrics?: MetricItem[];
}

const DEFAULT_METRICS: MetricItem[] = [
  { key: "latency", value: "14.2s", sub: "p50" },
  { key: "llm_calls", value: "9", sub: "8,409 tok" },
  { key: "sources", value: "10", sub: "12 searched" },
  { key: "accuracy", value: "83%", sub: "15/18" },
];

export function Metrics({ show, metrics = DEFAULT_METRICS }: MetricsProps) {
  if (!show) return null;

  return (
    <div className="flex border border-dashed border-terminal-borderDotted rounded-sm overflow-hidden animate-slide-in mb-5">
      {metrics.map((m, i) => (
        <div
          key={m.key}
          className={`flex-1 px-3 py-2.5 ${
            i < metrics.length - 1
              ? "border-r border-dashed border-terminal-borderDotted"
              : ""
          }`}
        >
          <div className="font-mono text-[0.62rem] text-terminal-dim uppercase tracking-wider mb-0.5">
            {m.key}
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-base font-semibold text-terminal-pure">
              {m.value}
            </span>
            <span className="font-mono text-[0.62rem] text-terminal-dim">
              {m.sub}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
