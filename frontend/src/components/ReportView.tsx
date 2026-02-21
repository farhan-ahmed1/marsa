"use client";

import { useState, useCallback, useRef, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import type { Report, Citation } from "@/lib/types";
import { SourceList } from "./SourceCard";

interface ReportViewProps {
  report: Report | null;
  rawReport?: string | null;
  isLoading?: boolean;
}

// Inline citation button [1] [2] ...
function CitationLink({ number, onHover, onLeave, onClick }: {
  number: number;
  onHover: (n: number) => void;
  onLeave: () => void;
  onClick: (n: number) => void;
}) {
  return (
    <button
      type="button"
      className="inline-flex items-center justify-center w-5 h-5 mx-0.5 text-[0.65rem] font-mono font-semibold text-accent bg-accent/10 border border-accent/30 rounded hover:bg-accent/20 hover:border-accent/60 transition-all cursor-pointer align-baseline"
      onMouseEnter={() => onHover(number)}
      onMouseLeave={onLeave}
      onClick={() => onClick(number)}
    >
      {number}
    </button>
  );
}

// Process text and replace [N] with citation links
function processCitations(
  text: string,
  onHover: (n: number) => void,
  onLeave: () => void,
  onClick: (n: number) => void
): React.ReactNode[] {
  return text.split(/(\[\d+\])/g).map((part, i) => {
    const match = part.match(/\[(\d+)\]/);
    if (match) {
      return <CitationLink key={i} number={parseInt(match[1], 10)} onHover={onHover} onLeave={onLeave} onClick={onClick} />;
    }
    return <span key={i}>{part}</span>;
  });
}

// Skeleton loader
function ReportSkeleton() {
  return (
    <div className="p-8 animate-pulse space-y-4">
      <div className="h-8 bg-terminal-surface rounded-lg w-3/4" />
      <div className="h-4 bg-terminal-surface rounded w-full" />
      <div className="h-4 bg-terminal-surface rounded w-5/6" />
      <div className="h-4 bg-terminal-surface rounded w-4/5 mb-6" />
      {[1, 2, 3].map((i) => (
        <div key={i} className="space-y-2">
          <div className="h-5 bg-terminal-surface rounded w-1/3" />
          <div className="h-3.5 bg-terminal-surface rounded w-full" />
          <div className="h-3.5 bg-terminal-surface rounded w-11/12" />
          <div className="h-3.5 bg-terminal-surface rounded w-4/5" />
        </div>
      ))}
    </div>
  );
}

function buildReportMarkdown(report: Report | null, rawReport?: string | null): string {
  if (rawReport) return rawReport;
  if (!report) return "";
  const lines: string[] = [];
  if (report.title) lines.push(`# ${report.title}`, "");
  if (report.summary) lines.push(report.summary, "");
  for (const s of report.sections || []) lines.push(`## ${s.heading}`, "", s.content, "");
  if (report.confidence_summary) lines.push("---", "", `*${report.confidence_summary}*`, "");
  if (report.citations?.length) {
    lines.push("## References", "");
    for (const c of report.citations) lines.push(`[${c.number}] ${c.title} — ${c.url}`);
  }
  return lines.join("\n");
}

export function ReportView({ report, rawReport, isLoading }: ReportViewProps) {
  const [hoveredCitation, setHoveredCitation] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const sourcesRef = useRef<HTMLDivElement>(null);

  const handleCitationHover = useCallback((n: number) => setHoveredCitation(n), []);
  const handleCitationLeave = useCallback(() => setHoveredCitation(null), []);
  const handleCitationClick = useCallback((n: number) => {
    sourcesRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    setHoveredCitation(n);
    setTimeout(() => setHoveredCitation(null), 2000);
  }, []);

  const handleCopy = useCallback(async () => {
    const text = buildReportMarkdown(report, rawReport);
    if (text) {
      try {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch {}
    }
  }, [report, rawReport]);

  if (isLoading) return <ReportSkeleton />;

  if (!report && !rawReport) {
    return (
      <div className="flex flex-col items-center justify-center h-full opacity-40 px-8">
        <p className="text-terminal-dim text-sm">Report will appear here once research is complete.</p>
      </div>
    );
  }

  const citations = report?.citations || [];

  return (
    <div className="p-8 max-w-3xl mx-auto animate-fade-in">
      {/* Toolbar */}
      <div className="flex justify-between items-center mb-6">
        <span className="text-xs font-medium text-terminal-dim uppercase tracking-wider">Research Report</span>
        <button
          onClick={handleCopy}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-sm transition-all ${
            copied
              ? "text-semantic-pass border-semantic-passBorder bg-semantic-passSubtle"
              : "text-terminal-mid border-terminal-border hover:text-terminal-white hover:border-terminal-mid bg-transparent"
          }`}
        >
          {copied ? (
            <>
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
              Copied
            </>
          ) : (
            <>
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
              Copy
            </>
          )}
        </button>
      </div>

      {/* Title */}
      {report?.title && (
        <h1 className="text-2xl font-bold text-terminal-pure leading-tight mb-4 tracking-tight">
          {report.title}
        </h1>
      )}

      {/* Executive summary */}
      {report?.summary && (
        <div className="border-l-4 border-accent/50 pl-4 mb-8">
          <p className="text-terminal-mid text-base leading-relaxed">
            {processCitations(report.summary, handleCitationHover, handleCitationLeave, handleCitationClick)}
          </p>
        </div>
      )}

      {/* Sections */}
      {report?.sections && report.sections.length > 0 ? (
        <div className="space-y-7 mb-10">
          {report.sections.map((section, i) => (
            <section key={i}>
              <h2 className="text-lg font-semibold text-terminal-white mb-3 pb-2 border-b border-terminal-border">
                {section.heading}
              </h2>
              <p className="text-terminal-mid text-sm leading-relaxed font-mono">
                {processCitations(section.content, handleCitationHover, handleCitationLeave, handleCitationClick)}
              </p>
            </section>
          ))}
        </div>
      ) : rawReport ? (
        <div className="mb-10 prose prose-sm max-w-none">
          <ReactMarkdown
            components={{
              h1: ({ children }) => <h1 className="text-2xl font-bold text-terminal-pure mb-4">{children}</h1>,
              h2: ({ children }) => <h2 className="text-lg font-semibold text-terminal-white mb-3 mt-6 pb-2 border-b border-terminal-border">{children}</h2>,
              h3: ({ children }) => <h3 className="text-base font-semibold text-terminal-white mb-2 mt-4">{children}</h3>,
              p: ({ children }) => <p className="text-terminal-mid text-sm leading-relaxed mb-3 font-mono">{children}</p>,
              ul: ({ children }) => <ul className="space-y-1.5 mb-3 pl-4">{children}</ul>,
              li: ({ children }) => (
                <li className="text-terminal-mid text-sm leading-relaxed font-mono flex items-start gap-2">
                  <span className="text-accent mt-1.5 flex-shrink-0">•</span>
                  <span>{children}</span>
                </li>
              ),
              a: ({ href, children }) => (
                <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">
                  {children}
                </a>
              ),
              code: ({ children }) => (
                <code className="px-1.5 py-0.5 bg-terminal-surface border border-terminal-border rounded text-xs text-terminal-white font-mono">{children}</code>
              ),
              blockquote: ({ children }) => (
                <blockquote className="border-l-4 border-accent/40 pl-4 my-4 text-terminal-dim italic">{children}</blockquote>
              ),
            }}
          >
            {rawReport}
          </ReactMarkdown>
        </div>
      ) : null}

      {/* Confidence summary */}
      {report?.confidence_summary && (
        <div className="rounded-lg border border-terminal-border bg-terminal-surface p-4 mb-8">
          <p className="text-xs font-semibold text-terminal-dim uppercase tracking-wider mb-2">Confidence Assessment</p>
          <p className="text-terminal-mid text-sm leading-relaxed font-mono">{report.confidence_summary}</p>
        </div>
      )}

      {/* Sources */}
      {citations.length > 0 && (
        <div ref={sourcesRef}>
          <SourceList citations={citations} highlightedCitation={hoveredCitation} />
        </div>
      )}

      {/* Citation tooltip */}
      {hoveredCitation !== null && (
        <div className="fixed bottom-5 right-5 px-3 py-2 bg-terminal-surface border border-terminal-border rounded-lg shadow-xl text-sm text-terminal-white animate-fade-in z-50 max-w-xs">
          <span className="text-terminal-dim text-xs">[{hoveredCitation}] </span>
          {citations.find((c) => c.number === hoveredCitation)?.title || "Unknown source"}
        </div>
      )}
    </div>
  );
}

