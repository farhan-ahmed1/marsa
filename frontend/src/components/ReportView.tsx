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

// Custom component to render citation links [1], [2], etc.
function CitationLink({
  number,
  onHover,
  onLeave,
  onClick,
}: {
  number: number;
  onHover: (num: number) => void;
  onLeave: () => void;
  onClick: (num: number) => void;
}) {
  return (
    <button
      type="button"
      className="inline-block px-1 py-0.5 mx-0.5 text-[0.68rem] font-mono text-terminal-mid border border-dashed border-terminal-borderDotted rounded-sm hover:text-terminal-white hover:border-terminal-mid transition-all duration-200 cursor-pointer align-baseline"
      onMouseEnter={() => onHover(number)}
      onMouseLeave={onLeave}
      onClick={() => onClick(number)}
    >
      [{number}]
    </button>
  );
}

// Process text to replace [1], [2], etc. with interactive citation components
function processCitations(
  text: string,
  onHover: (num: number) => void,
  onLeave: () => void,
  onClick: (num: number) => void
): React.ReactNode[] {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, index) => {
    const match = part.match(/\[(\d+)\]/);
    if (match) {
      const citationNum = parseInt(match[1], 10);
      return (
        <CitationLink
          key={index}
          number={citationNum}
          onHover={onHover}
          onLeave={onLeave}
          onClick={onClick}
        />
      );
    }
    return <span key={index}>{part}</span>;
  });
}

// Skeleton loader for report
function ReportSkeleton() {
  return (
    <div className="animate-pulse">
      <div className="h-8 bg-terminal-surface rounded-sm w-2/3 mb-4" />
      <div className="h-4 bg-terminal-surface rounded-sm w-full mb-2" />
      <div className="h-4 bg-terminal-surface rounded-sm w-5/6 mb-6" />
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i}>
            <div className="h-5 bg-terminal-surface rounded-sm w-1/3 mb-2" />
            <div className="h-3 bg-terminal-surface rounded-sm w-full mb-1" />
            <div className="h-3 bg-terminal-surface rounded-sm w-4/5 mb-1" />
            <div className="h-3 bg-terminal-surface rounded-sm w-3/4" />
          </div>
        ))}
      </div>
    </div>
  );
}

export function ReportView({ report, rawReport, isLoading }: ReportViewProps) {
  const [hoveredCitation, setHoveredCitation] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const sourcesRef = useRef<HTMLDivElement>(null);

  const handleCitationHover = useCallback((num: number) => {
    setHoveredCitation(num);
  }, []);

  const handleCitationLeave = useCallback(() => {
    setHoveredCitation(null);
  }, []);

  const handleCitationClick = useCallback((num: number) => {
    // Scroll to the sources section and highlight the citation
    if (sourcesRef.current) {
      sourcesRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
      setHoveredCitation(num);
      // Clear highlight after 2 seconds
      setTimeout(() => setHoveredCitation(null), 2000);
    }
  }, []);

  const handleCopyReport = useCallback(async () => {
    const reportText = buildReportMarkdown(report, rawReport);
    if (reportText) {
      try {
        await navigator.clipboard.writeText(reportText);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (err) {
        console.error("Failed to copy report:", err);
      }
    }
  }, [report, rawReport]);

  // Build markdown from structured report or use raw
  const reportMarkdown = useMemo(() => {
    return buildReportMarkdown(report, rawReport);
  }, [report, rawReport]);

  if (isLoading) {
    return <ReportSkeleton />;
  }

  if (!report && !rawReport) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] opacity-30">
        <span className="font-mono text-[0.74rem]">{"// report"}</span>
        <span className="font-mono text-[0.66rem] mt-1">pending synthesis...</span>
      </div>
    );
  }

  const citations = report?.citations || [];

  return (
    <div className="animate-slide-in">
      {/* Header with copy button */}
      <div className="flex justify-between items-center mb-5 pb-3 border-b border-dashed border-terminal-borderDotted">
        <span className="font-mono text-[0.68rem] text-terminal-dim uppercase tracking-wider">
          {"// generated report"}
        </span>
        <button
          onClick={handleCopyReport}
          className={`px-2.5 py-1 rounded-sm border border-dashed border-terminal-borderDotted bg-transparent text-[0.7rem] font-mono cursor-pointer transition-all duration-200 ${
            copied
              ? "text-semantic-pass border-semantic-pass/40"
              : "text-terminal-dim hover:text-terminal-white hover:border-terminal-mid"
          }`}
        >
          {copied ? "copied" : "copy"}
        </button>
      </div>

      {/* Report title */}
      {report?.title && (
        <h1 className="font-sans text-[1.35rem] font-semibold text-terminal-pure leading-tight mb-3.5 tracking-tight">
          {report.title}
        </h1>
      )}

      {/* Executive summary */}
      {report?.summary && (
        <div className="px-3.5 py-3 rounded-sm border-l-2 border-terminal-dim bg-white/[0.02] font-sans text-[0.82rem] text-terminal-mid leading-relaxed mb-7">
          {processCitations(report.summary, handleCitationHover, handleCitationLeave, handleCitationClick)}
        </div>
      )}

      {/* Report sections */}
      {report?.sections && report.sections.length > 0 ? (
        <div className="space-y-6 mb-8">
          {report.sections.map((section, index) => (
            <section key={index} className="animate-slide-in" style={{ animationDelay: `${index * 0.05}s` }}>
              <h2 className="font-sans text-[1rem] font-semibold text-terminal-white mb-2.5 tracking-tight">
                {section.heading}
              </h2>
              <div className="font-mono text-[0.78rem] text-terminal-mid leading-relaxed">
                {processCitations(section.content, handleCitationHover, handleCitationLeave, handleCitationClick)}
              </div>
            </section>
          ))}
        </div>
      ) : rawReport ? (
        <div className="mb-8 prose prose-invert prose-sm max-w-none">
          <ReactMarkdown
            components={{
              h1: ({ children }) => (
                <h1 className="font-sans text-[1.35rem] font-semibold text-terminal-pure mb-3">{children}</h1>
              ),
              h2: ({ children }) => (
                <h2 className="font-sans text-[1rem] font-semibold text-terminal-white mb-2 mt-5">{children}</h2>
              ),
              h3: ({ children }) => (
                <h3 className="font-sans text-[0.9rem] font-semibold text-terminal-white mb-2 mt-4">{children}</h3>
              ),
              p: ({ children }) => (
                <p className="font-mono text-[0.78rem] text-terminal-mid leading-relaxed mb-3">{children}</p>
              ),
              ul: ({ children }) => (
                <ul className="list-none space-y-1.5 mb-3 pl-3">{children}</ul>
              ),
              li: ({ children }) => (
                <li className="font-mono text-[0.78rem] text-terminal-mid leading-relaxed flex items-start gap-2">
                  <span className="text-terminal-dim mt-[0.35rem]">-</span>
                  <span>{children}</span>
                </li>
              ),
              a: ({ href, children }) => (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-terminal-white hover:text-terminal-pure underline underline-offset-2"
                >
                  {children}
                </a>
              ),
              code: ({ children }) => (
                <code className="px-1 py-0.5 bg-terminal-surface rounded-sm text-[0.72rem] text-terminal-white">{children}</code>
              ),
              blockquote: ({ children }) => (
                <blockquote className="border-l-2 border-terminal-borderDotted pl-3 my-3 italic text-terminal-dim">{children}</blockquote>
              ),
            }}
          >
            {rawReport}
          </ReactMarkdown>
        </div>
      ) : null}

      {/* Confidence summary */}
      {report?.confidence_summary && (
        <div className="px-3.5 py-3 rounded-sm border border-dashed border-terminal-borderDotted bg-terminal-surface/50 mb-8">
          <div className="font-mono text-[0.62rem] text-terminal-dim uppercase tracking-wider mb-1.5">
            {"// confidence assessment"}
          </div>
          <p className="font-mono text-[0.74rem] text-terminal-mid leading-relaxed">
            {report.confidence_summary}
          </p>
        </div>
      )}

      {/* Sources section */}
      <div ref={sourcesRef}>
        <SourceList
          citations={citations.map((c, i) => ({
            ...c,
            // Highlight the hovered citation
            ...(hoveredCitation === c.number ? { _highlighted: true } : {}),
          }))}
        />
        {hoveredCitation !== null && (
          <div className="fixed bottom-4 right-4 px-3 py-2 bg-terminal-black border border-terminal-borderDotted rounded-sm shadow-lg font-mono text-[0.7rem] text-terminal-white animate-fade-in z-50 max-w-xs">
            <span className="text-terminal-dim">Source [{hoveredCitation}]: </span>
            {citations.find((c) => c.number === hoveredCitation)?.title || "Unknown source"}
          </div>
        )}
      </div>
    </div>
  );
}

// Helper to build markdown string from report
function buildReportMarkdown(report: Report | null, rawReport?: string | null): string {
  if (rawReport) return rawReport;
  if (!report) return "";

  const lines: string[] = [];
  if (report.title) {
    lines.push(`# ${report.title}`, "");
  }
  if (report.summary) {
    lines.push(report.summary, "");
  }
  for (const section of report.sections || []) {
    lines.push(`## ${section.heading}`, "", section.content, "");
  }
  if (report.confidence_summary) {
    lines.push("---", "", `*${report.confidence_summary}*`, "");
  }
  if (report.citations && report.citations.length > 0) {
    lines.push("## References", "");
    for (const citation of report.citations) {
      lines.push(`[${citation.number}] ${citation.title} - ${citation.url}`);
    }
  }
  return lines.join("\n");
}
