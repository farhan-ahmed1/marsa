"use client";

import type { Citation } from "@/lib/types";

interface SourceCardProps {
  citation: Citation;
  verificationStatus?: "supported" | "unverifiable" | "contradicted";
}

// Extract domain from URL
function extractDomain(url: string): string {
  try {
    const parsed = new URL(url);
    return parsed.hostname.replace("www.", "");
  } catch {
    return "unknown";
  }
}

// Get quality score badge styling
function getQualityBadge(score: number): { text: string; className: string } {
  if (score >= 0.7) {
    return {
      text: `${Math.round(score * 100)}%`,
      className: "text-semantic-pass border-semantic-pass/40 bg-semantic-pass/10",
    };
  }
  if (score >= 0.5) {
    return {
      text: `${Math.round(score * 100)}%`,
      className: "text-semantic-unknown border-semantic-unknown/40 bg-semantic-unknown/10",
    };
  }
  return {
    text: `${Math.round(score * 100)}%`,
    className: "text-semantic-fail border-semantic-fail/40 bg-semantic-fail/10",
  };
}

// Get verification status indicator
function getVerificationIndicator(status?: "supported" | "unverifiable" | "contradicted"): { icon: string; className: string; label: string } {
  switch (status) {
    case "supported":
      return { icon: "OK", className: "text-semantic-pass", label: "verified" };
    case "unverifiable":
      return { icon: "??", className: "text-semantic-unknown", label: "unverifiable" };
    case "contradicted":
      return { icon: "!!", className: "text-semantic-fail", label: "contradicted" };
    default:
      return { icon: "--", className: "text-terminal-dim", label: "unchecked" };
  }
}

export function SourceCard({ citation, verificationStatus }: SourceCardProps) {
  const domain = citation.domain || extractDomain(citation.url);
  const qualityBadge = getQualityBadge(citation.source_quality_score);
  const verification = getVerificationIndicator(verificationStatus);

  return (
    <div className="border border-dashed border-terminal-borderDotted rounded-sm p-3 bg-terminal-surface/50 hover:bg-terminal-surfaceHover transition-all duration-200 animate-slide-in">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex-1 min-w-0">
          <a
            href={citation.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-terminal-white text-[0.78rem] font-mono hover:text-terminal-pure transition-colors duration-200 line-clamp-2 block"
          >
            <span className="text-terminal-mid mr-1.5">[{citation.number}]</span>
            {citation.title || "Untitled Source"}
          </a>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className={`px-1.5 py-0.5 border rounded-sm text-[0.62rem] font-mono ${qualityBadge.className}`}>
            {qualityBadge.text}
          </span>
        </div>
      </div>
      <div className="flex items-center gap-3 text-[0.66rem] font-mono">
        <span className="text-terminal-mid">{domain}</span>
        {citation.accessed_date && (
          <>
            <span className="text-terminal-dim">|</span>
            <span className="text-terminal-dim">{citation.accessed_date}</span>
          </>
        )}
        <span className="text-terminal-dim">|</span>
        <span className={`flex items-center gap-1 ${verification.className}`}>
          <span className="text-[0.6rem]">{verification.icon}</span>
          <span>{verification.label}</span>
        </span>
      </div>
    </div>
  );
}

interface SourceListProps {
  citations: Citation[];
  verificationResults?: Map<string, "supported" | "unverifiable" | "contradicted">;
}

export function SourceList({ citations, verificationResults }: SourceListProps) {
  if (citations.length === 0) {
    return (
      <div className="text-terminal-dim text-[0.72rem] font-mono py-4 text-center">
        {"// no sources cited"}
      </div>
    );
  }

  return (
    <div className="space-y-2.5">
      <div className="font-mono text-[0.62rem] text-terminal-dim uppercase tracking-wider mb-3">
        {"// sources"} ({citations.length})
      </div>
      {citations.map((citation, index) => (
        <SourceCard
          key={citation.number || index}
          citation={citation}
          verificationStatus={verificationResults?.get(citation.url)}
        />
      ))}
    </div>
  );
}
