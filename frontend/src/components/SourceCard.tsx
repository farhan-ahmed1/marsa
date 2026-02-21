"use client";

import type { Citation } from "@/lib/types";

interface SourceCardProps {
  citation: Citation;
  verificationStatus?: "supported" | "unverifiable" | "contradicted";
  highlighted?: boolean;
}

function extractDomain(url: string): string {
  try {
    return new URL(url).hostname.replace("www.", "");
  } catch {
    return "unknown";
  }
}

function QualityBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const cls =
    score >= 0.7
      ? "text-semantic-pass border-semantic-pass/40 bg-semantic-passSubtle"
      : score >= 0.5
      ? "text-semantic-unknown border-semantic-unknownBorder bg-semantic-unknownSubtle"
      : "text-semantic-fail border-semantic-failBorder bg-semantic-failSubtle";
  return (
    <span className={`px-1.5 py-0.5 border rounded text-xs font-mono ${cls}`}>
      {pct}%
    </span>
  );
}

function VerificationIcon({ status }: { status?: "supported" | "unverifiable" | "contradicted" }) {
  if (status === "supported") {
    return (
      <span className="flex items-center gap-1 text-xs text-semantic-pass">
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
        verified
      </span>
    );
  }
  if (status === "contradicted") {
    return (
      <span className="flex items-center gap-1 text-xs text-semantic-fail">
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
        contradicted
      </span>
    );
  }
  if (status === "unverifiable") {
    return (
      <span className="flex items-center gap-1 text-xs text-semantic-unknown">
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01" />
        </svg>
        unverifiable
      </span>
    );
  }
  return <span className="text-xs text-terminal-dim">unchecked</span>;
}

export function SourceCard({ citation, verificationStatus, highlighted }: SourceCardProps) {
  const domain = citation.domain || extractDomain(citation.url);

  return (
    <div
      className={`border rounded-lg p-3.5 bg-terminal-surface transition-all duration-300 ${
        highlighted
          ? "border-accent/60 bg-accent/5 ring-1 ring-accent/30"
          : "border-terminal-border hover:border-terminal-mid hover:bg-terminal-surfaceHover"
      }`}
    >
      <div className="flex items-start justify-between gap-3 mb-2.5">
        <div className="flex-1 min-w-0">
          <a
            href={citation.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-terminal-white text-sm font-medium hover:text-accent transition-colors line-clamp-2 block"
          >
            <span className="text-terminal-dim text-xs mr-1.5 font-mono">[{citation.number}]</span>
            {citation.title || "Untitled Source"}
          </a>
        </div>
        <div className="flex-shrink-0">
          <QualityBadge score={citation.source_quality_score} />
        </div>
      </div>
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs text-terminal-dim font-mono">{domain}</span>
        {citation.accessed_date && (
          <span className="text-xs text-terminal-dim">{citation.accessed_date}</span>
        )}
        <VerificationIcon status={verificationStatus} />
      </div>
    </div>
  );
}

interface SourceListProps {
  citations: Citation[];
  verificationResults?: Map<string, "supported" | "unverifiable" | "contradicted">;
  highlightedCitation?: number | null;
}

export function SourceList({ citations, verificationResults, highlightedCitation }: SourceListProps) {
  if (citations.length === 0) {
    return (
      <div className="text-terminal-dim text-sm py-6 text-center">
        No sources cited.
      </div>
    );
  }

  return (
    <div className="space-y-2.5">
      <h3 className="text-xs font-semibold text-terminal-dim uppercase tracking-wider mb-3">
        Sources ({citations.length})
      </h3>
      {citations.map((citation, index) => (
        <SourceCard
          key={citation.number || index}
          citation={citation}
          verificationStatus={verificationResults?.get(citation.url)}
          highlighted={highlightedCitation === citation.number}
        />
      ))}
    </div>
  );
}

