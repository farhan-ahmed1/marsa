"use client";

import { useState, useCallback } from "react";
import type { VerificationResult, FeedbackAction } from "@/lib/types";

interface HITLFeedbackProps {
  streamId: string;
  summary: string;
  claims: VerificationResult[];
  sourceQuality: number;
  onSubmitFeedback: (action: FeedbackAction, detail?: string) => Promise<void>;
  isSubmitting?: boolean;
}

function VerdictBadge({ verdict }: { verdict: "supported" | "contradicted" | "unverifiable" }) {
  const cfg = {
    supported: { label: "supported", cls: "text-semantic-pass bg-semantic-passSubtle border-semantic-passBorder" },
    contradicted: { label: "contradicted", cls: "text-semantic-fail bg-semantic-failSubtle border-semantic-failBorder" },
    unverifiable: { label: "unverifiable", cls: "text-semantic-unknown bg-semantic-unknownSubtle border-semantic-unknownBorder" },
  }[verdict];
  return (
    <span className={`px-2 py-0.5 border rounded-full text-xs font-medium ${cfg.cls}`}>
      {cfg.label}
    </span>
  );
}

function ClaimCard({ claim }: { claim: VerificationResult }) {
  const [expanded, setExpanded] = useState(false);
  const conf = claim.confidence;
  const confColor = conf >= 0.7 ? "text-semantic-pass" : conf >= 0.5 ? "text-semantic-unknown" : "text-semantic-fail";

  return (
    <div className="border border-terminal-border rounded-lg p-3.5 bg-terminal-surface hover:bg-terminal-surfaceHover transition-colors">
      <div className="flex items-start justify-between gap-3 mb-2">
        <p className="text-sm text-terminal-white leading-relaxed flex-1">
          {claim.claim.statement}
        </p>
        <VerdictBadge verdict={claim.verdict} />
      </div>
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs text-terminal-dim">
          confidence: <span className={`${confColor} font-medium`}>{Math.round(conf * 100)}%</span>
        </span>
        <span className="text-terminal-borderDotted">·</span>
        <span className="text-xs text-terminal-dim">{claim.claim.category}</span>
        {claim.reasoning && (
          <>
            <span className="text-terminal-borderDotted">·</span>
            <button
              type="button"
              onClick={() => setExpanded(!expanded)}
              className="text-xs text-accent hover:text-accent/80 transition-colors"
            >
              {expanded ? "hide reasoning" : "view reasoning"}
            </button>
          </>
        )}
      </div>
      {expanded && claim.reasoning && (
        <div className="mt-3 pt-3 border-t border-terminal-border">
          <p className="text-xs text-terminal-mid leading-relaxed font-mono">
            {claim.reasoning}
          </p>
          <div className="flex gap-4 mt-2">
            {claim.supporting_sources.length > 0 && (
              <span className="text-xs text-semantic-pass">{claim.supporting_sources.length} supporting</span>
            )}
            {claim.contradicting_sources.length > 0 && (
              <span className="text-xs text-semantic-fail">{claim.contradicting_sources.length} contradicting</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function HITLFeedback({
  streamId,
  summary,
  claims,
  sourceQuality,
  onSubmitFeedback,
  isSubmitting = false,
}: HITLFeedbackProps) {
  const [selectedAction, setSelectedAction] = useState<FeedbackAction | null>(null);
  const [detailInput, setDetailInput] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleActionClick = useCallback((action: FeedbackAction) => {
    setSelectedAction(action);
    setError(null);
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!selectedAction) return;
    if ((selectedAction === "dig_deeper" || selectedAction === "correct") && !detailInput.trim()) {
      setError(selectedAction === "dig_deeper" ? "Please specify what to explore further" : "Please provide your correction");
      return;
    }
    try {
      await onSubmitFeedback(selectedAction, detailInput.trim() || undefined);
      setSelectedAction(null);
      setDetailInput("");
    } catch {
      setError("Failed to submit feedback. Please try again.");
    }
  }, [selectedAction, detailInput, onSubmitFeedback]);

  const handleCancel = useCallback(() => {
    setSelectedAction(null);
    setDetailInput("");
    setError(null);
  }, []);

  const supportedCount = claims.filter((c) => c.verdict === "supported").length;
  const contradictedCount = claims.filter((c) => c.verdict === "contradicted").length;
  const unverifiableCount = claims.filter((c) => c.verdict === "unverifiable").length;

  return (
    <div className="animate-fade-in space-y-5">
      {/* Header */}
      <div className="flex items-center gap-2.5 pb-4 border-b border-terminal-border">
        <span className="w-2 h-2 rounded-full bg-semantic-unknown animate-pulse flex-shrink-0" />
        <span className="text-sm font-semibold text-terminal-white">Human Review Required</span>
      </div>

      {/* Summary */}
      <div className="border-l-4 border-accent/50 pl-4 py-1">
        <p className="text-sm text-terminal-mid leading-relaxed">
          {summary || "Research is complete. Please review the claims below before proceeding to synthesis."}
        </p>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-4 gap-2">
        {[
          { count: supportedCount, label: "Supported", color: "text-semantic-pass", bg: "bg-semantic-passSubtle border-semantic-passBorder" },
          { count: unverifiableCount, label: "Unverifiable", color: "text-semantic-unknown", bg: "bg-semantic-unknownSubtle border-semantic-unknownBorder" },
          { count: contradictedCount, label: "Contradicted", color: "text-semantic-fail", bg: "bg-semantic-failSubtle border-semantic-failBorder" },
          { count: Math.round(sourceQuality * 100), label: "Src Quality", color: "text-terminal-white", bg: "bg-terminal-surface border-terminal-border", suffix: "%" },
        ].map(({ count, label, color, bg, suffix }) => (
          <div key={label} className={`rounded-lg border p-3 text-center ${bg}`}>
            <div className={`text-xl font-bold ${color}`}>{count}{suffix ?? ""}</div>
            <div className="text-xs text-terminal-dim mt-0.5">{label}</div>
          </div>
        ))}
      </div>

      {/* Claims list */}
      <div>
        <h3 className="text-xs font-semibold text-terminal-dim uppercase tracking-wider mb-3">
          Verified Claims ({claims.length})
        </h3>
        <div className="space-y-2 max-h-72 overflow-y-auto scrollbar-terminal pr-1">
          {claims.map((claim, index) => (
            <ClaimCard key={index} claim={claim} />
          ))}
        </div>
      </div>

      {/* Actions */}
      {!selectedAction ? (
        <div className="space-y-3 pt-2">
          <h3 className="text-xs font-semibold text-terminal-dim uppercase tracking-wider">Choose Action</h3>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => handleActionClick("approve")}
              disabled={isSubmitting}
              className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg border border-semantic-passBorder bg-semantic-passSubtle text-semantic-pass text-sm font-medium hover:brightness-110 transition-all disabled:opacity-50 col-span-2"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Approve & Continue to Synthesis
            </button>
            <button
              onClick={() => handleActionClick("dig_deeper")}
              disabled={isSubmitting}
              className="px-4 py-2.5 rounded-lg border border-terminal-border bg-terminal-surface text-terminal-white text-sm font-medium hover:bg-terminal-surfaceHover hover:border-terminal-mid transition-all disabled:opacity-50"
            >
              Dig Deeper
            </button>
            <button
              onClick={() => handleActionClick("correct")}
              disabled={isSubmitting}
              className="px-4 py-2.5 rounded-lg border border-terminal-border bg-terminal-surface text-terminal-white text-sm font-medium hover:bg-terminal-surfaceHover hover:border-terminal-mid transition-all disabled:opacity-50"
            >
              Correct
            </button>
            <button
              onClick={() => handleActionClick("abort")}
              disabled={isSubmitting}
              className="px-4 py-2.5 rounded-lg border border-semantic-failBorder bg-semantic-failSubtle text-semantic-fail text-sm font-medium hover:brightness-110 transition-all disabled:opacity-50 col-span-2"
            >
              Abort Research
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-3 animate-fade-in pt-2">
          {(selectedAction === "dig_deeper" || selectedAction === "correct") && (
            <>
              <label className="block text-sm font-medium text-terminal-white">
                {selectedAction === "dig_deeper"
                  ? "What would you like to explore further?"
                  : "What corrections would you like to provide?"}
              </label>
              <textarea
                value={detailInput}
                onChange={(e) => setDetailInput(e.target.value)}
                placeholder={
                  selectedAction === "dig_deeper"
                    ? "e.g., performance benchmarks, memory usage comparisons..."
                    : "e.g., The claim about X is incorrect because..."
                }
                rows={3}
                className="w-full bg-terminal-surface border border-terminal-border rounded-lg p-3 text-terminal-white placeholder:text-terminal-dim resize-none outline-none text-sm focus:border-accent/60 focus:ring-2 focus:ring-accent/20 transition-all"
              />
            </>
          )}

          {(selectedAction === "approve" || selectedAction === "abort") && (
            <p className="text-sm text-terminal-mid">
              {selectedAction === "approve"
                ? "Confirm: Continue to synthesis with current findings?"
                : "Confirm: Abort and discard all research?"}
            </p>
          )}

          {error && (
            <p className="text-sm text-semantic-fail animate-fade-in">{error}</p>
          )}

          <div className="flex gap-2.5">
            <button
              onClick={handleSubmit}
              disabled={isSubmitting}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-all disabled:opacity-50 ${
                selectedAction === "abort"
                  ? "border-semantic-failBorder bg-semantic-failSubtle text-semantic-fail hover:brightness-110"
                  : "border-semantic-passBorder bg-semantic-passSubtle text-semantic-pass hover:brightness-110"
              }`}
            >
              {isSubmitting && (
                <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
              )}
              {isSubmitting ? "Submitting..." : "Confirm"}
            </button>
            <button
              onClick={handleCancel}
              disabled={isSubmitting}
              className="px-4 py-2 rounded-lg border border-terminal-border bg-transparent text-terminal-dim text-sm font-medium hover:text-terminal-white hover:border-terminal-mid transition-all disabled:opacity-50"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export function HITLFeedbackModal({
  isOpen,
  onClose,
  ...props
}: HITLFeedbackProps & { isOpen: boolean; onClose: () => void }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto bg-terminal-black border border-terminal-border rounded-xl p-6 shadow-2xl scrollbar-terminal">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-terminal-dim hover:text-terminal-white transition-colors"
          aria-label="Close"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        <HITLFeedback {...props} />
      </div>
    </div>
  );
}

