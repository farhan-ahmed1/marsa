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

// Get verdict styling
function getVerdictStyle(verdict: "supported" | "contradicted" | "unverifiable"): {
  icon: string;
  className: string;
  label: string;
} {
  switch (verdict) {
    case "supported":
      return { icon: "OK", className: "text-semantic-pass border-semantic-pass/40", label: "supported" };
    case "contradicted":
      return { icon: "!!", className: "text-semantic-fail border-semantic-fail/40", label: "contradicted" };
    case "unverifiable":
      return { icon: "??", className: "text-semantic-unknown border-semantic-unknown/40", label: "unverifiable" };
  }
}

// Claim card component
function ClaimCard({ claim }: { claim: VerificationResult }) {
  const verdictStyle = getVerdictStyle(claim.verdict);
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-dashed border-terminal-borderDotted rounded-sm p-3 bg-terminal-surface/30 animate-slide-in">
      <div className="flex items-start justify-between gap-3 mb-2">
        <p className="font-mono text-[0.74rem] text-terminal-white leading-relaxed flex-1">
          {claim.claim.statement}
        </p>
        <span
          className={`px-1.5 py-0.5 border rounded-sm text-[0.6rem] font-mono flex-shrink-0 ${verdictStyle.className}`}
        >
          {verdictStyle.icon} {verdictStyle.label}
        </span>
      </div>
      <div className="flex items-center gap-3 text-[0.64rem] font-mono">
        <span className="text-terminal-dim">
          confidence: <span className={claim.confidence >= 0.7 ? "text-semantic-pass" : claim.confidence >= 0.5 ? "text-semantic-unknown" : "text-semantic-fail"}>
            {Math.round(claim.confidence * 100)}%
          </span>
        </span>
        <span className="text-terminal-dim">|</span>
        <span className="text-terminal-dim">{claim.claim.category}</span>
        {claim.reasoning && (
          <>
            <span className="text-terminal-dim">|</span>
            <button
              type="button"
              onClick={() => setExpanded(!expanded)}
              className="text-terminal-mid hover:text-terminal-white transition-colors"
            >
              {expanded ? "hide reasoning" : "show reasoning"}
            </button>
          </>
        )}
      </div>
      {expanded && claim.reasoning && (
        <div className="mt-2.5 pt-2.5 border-t border-dashed border-terminal-borderDotted">
          <p className="font-mono text-[0.68rem] text-terminal-mid leading-relaxed">
            {claim.reasoning}
          </p>
          {claim.supporting_sources.length > 0 && (
            <div className="mt-2">
              <span className="text-[0.62rem] text-terminal-dim">Supporting: </span>
              <span className="text-[0.62rem] text-semantic-pass">
                {claim.supporting_sources.length} source(s)
              </span>
            </div>
          )}
          {claim.contradicting_sources.length > 0 && (
            <div className="mt-1">
              <span className="text-[0.62rem] text-terminal-dim">Contradicting: </span>
              <span className="text-[0.62rem] text-semantic-fail">
                {claim.contradicting_sources.length} source(s)
              </span>
            </div>
          )}
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
    if (action === "approve" || action === "abort") {
      // These don't need additional input
      setSelectedAction(action);
    } else {
      // dig_deeper and correct need input
      setSelectedAction(action);
    }
    setError(null);
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!selectedAction) return;

    // Validate input for actions that require it
    if ((selectedAction === "dig_deeper" || selectedAction === "correct") && !detailInput.trim()) {
      setError(selectedAction === "dig_deeper" ? "Please specify what to explore further" : "Please provide your correction");
      return;
    }

    try {
      await onSubmitFeedback(selectedAction, detailInput.trim() || undefined);
      setSelectedAction(null);
      setDetailInput("");
    } catch (err) {
      setError("Failed to submit feedback. Please try again.");
    }
  }, [selectedAction, detailInput, onSubmitFeedback]);

  const handleCancel = useCallback(() => {
    setSelectedAction(null);
    setDetailInput("");
    setError(null);
  }, []);

  // Calculate claim statistics
  const supportedCount = claims.filter((c) => c.verdict === "supported").length;
  const contradictedCount = claims.filter((c) => c.verdict === "contradicted").length;
  const unverifiableCount = claims.filter((c) => c.verdict === "unverifiable").length;

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-2.5 mb-5 pb-3 border-b border-dashed border-terminal-borderDotted">
        <span className="w-2 h-2 rounded-full bg-semantic-unknown animate-pulse" />
        <span className="font-mono text-[0.72rem] text-terminal-white uppercase tracking-wider">
          checkpoint: human review required
        </span>
      </div>

      {/* Summary */}
      <div className="px-3.5 py-3 rounded-sm border border-dashed border-terminal-borderDotted bg-terminal-surface/30 mb-5">
        <div className="font-mono text-[0.62rem] text-terminal-dim uppercase tracking-wider mb-2">
          {"// findings summary"}
        </div>
        <p className="font-mono text-[0.76rem] text-terminal-mid leading-relaxed">
          {summary || "The research phase has completed. Please review the claims below before proceeding to synthesis."}
        </p>
      </div>

      {/* Statistics */}
      <div className="flex gap-4 mb-5">
        <div className="flex items-center gap-2 px-3 py-2 border border-dashed border-terminal-borderDotted rounded-sm">
          <span className="text-semantic-pass text-[0.9rem]">OK</span>
          <div>
            <div className="font-mono text-[0.7rem] text-terminal-white">{supportedCount}</div>
            <div className="font-mono text-[0.58rem] text-terminal-dim">supported</div>
          </div>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 border border-dashed border-terminal-borderDotted rounded-sm">
          <span className="text-semantic-unknown text-[0.9rem]">??</span>
          <div>
            <div className="font-mono text-[0.7rem] text-terminal-white">{unverifiableCount}</div>
            <div className="font-mono text-[0.58rem] text-terminal-dim">unverifiable</div>
          </div>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 border border-dashed border-terminal-borderDotted rounded-sm">
          <span className="text-semantic-fail text-[0.9rem]">!!</span>
          <div>
            <div className="font-mono text-[0.7rem] text-terminal-white">{contradictedCount}</div>
            <div className="font-mono text-[0.58rem] text-terminal-dim">contradicted</div>
          </div>
        </div>
        <div className="flex items-center gap-2 px-3 py-2 border border-dashed border-terminal-borderDotted rounded-sm">
          <span className="text-terminal-mid text-[0.9rem]">Q</span>
          <div>
            <div className="font-mono text-[0.7rem] text-terminal-white">{Math.round(sourceQuality * 100)}%</div>
            <div className="font-mono text-[0.58rem] text-terminal-dim">source quality</div>
          </div>
        </div>
      </div>

      {/* Claims list */}
      <div className="mb-6">
        <div className="font-mono text-[0.62rem] text-terminal-dim uppercase tracking-wider mb-3">
          {"// verified claims"} ({claims.length})
        </div>
        <div className="space-y-2 max-h-[300px] overflow-y-auto scrollbar-terminal pr-1">
          {claims.map((claim, index) => (
            <ClaimCard key={index} claim={claim} />
          ))}
        </div>
      </div>

      {/* Action buttons */}
      {!selectedAction ? (
        <div className="space-y-3">
          <div className="font-mono text-[0.62rem] text-terminal-dim uppercase tracking-wider">
            {"// actions"}
          </div>
          <div className="flex flex-wrap gap-2.5">
            <button
              onClick={() => handleActionClick("approve")}
              disabled={isSubmitting}
              className="px-4 py-2 border border-semantic-pass/40 rounded-sm bg-semantic-pass/10 text-semantic-pass text-[0.74rem] font-mono cursor-pointer transition-all duration-200 hover:bg-semantic-pass/20 hover:border-semantic-pass/60 disabled:opacity-50"
            >
              approve & continue -&gt;
            </button>
            <button
              onClick={() => handleActionClick("dig_deeper")}
              disabled={isSubmitting}
              className="px-4 py-2 border border-dashed border-terminal-borderDotted rounded-sm bg-transparent text-terminal-white text-[0.74rem] font-mono cursor-pointer transition-all duration-200 hover:border-terminal-mid hover:bg-white/[0.02] disabled:opacity-50"
            >
              dig deeper
            </button>
            <button
              onClick={() => handleActionClick("correct")}
              disabled={isSubmitting}
              className="px-4 py-2 border border-dashed border-terminal-borderDotted rounded-sm bg-transparent text-terminal-white text-[0.74rem] font-mono cursor-pointer transition-all duration-200 hover:border-terminal-mid hover:bg-white/[0.02] disabled:opacity-50"
            >
              correct
            </button>
            <button
              onClick={() => handleActionClick("abort")}
              disabled={isSubmitting}
              className="px-4 py-2 border border-dashed border-semantic-fail/40 rounded-sm bg-transparent text-semantic-fail text-[0.74rem] font-mono cursor-pointer transition-all duration-200 hover:bg-semantic-fail/10 disabled:opacity-50"
            >
              abort
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-3 animate-slide-in">
          {/* Action confirmation or input */}
          {(selectedAction === "dig_deeper" || selectedAction === "correct") && (
            <>
              <div className="font-mono text-[0.68rem] text-terminal-white">
                {selectedAction === "dig_deeper"
                  ? "What topic would you like to explore further?"
                  : "What corrections would you like to provide?"}
              </div>
              <textarea
                value={detailInput}
                onChange={(e) => setDetailInput(e.target.value)}
                placeholder={
                  selectedAction === "dig_deeper"
                    ? "e.g., performance benchmarks, memory usage comparisons..."
                    : "e.g., The claim about X is incorrect because..."
                }
                rows={3}
                className="w-full bg-terminal-surface border border-dashed border-terminal-borderDotted rounded-sm p-2.5 text-terminal-white placeholder:text-terminal-dim resize-none outline-none font-mono text-[0.76rem] focus:border-terminal-mid transition-colors"
              />
            </>
          )}

          {(selectedAction === "approve" || selectedAction === "abort") && (
            <div className="font-mono text-[0.72rem] text-terminal-mid">
              {selectedAction === "approve"
                ? "Confirm: Continue to synthesis with current findings?"
                : "Confirm: Abort and discard research?"}
            </div>
          )}

          {error && (
            <div className="font-mono text-[0.68rem] text-semantic-fail animate-fade-in">
              {error}
            </div>
          )}

          <div className="flex gap-2.5">
            <button
              onClick={handleSubmit}
              disabled={isSubmitting}
              className={`px-4 py-2 border rounded-sm text-[0.74rem] font-mono cursor-pointer transition-all duration-200 disabled:opacity-50 flex items-center gap-2 ${
                selectedAction === "abort"
                  ? "border-semantic-fail/40 bg-semantic-fail/10 text-semantic-fail hover:bg-semantic-fail/20"
                  : "border-semantic-pass/40 bg-semantic-pass/10 text-semantic-pass hover:bg-semantic-pass/20"
              }`}
            >
              {isSubmitting && (
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
              )}
              {isSubmitting ? "submitting..." : "confirm"}
            </button>
            <button
              onClick={handleCancel}
              disabled={isSubmitting}
              className="px-4 py-2 border border-dashed border-terminal-borderDotted rounded-sm bg-transparent text-terminal-dim text-[0.74rem] font-mono cursor-pointer transition-all duration-200 hover:text-terminal-white disabled:opacity-50"
            >
              cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Compact inline version for overlay/modal
export function HITLFeedbackModal({
  isOpen,
  onClose,
  ...props
}: HITLFeedbackProps & { isOpen: boolean; onClose: () => void }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-terminal-black/80 backdrop-blur-sm animate-fade-in">
      <div className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto bg-terminal-black border border-dashed border-terminal-borderDotted rounded-sm p-6 shadow-2xl scrollbar-terminal">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-terminal-dim hover:text-terminal-white transition-colors text-lg"
          aria-label="Close"
        >
          x
        </button>
        <HITLFeedback {...props} />
      </div>
    </div>
  );
}
