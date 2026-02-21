"use client";

import { useState, useRef, useEffect } from "react";
import type { StreamStatus } from "@/lib/types";

const EXAMPLE_QUERIES = [
  "Rust vs Go for distributed systems",
  "Latest AI agent framework trends",
  "CAP theorem with examples",
];

interface QueryInputProps {
  onSubmit: (query: string) => void;
  status: StreamStatus;
}

export function QueryInput({ onSubmit, status }: QueryInputProps) {
  const [query, setQuery] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isLoading = status === "running" || status === "synth" || status === "hitl";

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
    }
  }, [query]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit(query.trim());
    }
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="p-4 border-b border-terminal-border flex-shrink-0">
      <form onSubmit={handleSubmit}>
        {/* Textarea with focus ring */}
        <div className="relative rounded-lg border border-terminal-border bg-terminal-surface focus-within:border-accent/60 focus-within:ring-2 focus-within:ring-accent/20 transition-all">
          <textarea
            ref={textareaRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a research question..."
            disabled={isLoading}
            rows={3}
            className="w-full bg-transparent text-terminal-white placeholder:text-terminal-dim resize-none outline-none text-sm leading-relaxed disabled:opacity-50 px-3.5 py-3 font-mono"
          />
          {/* Submit button inside textarea */}
          <div className="flex items-center justify-between px-3 pb-2.5">
            <span className="text-xs text-terminal-dim">⏎ to submit · Shift+⏎ for newline</span>
            <button
              type="submit"
              disabled={!query.trim() || isLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-accent text-white text-sm font-medium transition-all hover:bg-accent-hover disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <span className="w-3.5 h-3.5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                  <span>Running</span>
                </>
              ) : (
                <>
                  <span>Research</span>
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                  </svg>
                </>
              )}
            </button>
          </div>
        </div>
      </form>

      {/* Example queries */}
      <div className="mt-3">
        <p className="text-xs text-terminal-dim mb-2">Try an example:</p>
        <div className="flex flex-col gap-1.5">
          {EXAMPLE_QUERIES.map((example, i) => (
            <button
              key={i}
              onClick={() => handleExampleClick(example)}
              disabled={isLoading}
              className="text-left px-3 py-2 rounded-md border border-terminal-border bg-transparent text-sm text-terminal-mid transition-all hover:border-terminal-mid hover:text-terminal-white hover:bg-terminal-surface disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
