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
    <div className="p-4 border-b border-dashed border-terminal-borderDotted">
      <div className="font-mono text-[0.62rem] text-terminal-dim uppercase tracking-wider mb-2.5">
        {"// query_input"}
      </div>
      <form onSubmit={handleSubmit}>
        <div className="border border-dashed border-terminal-borderDotted rounded-sm p-2.5 mb-2.5 transition-all duration-200 focus-within:border-terminal-mid">
          <textarea
            ref={textareaRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="enter your research query..."
            disabled={isLoading}
            rows={3}
            className="w-full bg-transparent text-terminal-white placeholder:text-terminal-dim resize-none outline-none font-mono text-[0.78rem] leading-relaxed disabled:opacity-50 scrollbar-terminal"
          />
        </div>
        <div className="flex items-center justify-between">
          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="px-3.5 py-1.5 border border-dashed border-terminal-borderDotted rounded-sm bg-transparent text-terminal-white text-[0.72rem] font-mono cursor-pointer transition-all duration-200 hover:border-terminal-mid hover:bg-white/[0.02] disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1.5"
          >
            {isLoading ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-terminal-white animate-pulse" />
                <span>running</span>
              </>
            ) : (
              <>
                <span>run</span>
                <span className="text-terminal-dim">-&gt;</span>
              </>
            )}
          </button>
          <span className="text-terminal-dim text-[0.6rem] font-mono">
            shift+enter for newline
          </span>
        </div>
      </form>
      <div className="mt-4">
        <div className="font-mono text-[0.58rem] text-terminal-dim mb-2">{"// examples"}</div>
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLE_QUERIES.map((example, i) => (
            <button
              key={i}
              onClick={() => handleExampleClick(example)}
              disabled={isLoading}
              className="px-2 py-1 border border-dashed border-terminal-borderDotted rounded-sm bg-transparent text-terminal-dim text-[0.66rem] font-mono cursor-pointer transition-all duration-200 hover:border-terminal-mid hover:text-terminal-white disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <span className="text-terminal-mid mr-1">$</span>
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
