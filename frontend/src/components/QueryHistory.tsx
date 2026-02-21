"use client";

import type { HistoryEntry } from "@/hooks/useQueryHistory";

interface QueryHistoryProps {
  entries: HistoryEntry[];
  onSelect: (query: string) => void;
  onRemove: (id: string) => void;
  onClear: () => void;
}

function formatRelative(iso: string): string {
  const now = Date.now();
  const diff = now - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export function QueryHistory({
  entries,
  onSelect,
  onRemove,
  onClear,
}: QueryHistoryProps) {
  if (entries.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
        <svg
          className="w-8 h-8 text-terminal-border mb-3"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.5}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 6v6l4 2m6-2a10 10 0 11-20 0 10 10 0 0120 0z"
          />
        </svg>
        <p className="text-terminal-dim text-xs">No history yet</p>
        <p className="text-terminal-dim/60 text-xs mt-1">
          Past queries will appear here
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-terminal-border flex-shrink-0">
        <span className="text-xs font-semibold text-terminal-dim uppercase tracking-wider">
          History
        </span>
        <button
          type="button"
          onClick={onClear}
          className="text-xs text-terminal-dim hover:text-terminal-mid transition-colors px-1"
          title="Clear all history"
        >
          Clear
        </button>
      </div>

      {/* Entries */}
      <ul className="flex-1 overflow-y-auto divide-y divide-terminal-border/50">
        {entries.map((entry) => (
          <li
            key={entry.id}
            className="group flex items-start gap-2 px-4 py-2.5 hover:bg-terminal-surface transition-colors"
          >
            {/* Query text — clickable */}
            <button
              type="button"
              onClick={() => onSelect(entry.query)}
              className="flex-1 text-left min-w-0"
            >
              <p className="text-terminal-white text-xs leading-snug line-clamp-2 group-hover:text-terminal-pure transition-colors">
                {entry.query}
              </p>
              <p className="text-terminal-dim text-xs mt-0.5">
                {formatRelative(entry.timestamp)}
              </p>
            </button>

            {/* Remove button */}
            <button
              type="button"
              onClick={() => onRemove(entry.id)}
              className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity text-terminal-dim hover:text-terminal-mid mt-0.5"
              title="Remove"
            >
              <svg
                className="w-3.5 h-3.5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
