"use client";

import { useState, useCallback, useEffect } from "react";

export interface HistoryEntry {
  id: string;
  query: string;
  timestamp: string; // ISO string
}

const STORAGE_KEY = "marsa-query-history";
const MAX_ENTRIES = 50;

function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as HistoryEntry[];
  } catch {
    return [];
  }
}

function saveHistory(entries: HistoryEntry[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // ignore quota errors
  }
}

export function useQueryHistory() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  // Load on mount (client-side only)
  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  const addEntry = useCallback((query: string) => {
    setHistory((prev) => {
      // Deduplicate: remove any existing entry with the same query text
      const deduped = prev.filter(
        (e) => e.query.trim().toLowerCase() !== query.trim().toLowerCase()
      );
      const next: HistoryEntry[] = [
        {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          query: query.trim(),
          timestamp: new Date().toISOString(),
        },
        ...deduped,
      ].slice(0, MAX_ENTRIES);
      saveHistory(next);
      return next;
    });
  }, []);

  const removeEntry = useCallback((id: string) => {
    setHistory((prev) => {
      const next = prev.filter((e) => e.id !== id);
      saveHistory(next);
      return next;
    });
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {}
  }, []);

  return { history, addEntry, removeEntry, clearHistory };
}
