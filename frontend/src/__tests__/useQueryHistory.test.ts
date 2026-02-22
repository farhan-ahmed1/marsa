import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useQueryHistory } from "@/hooks/useQueryHistory";

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { store = {}; }),
    get length() { return Object.keys(store).length; },
    key: vi.fn((i: number) => Object.keys(store)[i] ?? null),
  };
})();

beforeEach(() => {
  Object.defineProperty(window, "localStorage", { value: localStorageMock, writable: true });
  localStorageMock.clear();
  vi.clearAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useQueryHistory", () => {
  it("starts with empty history", () => {
    const { result } = renderHook(() => useQueryHistory());
    // After useEffect runs, history should be empty (localStorage is empty)
    expect(result.current.history).toEqual([]);
  });

  it("loads existing history from localStorage", () => {
    const existing = [
      { id: "1", query: "test query", timestamp: "2025-01-01T00:00:00Z" },
    ];
    localStorageMock.setItem("marsa-query-history", JSON.stringify(existing));

    const { result } = renderHook(() => useQueryHistory());
    expect(result.current.history).toEqual(existing);
  });

  it("adds a new entry and persists to localStorage", () => {
    const { result } = renderHook(() => useQueryHistory());

    act(() => {
      result.current.addEntry("What is LangGraph?");
    });

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].query).toBe("What is LangGraph?");
    expect(localStorageMock.setItem).toHaveBeenCalledWith(
      "marsa-query-history",
      expect.any(String)
    );
  });

  it("deduplicates entries with same query text (case insensitive)", () => {
    const { result } = renderHook(() => useQueryHistory());

    act(() => {
      result.current.addEntry("hello world");
    });
    act(() => {
      result.current.addEntry("Hello World");
    });

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].query).toBe("Hello World");
  });

  it("puts newest entry first", () => {
    const { result } = renderHook(() => useQueryHistory());

    act(() => {
      result.current.addEntry("first");
    });
    act(() => {
      result.current.addEntry("second");
    });

    expect(result.current.history[0].query).toBe("second");
    expect(result.current.history[1].query).toBe("first");
  });

  it("removes an entry by id", () => {
    const { result } = renderHook(() => useQueryHistory());

    act(() => {
      result.current.addEntry("test query");
    });
    const id = result.current.history[0].id;

    act(() => {
      result.current.removeEntry(id);
    });

    expect(result.current.history).toHaveLength(0);
  });

  it("clears all history", () => {
    const { result } = renderHook(() => useQueryHistory());

    act(() => {
      result.current.addEntry("q1");
    });
    act(() => {
      result.current.addEntry("q2");
    });

    act(() => {
      result.current.clearHistory();
    });

    expect(result.current.history).toHaveLength(0);
    expect(localStorageMock.removeItem).toHaveBeenCalledWith("marsa-query-history");
  });

  it("caps history at MAX_ENTRIES (50)", () => {
    const { result } = renderHook(() => useQueryHistory());

    act(() => {
      for (let i = 0; i < 55; i++) {
        result.current.addEntry(`query ${i}`);
      }
    });

    expect(result.current.history.length).toBeLessThanOrEqual(50);
    // Most recent should be "query 54"
    expect(result.current.history[0].query).toBe("query 54");
  });

  it("handles corrupted localStorage gracefully", () => {
    localStorageMock.setItem("marsa-query-history", "not valid json {{{");

    const { result } = renderHook(() => useQueryHistory());
    expect(result.current.history).toEqual([]);
  });
});
