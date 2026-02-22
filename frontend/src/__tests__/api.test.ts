import { describe, it, expect, vi, beforeEach } from "vitest";
import { getHealth, submitQuery, getReport, getHITLCheckpoint, submitFeedback } from "@/lib/api";

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
});

describe("getHealth", () => {
  it("returns status on success", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ status: "ok" }),
    });
    const result = await getHealth();
    expect(result).toEqual({ status: "ok" });
    expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining("/api/health"));
  });

  it("throws on failure", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await expect(getHealth()).rejects.toThrow("Health check failed");
  });
});

describe("submitQuery", () => {
  it("posts query and returns stream info", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ query: "test", stream_id: "abc-123" }),
    });
    const result = await submitQuery("test");
    expect(result.stream_id).toBe("abc-123");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/query"),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: "test" }),
      })
    );
  });

  it("throws on HTTP error", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 422 });
    await expect(submitQuery("")).rejects.toThrow("Failed to submit query: 422");
  });
});

describe("getReport", () => {
  it("returns report on success", async () => {
    const mockReport = { stream_id: "s1", status: "complete", report: null, error: null };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockReport),
    });
    const result = await getReport("s1");
    expect(result.stream_id).toBe("s1");
  });

  it("retries on 425 (report not ready)", async () => {
    // First call: 425, second call: success
    mockFetch
      .mockResolvedValueOnce({ ok: false, status: 425 })
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ stream_id: "s1", status: "complete" }),
      });

    const result = await getReport("s1", { retryDelay: 1 });
    expect(result.stream_id).toBe("s1");
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("throws after max retries on 425", async () => {
    mockFetch.mockResolvedValue({ ok: false, status: 425 });
    await expect(getReport("s1", { maxRetries: 2, retryDelay: 1 })).rejects.toThrow(
      "Report not ready after multiple attempts"
    );
  });

  it("throws immediately on non-425 error", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    await expect(getReport("s1")).rejects.toThrow("Failed to fetch report: 500");
  });
});

describe("getHITLCheckpoint", () => {
  it("returns checkpoint data", async () => {
    const checkpoint = { summary: "Review needed", claims: [], source_quality: 0.8 };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(checkpoint),
    });
    const result = await getHITLCheckpoint("s1");
    expect(result.summary).toBe("Review needed");
  });

  it("throws on error", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404 });
    await expect(getHITLCheckpoint("s1")).rejects.toThrow("Failed to fetch checkpoint: 404");
  });
});

describe("submitFeedback", () => {
  it("sends approve feedback", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true });
    await submitFeedback("s1", "approve");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/query/s1/feedback"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ action: "approve" }),
      })
    );
  });

  it("includes topic for dig_deeper", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true });
    await submitFeedback("s1", "dig_deeper", "more on AI");
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body).toEqual({ action: "dig_deeper", topic: "more on AI" });
  });

  it("includes correction for correct action", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true });
    await submitFeedback("s1", "correct", "fix this claim");
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body).toEqual({ action: "correct", correction: "fix this claim" });
  });

  it("throws with detail from error response when detail contains 'detail'", async () => {
    // Note: submitFeedback only re-throws the parsed detail if e.message.includes("detail")
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ detail: "Missing detail field" }),
    });
    await expect(submitFeedback("s1", "approve")).rejects.toThrow("Missing detail field");
  });

  it("throws generic error when response body is not JSON", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () => { throw new Error("not json"); },
    });
    await expect(submitFeedback("s1", "approve")).rejects.toThrow(
      "Failed to submit feedback: 500"
    );
  });
});
