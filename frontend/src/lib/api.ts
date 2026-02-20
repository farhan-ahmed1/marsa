import type { QuerySubmission, Report, ReportResponse, HITLCheckpoint, FeedbackAction } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function getHealth(): Promise<{ status: string }> {
	const response = await fetch(`${API_BASE}/api/health`);
	if (!response.ok) {
		throw new Error("Health check failed");
	}
	return response.json() as Promise<{ status: string }>;
}

export async function submitQuery(query: string): Promise<QuerySubmission> {
	const response = await fetch(`${API_BASE}/api/query`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({ query }),
	});

	if (!response.ok) {
		throw new Error(`Failed to submit query: ${response.status}`);
	}

	return response.json() as Promise<QuerySubmission>;
}

export async function getReport(
	streamId: string,
	options?: { maxRetries?: number; retryDelay?: number }
): Promise<ReportResponse> {
	const maxRetries = options?.maxRetries ?? 10;
	const baseDelay = options?.retryDelay ?? 500;

	for (let attempt = 0; attempt < maxRetries; attempt++) {
		const response = await fetch(`${API_BASE}/api/query/${streamId}/report`);

		if (response.ok) {
			return response.json() as Promise<ReportResponse>;
		}

		if (response.status === 425) {
			// Report not ready yet, retry with exponential backoff
			if (attempt < maxRetries - 1) {
				const delay = baseDelay * Math.pow(1.5, attempt);
				await new Promise((resolve) => setTimeout(resolve, delay));
				continue;
			}
			throw new Error("Report not ready after multiple attempts");
		}

		throw new Error(`Failed to fetch report: ${response.status}`);
	}

	throw new Error("Failed to fetch report: max retries exceeded");
}

export async function getHITLCheckpoint(streamId: string): Promise<HITLCheckpoint> {
	const response = await fetch(`${API_BASE}/api/query/${streamId}/checkpoint`);

	if (!response.ok) {
		throw new Error(`Failed to fetch checkpoint: ${response.status}`);
	}

	return response.json() as Promise<HITLCheckpoint>;
}

export async function submitFeedback(
	streamId: string,
	action: FeedbackAction,
	detail?: string
): Promise<void> {
	const body: Record<string, string> = { action };
	if (action === "dig_deeper" && detail) {
		body.topic = detail;
	} else if (action === "correct" && detail) {
		body.correction = detail;
	}

	const response = await fetch(`${API_BASE}/api/query/${streamId}/feedback`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify(body),
	});

	if (!response.ok) {
		// Try to get error details from response
		try {
			const errorData = await response.json();
			throw new Error(errorData.detail || `Failed to submit feedback: ${response.status}`);
		} catch (e) {
			if (e instanceof Error && e.message.includes("detail")) {
				throw e;
			}
			throw new Error(`Failed to submit feedback: ${response.status}`);
		}
	}
}
