import type { QuerySubmission, Report } from "./types";

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

export async function getReport(streamId: string): Promise<Report> {
	const response = await fetch(`${API_BASE}/api/query/${streamId}/report`);

	if (!response.ok) {
		throw new Error(`Failed to fetch report: ${response.status}`);
	}

	return response.json() as Promise<Report>;
}

export async function submitFeedback(
	streamId: string,
	action: "approve" | "dig_deeper" | "correct",
	detail?: string
): Promise<void> {
	const response = await fetch(`${API_BASE}/api/query/${streamId}/feedback`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({ action, detail }),
	});

	if (!response.ok) {
		throw new Error(`Failed to submit feedback: ${response.status}`);
	}
}
