export async function getHealth(): Promise<{ status: string }> {
	const response = await fetch("http://localhost:8000/api/health");
	if (!response.ok) {
		throw new Error("Health check failed");
	}
	return response.json() as Promise<{ status: string }>;
}
