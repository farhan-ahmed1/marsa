export function useAgentStream() {
	return { events: [], status: "idle" as const };
}
