import asyncio
from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse


async def stream_events() -> AsyncGenerator[str, None]:
	while True:
		await asyncio.sleep(1)
		yield "data: {\"type\": \"heartbeat\"}\n\n"


def sse_response() -> StreamingResponse:
	return StreamingResponse(stream_events(), media_type="text/event-stream")
