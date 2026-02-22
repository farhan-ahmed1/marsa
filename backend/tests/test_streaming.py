"""Unit tests for the SSE streaming module.

Tests cover EventQueueManager, trace_event_to_sse conversion,
and SSE event streaming without real workflow execution.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from api.models import SSEEvent, SSEEventType  # noqa: E402
from api.streaming import (  # noqa: E402
    EventQueueManager,
    create_sse_response,
    stream_agent_events,
    trace_event_to_sse,
)
from graph.state import AgentName, TraceEvent  # noqa: E402


# ---------------------------------------------------------------------------
# EventQueueManager Tests
# ---------------------------------------------------------------------------


class TestEventQueueManager:
    """Tests for the EventQueueManager class."""

    async def test_create_stream(self):
        manager = EventQueueManager()
        queue = await manager.create_stream("test-stream-1")
        assert queue is not None
        assert isinstance(queue, asyncio.Queue)
        assert manager.stream_exists("test-stream-1")

    async def test_create_stream_duplicate(self):
        manager = EventQueueManager()
        _queue1 = await manager.create_stream("dup-stream")
        queue2 = await manager.create_stream("dup-stream")
        # Should create a new queue but still exist
        assert queue2 is not None
        assert manager.stream_exists("dup-stream")

    async def test_get_queue(self):
        manager = EventQueueManager()
        await manager.create_stream("q-stream")
        queue = await manager.get_queue("q-stream")
        assert queue is not None

    async def test_get_queue_nonexistent(self):
        manager = EventQueueManager()
        queue = await manager.get_queue("nonexistent")
        assert queue is None

    async def test_push_event(self):
        manager = EventQueueManager()
        queue = await manager.create_stream("push-stream")
        event = SSEEvent(type=SSEEventType.AGENT_STARTED, data={"agent": "planner"})
        result = await manager.push_event("push-stream", event)
        assert result is True
        assert not queue.empty()

    async def test_push_event_unknown_stream(self):
        manager = EventQueueManager()
        event = SSEEvent(type=SSEEventType.ERROR, data={"error": "test"})
        result = await manager.push_event("unknown-stream", event)
        assert result is False

    async def test_update_and_get_state(self):
        manager = EventQueueManager()
        await manager.create_stream("state-stream")
        state = {"query": "test", "status": "planning"}
        await manager.update_state("state-stream", state)
        retrieved = await manager.get_state("state-stream")
        assert retrieved["query"] == "test"

    async def test_get_state_nonexistent(self):
        manager = EventQueueManager()
        state = await manager.get_state("no-state")
        assert state is None

    async def test_set_and_get_workflow_task(self):
        manager = EventQueueManager()
        await manager.create_stream("task-stream")

        mock_task = MagicMock(spec=asyncio.Task)
        manager.set_workflow_task("task-stream", mock_task)
        retrieved = manager.get_workflow_task("task-stream")
        assert retrieved is mock_task

    async def test_get_workflow_task_nonexistent(self):
        manager = EventQueueManager()
        task = manager.get_workflow_task("no-task")
        assert task is None

    async def test_cleanup_stream(self):
        manager = EventQueueManager()
        await manager.create_stream("cleanup-stream")
        await manager.cleanup_stream("cleanup-stream")
        queue = await manager.get_queue("cleanup-stream")
        assert queue is None
        # State should be preserved for report retrieval
        assert manager.get_workflow_task("cleanup-stream") is None

    async def test_stream_exists(self):
        manager = EventQueueManager()
        assert not manager.stream_exists("nope")
        await manager.create_stream("exists-stream")
        assert manager.stream_exists("exists-stream")

    async def test_stream_exists_via_state(self):
        """Stream exists if state is stored even after queue cleanup."""
        manager = EventQueueManager()
        await manager.create_stream("state-only")
        await manager.update_state("state-only", {"query": "test"})
        await manager.cleanup_stream("state-only")
        # State is preserved, so stream still "exists"
        assert manager.stream_exists("state-only")

    async def test_get_active_streams(self):
        manager = EventQueueManager()
        await manager.create_stream("s1")
        await manager.create_stream("s2")
        active = manager.get_active_streams()
        assert "s1" in active
        assert "s2" in active

    async def test_get_active_streams_empty(self):
        manager = EventQueueManager()
        assert manager.get_active_streams() == []


# ---------------------------------------------------------------------------
# trace_event_to_sse Tests
# ---------------------------------------------------------------------------


class TestTraceEventToSSE:
    """Tests for converting TraceEvent to SSEEvent."""

    def test_start_action_maps_to_agent_started(self):
        trace = TraceEvent(
            agent=AgentName.PLANNER,
            action="start",
            detail="Planning started",
        )
        sse = trace_event_to_sse(trace, "stream-123")
        assert sse.type == SSEEventType.AGENT_STARTED
        assert sse.data["agent"] == "planner"
        assert sse.stream_id == "stream-123"

    def test_complete_action_maps_to_agent_completed(self):
        trace = TraceEvent(
            agent=AgentName.SYNTHESIZER,
            action="complete",
            detail="Report generated",
            tokens_used=1500,
            latency_ms=2000.0,
        )
        sse = trace_event_to_sse(trace, "s1")
        assert sse.type == SSEEventType.AGENT_COMPLETED
        assert sse.data["tokens_used"] == 1500
        assert sse.data["latency_ms"] == 2000.0

    def test_tool_call_action(self):
        trace = TraceEvent(
            agent=AgentName.RESEARCHER,
            action="tool_call",
            detail="Searching web",
            metadata={"tool": "tavily"},
        )
        sse = trace_event_to_sse(trace, "s2")
        assert sse.type == SSEEventType.TOOL_CALLED
        assert sse.data["metadata"]["tool"] == "tavily"

    def test_claim_extracted_action(self):
        trace = TraceEvent(
            agent=AgentName.RESEARCHER,
            action="claim_extracted",
            detail="Extracted 5 claims",
        )
        sse = trace_event_to_sse(trace, "s3")
        assert sse.type == SSEEventType.CLAIM_EXTRACTED

    def test_claim_verified_action(self):
        trace = TraceEvent(
            agent=AgentName.FACT_CHECKER,
            action="claim_verified",
            detail="Claim verified",
        )
        sse = trace_event_to_sse(trace, "s4")
        assert sse.type == SSEEventType.CLAIM_VERIFIED

    def test_error_action(self):
        trace = TraceEvent(
            agent=AgentName.PLANNER,
            action="error",
            detail="Planning failed",
        )
        sse = trace_event_to_sse(trace, "s5")
        assert sse.type == SSEEventType.ERROR

    def test_unknown_action_defaults_to_tool_result(self):
        trace = TraceEvent(
            agent=AgentName.RESEARCHER,
            action="unknown_action",
            detail="Something happened",
        )
        sse = trace_event_to_sse(trace, "s6")
        assert sse.type == SSEEventType.TOOL_RESULT

    def test_report_generating_action(self):
        trace = TraceEvent(
            agent=AgentName.SYNTHESIZER,
            action="report_generating",
            detail="Generating report",
        )
        sse = trace_event_to_sse(trace, "s7")
        assert sse.type == SSEEventType.REPORT_GENERATING


# ---------------------------------------------------------------------------
# stream_agent_events Tests
# ---------------------------------------------------------------------------


class TestStreamAgentEvents:
    """Tests for SSE event streaming generator."""

    async def test_stream_unknown_stream_yields_error(self):
        """Non-existent stream with no state should yield error."""

        # Make sure the stream doesn't exist
        stream_id = "totally-unknown-stream"
        events = []
        async for event_str in stream_agent_events(stream_id):
            events.append(event_str)
        assert len(events) == 1
        assert "stream_not_found" in events[0]

    async def test_stream_completed_state_yields_complete(self):
        """Stream with stored state but no queue yields complete event."""
        manager = EventQueueManager()
        # Directly set state without queue
        manager._states["completed-stream"] = {
            "status": "completed",
            "report": "Test report",
            "report_structured": None,
            "claims": [],
            "verification_results": [],
            "citations": [],
            "agent_trace": [],
            "iteration_count": 1,
        }

        events = []
        # We need to use the manager's internal queue system
        # The module-level event_queue_manager won't have our data,
        # so we patch it
        with patch("api.streaming.event_queue_manager", manager):
            async for event_str in stream_agent_events("completed-stream"):
                events.append(event_str)
        assert len(events) == 1
        assert "complete" in events[0].lower() or "COMPLETE" in events[0]

    async def test_stream_with_queue_yields_events(self):
        """Stream with active queue should yield events."""
        manager = EventQueueManager()
        queue = await manager.create_stream("active-stream")

        # Pre-populate queue with events
        await queue.put(SSEEvent(
            type=SSEEventType.AGENT_STARTED,
            data={"agent": "planner"},
            stream_id="active-stream",
        ))
        await queue.put(SSEEvent(
            type=SSEEventType.COMPLETE,
            data={"status": "completed"},
            stream_id="active-stream",
        ))

        events = []
        with patch("api.streaming.event_queue_manager", manager):
            async for event_str in stream_agent_events("active-stream"):
                events.append(event_str)

        # Should get: connected + agent_started + complete
        assert len(events) == 3
        assert "connected" in events[0].lower()


# ---------------------------------------------------------------------------
# create_sse_response Tests
# ---------------------------------------------------------------------------


class TestCreateSSEResponse:
    """Tests for SSE response creation."""

    def test_creates_streaming_response(self):
        response = create_sse_response("test-id")
        assert response.media_type == "text/event-stream"
        assert response.headers.get("Cache-Control") == "no-cache"
        assert response.headers.get("Connection") == "keep-alive"
