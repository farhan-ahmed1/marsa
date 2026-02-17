"""SSE streaming utilities for real-time agent trace events.

This module provides the infrastructure for streaming LangGraph workflow
events to the frontend in real-time using Server-Sent Events (SSE).
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Optional

import structlog
from fastapi.responses import StreamingResponse

from api.models import SSEEvent, SSEEventType
from graph.state import (
    AgentState,
    PipelineStatus,
    TraceEvent,
    create_initial_state,
)
from graph.workflow import create_workflow

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Event Queue Manager
# ---------------------------------------------------------------------------


class EventQueueManager:
    """Manages event queues for active query streams.
    
    Each active query session has its own asyncio.Queue for SSE events.
    The workflow pushes events to the queue, and the SSE endpoint
    consumes them.
    
    Attributes:
        _queues: Mapping of stream_id to event queues.
        _states: Mapping of stream_id to current workflow state.
        _workflows: Mapping of stream_id to running workflow tasks.
    """
    
    def __init__(self):
        self._queues: dict[str, asyncio.Queue[SSEEvent]] = {}
        self._states: dict[str, AgentState] = {}
        self._workflows: dict[str, Optional[asyncio.Task]] = {}
        self._lock = asyncio.Lock()
    
    async def create_stream(self, stream_id: str) -> asyncio.Queue[SSEEvent]:
        """Create a new event queue for a stream.
        
        Args:
            stream_id: Unique identifier for this stream.
            
        Returns:
            The event queue for this stream.
        """
        async with self._lock:
            if stream_id in self._queues:
                logger.warning("stream_already_exists", stream_id=stream_id)
            
            queue: asyncio.Queue[SSEEvent] = asyncio.Queue()
            self._queues[stream_id] = queue
            self._states[stream_id] = {}
            self._workflows[stream_id] = None
            
            logger.info("stream_created", stream_id=stream_id)
            return queue
    
    async def get_queue(self, stream_id: str) -> Optional[asyncio.Queue[SSEEvent]]:
        """Get the event queue for a stream.
        
        Args:
            stream_id: Stream identifier.
            
        Returns:
            The queue if it exists, None otherwise.
        """
        return self._queues.get(stream_id)
    
    async def push_event(self, stream_id: str, event: SSEEvent) -> bool:
        """Push an event to a stream's queue.
        
        Args:
            stream_id: Target stream.
            event: Event to push.
            
        Returns:
            True if successful, False if stream not found.
        """
        queue = self._queues.get(stream_id)
        if queue is None:
            logger.warning("push_to_unknown_stream", stream_id=stream_id)
            return False
        
        event.stream_id = stream_id
        await queue.put(event)
        return True
    
    async def update_state(self, stream_id: str, state: AgentState) -> None:
        """Update the stored state for a stream.
        
        Args:
            stream_id: Stream identifier.
            state: Current workflow state.
        """
        self._states[stream_id] = state
    
    async def get_state(self, stream_id: str) -> Optional[AgentState]:
        """Get the current state for a stream.
        
        Args:
            stream_id: Stream identifier.
            
        Returns:
            The current state if available.
        """
        return self._states.get(stream_id)
    
    def set_workflow_task(self, stream_id: str, task: asyncio.Task) -> None:
        """Store the workflow task for a stream.
        
        Args:
            stream_id: Stream identifier.
            task: The running workflow task.
        """
        self._workflows[stream_id] = task
    
    def get_workflow_task(self, stream_id: str) -> Optional[asyncio.Task]:
        """Get the workflow task for a stream.
        
        Args:
            stream_id: Stream identifier.
            
        Returns:
            The task if it exists.
        """
        return self._workflows.get(stream_id)
    
    async def cleanup_stream(self, stream_id: str) -> None:
        """Clean up resources for a completed stream.
        
        Args:
            stream_id: Stream to clean up.
        """
        async with self._lock:
            self._queues.pop(stream_id, None)
            # Keep state for report retrieval
            # self._states.pop(stream_id, None)
            self._workflows.pop(stream_id, None)
            logger.info("stream_cleaned_up", stream_id=stream_id)
    
    def stream_exists(self, stream_id: str) -> bool:
        """Check if a stream exists.
        
        Args:
            stream_id: Stream identifier.
            
        Returns:
            True if the stream exists.
        """
        return stream_id in self._queues or stream_id in self._states
    
    def get_active_streams(self) -> list[str]:
        """Get list of active stream IDs.
        
        Returns:
            List of stream IDs with active queues.
        """
        return list(self._queues.keys())


# Global event queue manager
event_queue_manager = EventQueueManager()


# ---------------------------------------------------------------------------
# SSE Event Streaming
# ---------------------------------------------------------------------------


async def stream_agent_events(stream_id: str) -> AsyncGenerator[str, None]:
    """Stream SSE events for a query session.
    
    This generator yields SSE-formatted events from the event queue.
    It handles connection setup, heartbeats, and graceful completion.
    
    Args:
        stream_id: The query session to stream events for.
        
    Yields:
        SSE-formatted event strings.
    """
    queue = await event_queue_manager.get_queue(stream_id)
    
    if queue is None:
        # Check if state exists (stream completed)
        state = await event_queue_manager.get_state(stream_id)
        if state:
            # Send final state as complete event
            yield SSEEvent(
                type=SSEEventType.COMPLETE,
                data={"status": state.get("status", "completed")},
                stream_id=stream_id,
            ).to_sse_format()
            return
        else:
            yield SSEEvent(
                type=SSEEventType.ERROR,
                data={"error": "stream_not_found", "message": f"Stream {stream_id} not found"},
                stream_id=stream_id,
            ).to_sse_format()
            return
    
    # Send connection confirmation
    yield SSEEvent(
        type=SSEEventType.CONNECTED,
        data={"stream_id": stream_id},
        stream_id=stream_id,
    ).to_sse_format()
    
    # Stream events from queue
    heartbeat_interval = 15.0  # seconds
    
    while True:
        try:
            # Wait for event with timeout (for heartbeat)
            event = await asyncio.wait_for(
                queue.get(),
                timeout=heartbeat_interval
            )
            
            yield event.to_sse_format()
            
            # Check if this is a terminal event
            if event.type in (SSEEventType.COMPLETE, SSEEventType.ERROR):
                logger.info(
                    "stream_ended",
                    stream_id=stream_id,
                    event_type=event.type.value,
                )
                break
                
        except asyncio.TimeoutError:
            # Send heartbeat to keep connection alive
            yield SSEEvent(
                type=SSEEventType.HEARTBEAT,
                data={"elapsed_since_last": heartbeat_interval},
                stream_id=stream_id,
            ).to_sse_format()
            
        except asyncio.CancelledError:
            logger.info("stream_cancelled", stream_id=stream_id)
            break
            
        except Exception as e:
            logger.error("stream_error", stream_id=stream_id, error=str(e))
            yield SSEEvent(
                type=SSEEventType.ERROR,
                data={"error": "stream_error", "message": str(e)},
                stream_id=stream_id,
            ).to_sse_format()
            break


def create_sse_response(stream_id: str) -> StreamingResponse:
    """Create an SSE StreamingResponse for a query session.
    
    Args:
        stream_id: The query session to stream.
        
    Returns:
        FastAPI StreamingResponse configured for SSE.
    """
    return StreamingResponse(
        stream_agent_events(stream_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# Workflow Event Bridge
# ---------------------------------------------------------------------------


def trace_event_to_sse(trace: TraceEvent, stream_id: str) -> SSEEvent:
    """Convert a TraceEvent to an SSEEvent.
    
    Maps internal trace events to SSE event types for the frontend.
    
    Args:
        trace: The internal trace event.
        stream_id: Associated stream ID.
        
    Returns:
        SSEEvent for streaming.
    """
    # Map trace actions to SSE event types
    action_mapping = {
        "start": SSEEventType.AGENT_STARTED,
        "tool_call": SSEEventType.TOOL_CALLED,
        "tool_result": SSEEventType.TOOL_RESULT,
        "claim_extracted": SSEEventType.CLAIM_EXTRACTED,
        "claim_verified": SSEEventType.CLAIM_VERIFIED,
        "report_generating": SSEEventType.REPORT_GENERATING,
        "complete": SSEEventType.COMPLETE,
        "error": SSEEventType.ERROR,
    }
    
    event_type = action_mapping.get(trace.action, SSEEventType.TOOL_RESULT)
    
    return SSEEvent(
        type=event_type,
        data={
            "agent": trace.agent.value if hasattr(trace.agent, "value") else str(trace.agent),
            "action": trace.action,
            "detail": trace.detail,
            "latency_ms": trace.latency_ms,
            "tokens_used": trace.tokens_used,
            "metadata": trace.metadata,
        },
        timestamp=trace.timestamp,
        stream_id=stream_id,
    )


async def run_workflow_with_streaming(
    stream_id: str,
    query: str,
    enable_hitl: bool = False,
    enable_parallel: bool = True,
) -> AgentState:
    """Run the research workflow while streaming events.
    
    This function runs the LangGraph workflow and pushes trace events
    to the SSE queue as they occur.
    
    Args:
        stream_id: The stream to push events to.
        query: The research query.
        enable_hitl: Whether to enable human-in-the-loop.
        enable_parallel: Whether to enable parallel execution.
        
    Returns:
        Final workflow state.
    """
    # Create initial state
    initial_state = create_initial_state(query)
    
    # Create workflow
    workflow = create_workflow(
        enable_hitl=enable_hitl,
        enable_parallel=enable_parallel,
        use_memory_checkpointer=True,
    )
    
    # Config for checkpointing
    config = {
        "configurable": {
            "thread_id": stream_id,
        }
    }
    
    logger.info(
        "workflow_starting",
        stream_id=stream_id,
        query=query[:50],
        enable_hitl=enable_hitl,
        enable_parallel=enable_parallel,
    )
    
    # Push initial event
    await event_queue_manager.push_event(
        stream_id,
        SSEEvent(
            type=SSEEventType.AGENT_STARTED,
            data={
                "agent": "system",
                "action": "start",
                "detail": f"Starting research for: {query[:100]}...",
                "query": query,
            },
        ),
    )
    
    last_trace_count = 0
    final_state = None
    
    try:
        # Stream workflow events
        async for event in workflow.astream_events(
            initial_state,
            config,
            version="v2",
        ):
            # Process LangGraph events
            event_kind = event.get("event", "")
            
            if event_kind == "on_chain_end":
                # Check for updated state with new trace events
                output = event.get("data", {}).get("output")
                if isinstance(output, dict):
                    trace_events = output.get("agent_trace", [])
                    
                    # Push new trace events
                    for trace in trace_events[last_trace_count:]:
                        sse_event = trace_event_to_sse(trace, stream_id)
                        await event_queue_manager.push_event(stream_id, sse_event)
                    
                    last_trace_count = len(trace_events)
                    
                    # Update stored state
                    await event_queue_manager.update_state(stream_id, output)
                    final_state = output
                    
                    # Check for HITL checkpoint
                    status = output.get("status")
                    if status == PipelineStatus.AWAITING_FEEDBACK.value:
                        await event_queue_manager.push_event(
                            stream_id,
                            SSEEvent(
                                type=SSEEventType.HITL_CHECKPOINT,
                                data={
                                    "status": "awaiting_feedback",
                                    "message": "Review required before proceeding",
                                    "claims_count": len(output.get("claims", [])),
                                    "verified_count": len(output.get("verification_results", [])),
                                },
                            ),
                        )
    
    except Exception as e:
        logger.error("workflow_error", stream_id=stream_id, error=str(e))
        await event_queue_manager.push_event(
            stream_id,
            SSEEvent(
                type=SSEEventType.ERROR,
                data={"error": "workflow_error", "message": str(e)},
            ),
        )
        raise
    
    # If we completed successfully without HITL interrupt
    if final_state:
        status = final_state.get("status", PipelineStatus.COMPLETED.value)
        
        if status != PipelineStatus.AWAITING_FEEDBACK.value:
            # Send completion event
            await event_queue_manager.push_event(
                stream_id,
                SSEEvent(
                    type=SSEEventType.COMPLETE,
                    data={
                        "status": status,
                        "report_available": bool(final_state.get("report")),
                        "total_claims": len(final_state.get("claims", [])),
                        "citations_count": len(final_state.get("citations", [])),
                    },
                ),
            )
    
    return final_state


async def resume_workflow_with_streaming(
    stream_id: str,
    feedback: dict,
    enable_hitl: bool = True,
    enable_parallel: bool = True,
) -> AgentState:
    """Resume a paused workflow after HITL feedback.
    
    Args:
        stream_id: The stream to resume.
        feedback: The user's feedback.
        enable_hitl: Whether to enable HITL (should match original).
        enable_parallel: Whether to enable parallel execution.
        
    Returns:
        Updated workflow state.
    """
    from graph.state import HITLFeedback
    
    # Create workflow with same settings
    workflow = create_workflow(
        enable_hitl=enable_hitl,
        enable_parallel=enable_parallel,
        use_memory_checkpointer=True,
    )
    
    config = {
        "configurable": {
            "thread_id": stream_id,
        }
    }
    
    # Create feedback object
    hitl_feedback = HITLFeedback(
        action=feedback.get("action", "approve"),
        topic=feedback.get("topic"),
        correction=feedback.get("correction"),
    )
    
    logger.info(
        "workflow_resuming",
        stream_id=stream_id,
        action=hitl_feedback.action,
    )
    
    # Push resuming event
    await event_queue_manager.push_event(
        stream_id,
        SSEEvent(
            type=SSEEventType.AGENT_STARTED,
            data={
                "agent": "system",
                "action": "resume",
                "detail": f"Resuming with action: {hitl_feedback.action}",
            },
        ),
    )
    
    # Update state with feedback and resume
    await workflow.aupdate_state(
        config,
        {"hitl_feedback": hitl_feedback},
    )
    
    last_trace_count = 0
    current_state = await event_queue_manager.get_state(stream_id)
    if current_state:
        last_trace_count = len(current_state.get("agent_trace", []))
    
    final_state = None
    
    try:
        # Stream remaining events
        async for event in workflow.astream_events(
            None,  # Resume from checkpoint
            config,
            version="v2",
        ):
            event_kind = event.get("event", "")
            
            if event_kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict):
                    trace_events = output.get("agent_trace", [])
                    
                    for trace in trace_events[last_trace_count:]:
                        sse_event = trace_event_to_sse(trace, stream_id)
                        await event_queue_manager.push_event(stream_id, sse_event)
                    
                    last_trace_count = len(trace_events)
                    await event_queue_manager.update_state(stream_id, output)
                    final_state = output
    
    except Exception as e:
        logger.error("workflow_resume_error", stream_id=stream_id, error=str(e))
        await event_queue_manager.push_event(
            stream_id,
            SSEEvent(
                type=SSEEventType.ERROR,
                data={"error": "resume_error", "message": str(e)},
            ),
        )
        raise
    
    # Send completion event
    if final_state:
        status = final_state.get("status", PipelineStatus.COMPLETED.value)
        
        if status != PipelineStatus.AWAITING_FEEDBACK.value:
            await event_queue_manager.push_event(
                stream_id,
                SSEEvent(
                    type=SSEEventType.COMPLETE,
                    data={
                        "status": status,
                        "report_available": bool(final_state.get("report")),
                    },
                ),
            )
    
    return final_state
