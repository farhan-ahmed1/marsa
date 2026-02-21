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
                # Don't clear state if stream exists - preserve it for resumption
                # Only create new queue for new events
                queue: asyncio.Queue[SSEEvent] = asyncio.Queue()
                self._queues[stream_id] = queue
                # Keep existing state and workflow
                return queue
            
            # New stream - initialize everything
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
            # Build metrics
            metrics = {
                "claims_count": len(state.get("claims", [])),
                "verification_count": len(state.get("verification_results", [])),
                "citations_count": len(state.get("citations", [])),
                "trace_events_count": len(state.get("agent_trace", [])),
                "iteration_count": state.get("iteration_count", 0),
            }
            
            # Calculate fact-check pass rate
            verification_results = state.get("verification_results", [])
            if verification_results:
                supported = sum(
                    1 for v in verification_results
                    if getattr(v, "verdict", None) == "supported" or 
                    (isinstance(v, dict) and v.get("verdict") == "supported")
                )
                metrics["fact_check_pass_rate"] = supported / len(verification_results)
            
            # Get report and convert to dict if it's a Pydantic model
            report_structured = state.get("report_structured")
            report_dict = None
            if report_structured:
                if hasattr(report_structured, "model_dump"):
                    report_dict = report_structured.model_dump()
                else:
                    report_dict = report_structured
            
            # Send final state as complete event with full report data
            yield SSEEvent(
                type=SSEEventType.COMPLETE,
                data={
                    "status": state.get("status", "completed"),
                    "report": report_dict,
                    "raw_report": state.get("report"),
                    "metrics": metrics,
                    "report_available": bool(report_dict),
                },
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
    # Note: Agent "complete" actions become AGENT_COMPLETED to distinguish
    # from workflow-level COMPLETE events (which include the report)
    action_mapping = {
        "start": SSEEventType.AGENT_STARTED,
        "tool_call": SSEEventType.TOOL_CALLED,
        "tool_result": SSEEventType.TOOL_RESULT,
        "claim_extracted": SSEEventType.CLAIM_EXTRACTED,
        "claim_verified": SSEEventType.CLAIM_VERIFIED,
        "report_generating": SSEEventType.REPORT_GENERATING,
        "complete": SSEEventType.AGENT_COMPLETED,  # Agent-level complete, not workflow complete
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
    
    # Config for checkpointing + LangSmith metadata
    config = {
        "configurable": {
            "thread_id": stream_id,
        },
        # LangSmith tracing metadata — automatically captured when tracing is enabled
        "run_name": f"MARSA: {query[:60]}",
        "tags": ["marsa", "research"],
        "metadata": {
            "query": query[:200],
            "stream_id": stream_id,
            "enable_hitl": enable_hitl,
            "enable_parallel": enable_parallel,
        },
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
        # Ensure final state is saved to event_queue_manager before sending complete event
        await event_queue_manager.update_state(stream_id, final_state)
        
        # Give a small delay to ensure any async operations complete
        await asyncio.sleep(0.1)
        
        status = final_state.get("status", PipelineStatus.COMPLETED.value)
        
        # Check if we hit HITL checkpoint (workflow interrupted before hitl_checkpoint node)
        # Status will be "fact_checking" because hitl_checkpoint node hasn't run yet
        if enable_hitl and status == PipelineStatus.FACT_CHECKING.value:
            # Workflow paused for HITL checkpoint
            logger.info("workflow_paused_for_hitl", stream_id=stream_id, status=status)
            
            # Send HITL checkpoint event to frontend
            await event_queue_manager.push_event(
                stream_id,
                SSEEvent(
                    type=SSEEventType.HITL_CHECKPOINT,
                    data={
                        "status": "awaiting_feedback",
                        "message": "Review required before proceeding",
                        "claims_count": len(final_state.get("claims", [])),
                        "verified_count": len(final_state.get("verification_results", [])),
                    },
                ),
            )
            
            # Update state to awaiting_feedback so /feedback endpoint accepts submission
            final_state = {**final_state, "status": PipelineStatus.AWAITING_FEEDBACK.value}
            await event_queue_manager.update_state(stream_id, final_state)
            logger.info("state_updated_to_awaiting_feedback", stream_id=stream_id)
        
        # Only send complete event if workflow actually completed or hit HITL checkpoint
        # Do not send complete if still in intermediate states (planning, researching, fact_checking)
        elif status == PipelineStatus.AWAITING_FEEDBACK.value:
            # HITL checkpoint was already sent in the astream_events loop above
            pass
        elif status in (
            PipelineStatus.COMPLETED.value,
            PipelineStatus.FAILED.value,
            PipelineStatus.SYNTHESIZING.value,  # Synthesizer in progress
            "completed",
            "failed",
            "synthesizing",
        ):
            # Build metrics for the complete event
            agent_trace_items = final_state.get("agent_trace", [])
            total_tokens = sum(
                getattr(e, "tokens_used", 0) or 0 for e in agent_trace_items
            )
            
            # Compute total latency from trace timestamps
            total_latency_ms: float = 0.0
            if agent_trace_items:
                try:
                    from datetime import datetime as _dt
                    _ts0 = str(
                        agent_trace_items[0].timestamp
                        if hasattr(agent_trace_items[0], "timestamp")
                        else agent_trace_items[0].get("timestamp", "")
                    )
                    _last_ev = agent_trace_items[-1]
                    _ts1 = str(
                        _last_ev.timestamp if hasattr(_last_ev, "timestamp")
                        else _last_ev.get("timestamp", "")
                    )
                    _last_lat = (
                        _last_ev.latency_ms if hasattr(_last_ev, "latency_ms")
                        else _last_ev.get("latency_ms", 0)
                    ) or 0
                    total_latency_ms = (
                        (_dt.fromisoformat(_ts1) - _dt.fromisoformat(_ts0)).total_seconds() * 1000
                        + _last_lat
                    )
                except Exception:
                    pass
            
            plan = final_state.get("plan")
            metrics = {
                "claims_count": len(final_state.get("claims", [])),
                "verification_count": len(final_state.get("verification_results", [])),
                "citations_count": len(final_state.get("citations", [])),
                "trace_events_count": len(agent_trace_items),
                "iteration_count": final_state.get("iteration_count", 0),
                "total_tokens": total_tokens,
                "total_latency_ms": round(total_latency_ms),
                "query_type": plan.query_type.value if plan else "unknown",
                "estimated_complexity": plan.estimated_complexity.value if plan else "unknown",
                "sub_query_count": len(plan.sub_queries) if plan else 0,
            }
            
            # Calculate fact-check pass rate
            verification_results = final_state.get("verification_results", [])
            if verification_results:
                supported = sum(
                    1 for v in verification_results
                    if getattr(v, "verdict", None) == "supported" or 
                    (isinstance(v, dict) and v.get("verdict") == "supported")
                )
                metrics["fact_check_pass_rate"] = supported / len(verification_results)
            
            # Get report and convert to dict if it's a Pydantic model
            report_structured = final_state.get("report_structured")
            report_dict = None
            if report_structured:
                if hasattr(report_structured, "model_dump"):
                    report_dict = report_structured.model_dump()
                else:
                    report_dict = report_structured
            
            # Send completion event with full report data
            await event_queue_manager.push_event(
                stream_id,
                SSEEvent(
                    type=SSEEventType.COMPLETE,
                    data={
                        "status": status,
                        "report_available": bool(report_dict),
                        "report": report_dict,
                        "raw_report": final_state.get("report"),
                        "metrics": metrics,
                        "total_claims": len(final_state.get("claims", [])),
                        "citations_count": len(final_state.get("citations", [])),
                    },
                ),
            )
        else:
            # Workflow is in intermediate state (planning, researching, fact_checking)
            # This means it was interrupted - do not send complete event
            logger.info(
                "workflow_interrupted",
                stream_id=stream_id,
                status=status,
                message="Workflow interrupted in intermediate state, not sending complete event"
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
        },
        "run_name": f"MARSA resume: {stream_id[:12]}",
        "tags": ["marsa", "research", "resume"],
        "metadata": {
            "stream_id": stream_id,
            "feedback_action": feedback.get("action", "approve"),
        },
    }
    hitl_feedback = HITLFeedback(
        action=feedback.get("action", "approve"),
        topic=feedback.get("topic"),
        correction=feedback.get("correction"),
    )
    
    # Get current state for logging
    current_state = await event_queue_manager.get_state(stream_id)
    
    logger.info(
        "workflow_resuming",
        stream_id=stream_id,
        action=hitl_feedback.action,
        has_existing_state=bool(current_state),
        current_query=current_state.get("query", "")[:50] if current_state else "none",
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
    
    # Verify state after update
    updated_state = await event_queue_manager.get_state(stream_id)
    last_trace_count = 0
    if updated_state:
        last_trace_count = len(updated_state.get("agent_trace", []))
        logger.info(
            "state_after_feedback_update",
            query_preserved=bool(updated_state.get("query")),
            trace_count=last_trace_count,
        )
    
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
                    
                    # Only update stored state when the output looks like a
                    # full AgentState (has the query field).  Intermediate
                    # node outputs (e.g. hitl_checkpoint returning just
                    # {"status": "awaiting_feedback"}) are partial and would
                    # overwrite the complete state we need for the report.
                    if output.get("query"):
                        await event_queue_manager.update_state(stream_id, output)
                        final_state = output
                    elif final_state is None:
                        # First output - keep as fallback
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
        # Ensure final state is saved to event_queue_manager before sending complete event
        await event_queue_manager.update_state(stream_id, final_state)
        
        # Give a small delay to ensure any async operations complete
        await asyncio.sleep(0.1)
        
        status = final_state.get("status", PipelineStatus.COMPLETED.value)
        
        if status != PipelineStatus.AWAITING_FEEDBACK.value:
            # Build metrics for the complete event
            metrics = {
                "claims_count": len(final_state.get("claims", [])),
                "verification_count": len(final_state.get("verification_results", [])),
                "citations_count": len(final_state.get("citations", [])),
                "trace_events_count": len(final_state.get("agent_trace", [])),
                "iteration_count": final_state.get("iteration_count", 0),
            }
            
            # Calculate fact-check pass rate
            verification_results = final_state.get("verification_results", [])
            if verification_results:
                supported = sum(
                    1 for v in verification_results
                    if getattr(v, "verdict", None) == "supported" or 
                    (isinstance(v, dict) and v.get("verdict") == "supported")
                )
                metrics["fact_check_pass_rate"] = supported / len(verification_results)
            
            # Get report and convert to dict if it's a Pydantic model
            report_structured = final_state.get("report_structured")
            report_dict = None
            if report_structured:
                if hasattr(report_structured, "model_dump"):
                    report_dict = report_structured.model_dump()
                else:
                    report_dict = report_structured
            
            await event_queue_manager.push_event(
                stream_id,
                SSEEvent(
                    type=SSEEventType.COMPLETE,
                    data={
                        "status": status,
                        "report_available": bool(report_dict),
                        "report": report_dict,
                        "raw_report": final_state.get("report"),
                        "metrics": metrics,
                    },
                ),
            )
    
    return final_state
