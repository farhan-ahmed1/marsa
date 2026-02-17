"""API routes for the MARSA research assistant.

This module defines all HTTP endpoints for query submission, SSE streaming,
report retrieval, HITL feedback, and health checks.
"""

import asyncio
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse

from api.models import (
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    HITLCheckpointResponse,
    QueryAcceptedResponse,
    QueryRequest,
    ReportResponse,
)
from api.streaming import (
    create_sse_response,
    event_queue_manager,
    resume_workflow_with_streaming,
    run_workflow_with_streaming,
)
from graph.state import PipelineStatus, Report

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Store for tracking query settings (for resume)
# ---------------------------------------------------------------------------

_query_settings: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Query Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=QueryAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Query accepted for processing"},
        400: {"model": ErrorResponse, "description": "Invalid query"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    summary="Submit a research query",
    description="Submit a research query for processing by the agent pipeline.",
)
async def submit_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
) -> QueryAcceptedResponse:
    """Submit a research query and receive a stream ID for tracking.
    
    The query will be processed asynchronously by the agent pipeline.
    Use the stream_id to connect to the SSE stream or retrieve the report.
    
    Args:
        request: Query submission request with the research question.
        background_tasks: FastAPI background task manager.
        
    Returns:
        Response with stream_id for tracking this query.
    """
    # Generate unique stream ID
    stream_id = str(uuid.uuid4())
    
    logger.info(
        "query_submitted",
        stream_id=stream_id,
        query_length=len(request.query),
        enable_hitl=request.enable_hitl,
        enable_parallel=request.enable_parallel,
    )
    
    # Create event queue for this stream
    await event_queue_manager.create_stream(stream_id)
    
    # Store settings for potential resume
    _query_settings[stream_id] = {
        "enable_hitl": request.enable_hitl,
        "enable_parallel": request.enable_parallel,
        "query": request.query,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Run workflow in background
    async def run_workflow():
        try:
            await run_workflow_with_streaming(
                stream_id=stream_id,
                query=request.query,
                enable_hitl=request.enable_hitl,
                enable_parallel=request.enable_parallel,
            )
        except Exception as e:
            logger.error(
                "workflow_background_error",
                stream_id=stream_id,
                error=str(e),
            )
    
    # Start workflow task
    task = asyncio.create_task(run_workflow())
    event_queue_manager.set_workflow_task(stream_id, task)
    
    return QueryAcceptedResponse(
        stream_id=stream_id,
        status="accepted",
        message=f"Query accepted. Connect to /api/query/{stream_id}/stream for updates.",
    )


@router.get(
    "/query/{stream_id}/stream",
    response_class=StreamingResponse,
    responses={
        200: {"description": "SSE event stream"},
        404: {"model": ErrorResponse, "description": "Stream not found"},
    },
    summary="Stream agent events",
    description="Connect to SSE stream for real-time agent trace events.",
)
async def stream_query_events(stream_id: str) -> StreamingResponse:
    """Stream agent events for a query via Server-Sent Events.
    
    Connect to this endpoint to receive real-time updates as agents
    process the query. Events include agent actions, tool calls,
    claim extractions, and completion notifications.
    
    Args:
        stream_id: The query session ID from submit_query.
        
    Returns:
        SSE stream of agent events.
        
    Raises:
        HTTPException: If stream_id is not found.
    """
    if not event_queue_manager.stream_exists(stream_id):
        logger.warning("stream_not_found", stream_id=stream_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream {stream_id} not found",
        )
    
    logger.info("stream_connected", stream_id=stream_id)
    return create_sse_response(stream_id)


@router.get(
    "/query/{stream_id}/report",
    response_model=ReportResponse,
    responses={
        200: {"description": "Research report"},
        404: {"model": ErrorResponse, "description": "Stream not found"},
        425: {"model": ErrorResponse, "description": "Report not ready yet"},
    },
    summary="Get research report",
    description="Retrieve the final research report once processing is complete.",
)
async def get_report(stream_id: str) -> ReportResponse:
    """Get the final research report for a query.
    
    This endpoint returns the synthesized report with citations
    once the pipeline has completed. Returns 425 if still processing.
    
    Args:
        stream_id: The query session ID.
        
    Returns:
        The research report with metadata.
        
    Raises:
        HTTPException: If stream not found or report not ready.
    """
    state = await event_queue_manager.get_state(stream_id)
    
    if state is None:
        logger.warning("report_stream_not_found", stream_id=stream_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream {stream_id} not found",
        )
    
    pipeline_status = state.get("status", "")
    
    # Handle different statuses
    if pipeline_status == PipelineStatus.AWAITING_FEEDBACK.value:
        raise HTTPException(
            status_code=425,  # Too Early
            detail="Awaiting HITL feedback before completion",
        )
    
    if pipeline_status not in (
        PipelineStatus.COMPLETED.value,
        PipelineStatus.FAILED.value,
        "completed",
        "failed",
    ):
        raise HTTPException(
            status_code=425,  # Too Early
            detail=f"Report not ready. Current status: {pipeline_status}",
        )
    
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
    
    logger.info(
        "report_retrieved",
        stream_id=stream_id,
        status=pipeline_status,
    )
    
    return ReportResponse(
        stream_id=stream_id,
        status=pipeline_status,
        report=state.get("report_structured"),
        raw_report=state.get("report"),
        metrics=metrics,
        error=state.get("errors", [None])[0] if state.get("errors") else None,
    )


@router.post(
    "/query/{stream_id}/feedback",
    response_model=FeedbackResponse,
    responses={
        200: {"description": "Feedback accepted"},
        404: {"model": ErrorResponse, "description": "Stream not found"},
        400: {"model": ErrorResponse, "description": "Invalid feedback or not at checkpoint"},
    },
    summary="Submit HITL feedback",
    description="Submit human-in-the-loop feedback at a checkpoint.",
)
async def submit_feedback(
    stream_id: str,
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
) -> FeedbackResponse:
    """Submit feedback at a human-in-the-loop checkpoint.
    
    When the pipeline pauses for human review, use this endpoint to
    provide feedback and control the next steps.
    
    Actions:
    - approve: Continue to synthesis
    - dig_deeper: Re-research with focus on specified topic
    - correct: Provide corrections and re-research
    - abort: Cancel the research
    
    Args:
        stream_id: The query session ID.
        request: Feedback with action and optional details.
        
    Returns:
        Confirmation of feedback processing.
    """
    state = await event_queue_manager.get_state(stream_id)
    
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream {stream_id} not found",
        )
    
    pipeline_status = state.get("status", "")
    
    # Only accept feedback when at HITL checkpoint
    if pipeline_status != PipelineStatus.AWAITING_FEEDBACK.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not at HITL checkpoint. Current status: {pipeline_status}",
        )
    
    logger.info(
        "feedback_received",
        stream_id=stream_id,
        action=request.action,
    )
    
    # Validate action-specific requirements
    if request.action == "dig_deeper" and not request.topic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Topic required for 'dig_deeper' action",
        )
    
    if request.action == "correct" and not request.correction:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Correction text required for 'correct' action",
        )
    
    # Get original settings
    settings = _query_settings.get(stream_id, {})
    
    # Recreate event queue for resumption
    await event_queue_manager.create_stream(stream_id)
    
    # Define next action descriptions
    next_actions = {
        "approve": "Proceeding to report synthesis",
        "dig_deeper": f"Re-researching with focus on: {request.topic}",
        "correct": "Re-researching with corrections applied",
        "abort": "Workflow aborted",
    }
    
    if request.action != "abort":
        # Resume workflow in background
        async def resume_workflow():
            try:
                await resume_workflow_with_streaming(
                    stream_id=stream_id,
                    feedback={
                        "action": request.action,
                        "topic": request.topic,
                        "correction": request.correction,
                    },
                    enable_hitl=settings.get("enable_hitl", True),
                    enable_parallel=settings.get("enable_parallel", True),
                )
            except Exception as e:
                logger.error(
                    "resume_background_error",
                    stream_id=stream_id,
                    error=str(e),
                )
        
        task = asyncio.create_task(resume_workflow())
        event_queue_manager.set_workflow_task(stream_id, task)
    
    return FeedbackResponse(
        stream_id=stream_id,
        status="processing" if request.action != "abort" else "aborted",
        message=f"Feedback '{request.action}' accepted",
        next_action=next_actions.get(request.action, "Unknown"),
    )


@router.get(
    "/query/{stream_id}/checkpoint",
    response_model=HITLCheckpointResponse,
    responses={
        200: {"description": "Checkpoint data"},
        404: {"model": ErrorResponse, "description": "Stream not found"},
        400: {"model": ErrorResponse, "description": "Not at checkpoint"},
    },
    summary="Get HITL checkpoint data",
    description="Get current checkpoint data for human review.",
)
async def get_checkpoint(stream_id: str) -> HITLCheckpointResponse:
    """Get data for HITL review at a checkpoint.
    
    Returns summary of findings, claims, and verification results
    for human review before proceeding.
    
    Args:
        stream_id: The query session ID.
        
    Returns:
        Checkpoint data for review.
    """
    state = await event_queue_manager.get_state(stream_id)
    
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream {stream_id} not found",
        )
    
    pipeline_status = state.get("status", "")
    
    if pipeline_status != PipelineStatus.AWAITING_FEEDBACK.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not at HITL checkpoint. Current status: {pipeline_status}",
        )
    
    # Build claims summary
    claims_summary = []
    verification_results = state.get("verification_results", [])
    
    for vr in verification_results:
        if hasattr(vr, "claim"):
            claims_summary.append({
                "statement": vr.claim.statement[:200],
                "verdict": vr.verdict.value if hasattr(vr.verdict, "value") else str(vr.verdict),
                "confidence": vr.confidence,
            })
        elif isinstance(vr, dict):
            claim = vr.get("claim", {})
            claims_summary.append({
                "statement": claim.get("statement", "")[:200],
                "verdict": vr.get("verdict", "unknown"),
                "confidence": vr.get("confidence", 0.5),
            })
    
    # Calculate average source quality
    source_scores = state.get("source_scores", {})
    avg_quality = sum(source_scores.values()) / len(source_scores) if source_scores else 0.5
    
    # Build summary
    claims_count = len(state.get("claims", []))
    verified_count = len(verification_results)
    supported_count = sum(
        1 for cs in claims_summary if cs.get("verdict") == "supported"
    )
    
    summary = (
        f"Researched query with {claims_count} claims extracted. "
        f"{verified_count} claims verified: {supported_count} supported, "
        f"{verified_count - supported_count} need review. "
        f"Average source quality: {avg_quality:.1%}."
    )
    
    return HITLCheckpointResponse(
        stream_id=stream_id,
        status=pipeline_status,
        summary=summary,
        claims_summary=claims_summary,
        source_quality=avg_quality,
    )


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and dependencies status.",
)
async def health_check() -> HealthResponse:
    """Check API health and status of dependencies.
    
    Returns:
        Health status with dependency information.
    """
    # Check dependencies
    dependencies = {}
    
    # Check if we can import required modules
    try:
        from graph.workflow import create_workflow
        dependencies["langgraph"] = "ok"
    except Exception as e:
        dependencies["langgraph"] = f"error: {str(e)}"
    
    try:
        import chromadb
        dependencies["chromadb"] = "ok"
    except Exception as e:
        dependencies["chromadb"] = f"error: {str(e)}"
    
    # Active streams count
    active_streams = event_queue_manager.get_active_streams()
    dependencies["active_streams"] = len(active_streams)
    
    overall_status = "ok" if all(
        v == "ok" or isinstance(v, int)
        for v in dependencies.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        dependencies=dependencies,
    )
