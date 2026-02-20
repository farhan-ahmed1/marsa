"""Pydantic models for the MARSA API request/response bodies.

This module defines all the data validation models used by the FastAPI
endpoints for clean, type-safe API contracts.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from graph.state import Report


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for submitting a research query.
    
    Attributes:
        query: The research question to investigate.
        enable_hitl: Whether to pause for human review after fact-checking.
        enable_parallel: Whether to execute sub-queries in parallel.
    """
    
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The research question to investigate"
    )
    enable_hitl: bool = Field(
        default=False,
        description="Enable human-in-the-loop checkpoint after fact-checking"
    )
    enable_parallel: bool = Field(
        default=True,
        description="Enable parallel sub-query execution"
    )
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Strip whitespace and validate query content."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v


class FeedbackRequest(BaseModel):
    """Request body for HITL feedback.
    
    Attributes:
        action: What action to take - approve, dig_deeper, correct, or abort.
        topic: Topic to dig deeper into (required if action is 'dig_deeper').
        correction: Correction text (required if action is 'correct').
    """
    
    action: str = Field(
        ...,
        description="Action: 'approve', 'dig_deeper', 'correct', or 'abort'"
    )
    topic: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Topic to explore (for 'dig_deeper' action)"
    )
    correction: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Correction text (for 'correct' action)"
    )
    
    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate action is one of the allowed values."""
        allowed = {"approve", "dig_deeper", "correct", "abort"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"Action must be one of: {', '.join(allowed)}")
        return v


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class QueryAcceptedResponse(BaseModel):
    """Response when a query is accepted for processing.
    
    Attributes:
        stream_id: Unique identifier for this query session.
        status: Current pipeline status.
        message: Confirmation message.
    """
    
    stream_id: str = Field(
        description="Unique identifier for tracking this query"
    )
    status: str = Field(
        default="accepted",
        description="Current status"
    )
    message: str = Field(
        default="Query accepted for processing",
        description="Human-readable message"
    )


class ReportResponse(BaseModel):
    """Response containing the final research report.
    
    Attributes:
        stream_id: The query session identifier.
        status: Pipeline status (should be 'completed').
        report: The structured research report.
        metrics: Summary metrics about the research process.
    """
    
    stream_id: str = Field(
        description="Query session identifier"
    )
    status: str = Field(
        description="Pipeline status"
    )
    report: Optional[Report] = Field(
        default=None,
        description="The generated research report"
    )
    raw_report: Optional[str] = Field(
        default=None,
        description="Raw report text if structured report not available"
    )
    metrics: dict = Field(
        default_factory=dict,
        description="Research process metrics"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if pipeline failed"
    )


class FeedbackResponse(BaseModel):
    """Response after processing HITL feedback.
    
    Attributes:
        stream_id: Query session identifier.
        status: Updated pipeline status.
        message: Confirmation message.
        next_action: What will happen next.
    """
    
    stream_id: str = Field(
        description="Query session identifier"
    )
    status: str = Field(
        description="Updated pipeline status"
    )
    message: str = Field(
        description="Confirmation message"
    )
    next_action: str = Field(
        description="What the system will do next"
    )


class HITLCheckpointResponse(BaseModel):
    """Response containing HITL checkpoint data for user review.
    
    Attributes:
        stream_id: Query session identifier.
        status: Should be 'awaiting_feedback'.
        summary: Summary of findings so far.
        claims_summary: Summary of claims and their verification status.
        source_quality: Average source quality score.
        available_actions: Actions the user can take.
    """
    
    stream_id: str
    status: str
    summary: str
    claims_summary: list[dict]
    source_quality: float
    available_actions: list[str] = Field(
        default=["approve", "dig_deeper", "correct", "abort"]
    )


class HealthResponse(BaseModel):
    """Response for health check endpoint.
    
    Attributes:
        status: Service status ('ok' or 'unhealthy').
        version: API version.
        timestamp: Current server time.
        dependencies: Status of dependent services.
    """
    
    status: str = Field(
        default="ok",
        description="Service status"
    )
    version: str = Field(
        default="1.0.0",
        description="API version"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Current server time"
    )
    dependencies: dict = Field(
        default_factory=dict,
        description="Health status of dependencies"
    )


class ErrorResponse(BaseModel):
    """Standard error response.
    
    Attributes:
        error: Error type/code.
        message: Human-readable error message.
        detail: Additional error details.
    """
    
    error: str = Field(
        description="Error type or code"
    )
    message: str = Field(
        description="Human-readable error message"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details"
    )


# ---------------------------------------------------------------------------
# SSE Event Models
# ---------------------------------------------------------------------------


class SSEEventType(str, Enum):
    """Types of SSE events streamed to the client."""
    
    CONNECTED = "connected"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"  # Individual agent finished its work
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    CLAIM_EXTRACTED = "claim_extracted"
    CLAIM_VERIFIED = "claim_verified"
    REPORT_GENERATING = "report_generating"
    HITL_CHECKPOINT = "hitl_checkpoint"
    COMPLETE = "complete"  # Workflow-level completion with report
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class SSEEvent(BaseModel):
    """Server-Sent Event payload.
    
    Attributes:
        type: The event type.
        data: Event-specific payload data.
        timestamp: When the event occurred.
        stream_id: Associated query session.
    """
    
    type: SSEEventType = Field(
        description="Event type"
    )
    data: dict = Field(
        default_factory=dict,
        description="Event payload"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Event timestamp"
    )
    stream_id: Optional[str] = Field(
        default=None,
        description="Associated stream ID"
    )
    
    def to_sse_format(self) -> str:
        """Format as SSE message string."""
        return f"data: {self.model_dump_json()}\n\n"
