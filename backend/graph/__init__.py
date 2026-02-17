"""Graph modules for MARSA LangGraph workflow.

This package contains the LangGraph state definitions, workflow assembly,
and checkpointing configuration.

Modules:
- state: AgentState TypedDict and nested Pydantic models
- workflow: LangGraph StateGraph assembly and compilation
- checkpointer: SQLite persistence configuration

Note: workflow imports are deferred to avoid circular imports with agents.
Use explicit imports: `from graph.workflow import create_workflow, get_workflow, run_research`
"""

from graph.checkpointer import (
    create_checkpointer,
    get_latest_state,
    get_thread_history,
)
from graph.state import (
    AgentName,
    AgentState,
    Citation,
    Claim,
    ComplexityLevel,
    ConfidenceLevel,
    HITLFeedback,
    PipelineStatus,
    QueryPlan,
    QueryType,
    Report,
    ReportMetadata,
    ReportSection,
    ResearchResult,
    SearchStrategy,
    TraceEvent,
    VerificationResult,
    VerificationVerdict,
    add_error,
    add_trace_event,
    create_initial_state,
)

# Note: workflow imports are deferred to avoid circular imports.
# Import directly from graph.workflow when needed.

__all__ = [
    # State
    "AgentState",
    "AgentName",
    "PipelineStatus",
    "QueryType",
    "SearchStrategy",
    "ComplexityLevel",
    "ConfidenceLevel",
    "VerificationVerdict",
    "QueryPlan",
    "ResearchResult",
    "Claim",
    "VerificationResult",
    "Citation",
    "TraceEvent",
    "ReportSection",
    "ReportMetadata",
    "Report",
    "HITLFeedback",
    "create_initial_state",
    "add_trace_event",
    "add_error",
    # Checkpointer
    "create_checkpointer",
    "get_latest_state",
    "get_thread_history",
    # Note: workflow exports (create_workflow, get_workflow, run_research) must be
    # imported directly from graph.workflow to avoid circular imports
]
