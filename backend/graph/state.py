"""LangGraph state schema for the MARSA multi-agent research system.

This module defines the core state structure that flows through the LangGraph
workflow, including all nested Pydantic models for type safety and validation.

The AgentState TypedDict is the main state object that gets passed between
agents (Planner, Researcher, Fact-Checker, Synthesizer) in the research pipeline.
"""

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional, TypedDict

from pydantic import BaseModel, Field


def _utcnow_iso() -> str:
    """Get current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _today_str() -> str:
    """Get today's date as YYYY-MM-DD string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Enums for type-safe string values
# ---------------------------------------------------------------------------


class QueryType(str, Enum):
    """Types of research queries the system can handle."""
    
    FACTUAL = "factual"           # Simple factual questions with clear answers
    COMPARISON = "comparison"      # Comparing multiple items/concepts
    EXPLORATORY = "exploratory"    # Open-ended research questions
    OPINION = "opinion"           # Questions seeking analysis or opinions
    HOWTO = "howto"               # Step-by-step guides or tutorials
    DEFINITION = "definition"      # Definitions and explanations


class SearchStrategy(str, Enum):
    """Search strategies for research."""
    
    WEB_ONLY = "web_only"         # Only use web search (Tavily)
    DOCS_ONLY = "docs_only"       # Only use document store (ChromaDB)
    HYBRID = "hybrid"             # Use both sources


class ComplexityLevel(str, Enum):
    """Estimated complexity of a query."""
    
    LOW = "low"                   # Simple, single-source answer
    MEDIUM = "medium"             # Requires multiple sources
    HIGH = "high"                 # Complex, multi-faceted research


class VerificationVerdict(str, Enum):
    """Verdict from fact-checking a claim."""
    
    SUPPORTED = "supported"        # Claim is verified as accurate
    CONTRADICTED = "contradicted"  # Claim is contradicted by sources
    UNVERIFIABLE = "unverifiable"  # Cannot be verified with available sources


class ConfidenceLevel(str, Enum):
    """Confidence level for claims and results."""
    
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentName(str, Enum):
    """Names of agents in the pipeline."""
    
    PLANNER = "planner"
    RESEARCHER = "researcher"
    FACT_CHECKER = "fact_checker"
    SYNTHESIZER = "synthesizer"


class PipelineStatus(str, Enum):
    """Status of the research pipeline."""
    
    PLANNING = "planning"
    RESEARCHING = "researching"
    FACT_CHECKING = "fact_checking"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_FEEDBACK = "awaiting_feedback"  # HITL checkpoint


# ---------------------------------------------------------------------------
# Nested Pydantic Models
# ---------------------------------------------------------------------------


class QueryPlan(BaseModel):
    """Planner agent output defining the research strategy.
    
    The Planner analyzes the incoming query and produces this plan
    to guide the Researcher agent.
    """
    
    query_type: QueryType = Field(
        description="Classification of the query type"
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Decomposed research questions to investigate"
    )
    parallel: bool = Field(
        default=True,
        description="Whether sub-queries can be executed in parallel"
    )
    needs_fact_check: bool = Field(
        default=True,
        description="Whether results need fact-checking (skip for simple factual queries)"
    )
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.HYBRID,
        description="Which data sources to use"
    )
    estimated_complexity: ComplexityLevel = Field(
        default=ComplexityLevel.MEDIUM,
        description="Estimated complexity of the query"
    )
    reasoning: str = Field(
        default="",
        description="Planner's reasoning for this plan"
    )


class ResearchResult(BaseModel):
    """A single research finding from the Researcher agent.
    
    Contains the raw information retrieved from a data source
    along with metadata for source tracking.
    """
    
    content: str = Field(
        description="The retrieved content or finding"
    )
    source_url: str = Field(
        description="URL or identifier of the source"
    )
    source_title: str = Field(
        default="",
        description="Title of the source document/page"
    )
    source_type: str = Field(
        default="web",
        description="Type of source: 'web', 'document', 'github'"
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance score from search (0.0 to 1.0)"
    )
    sub_query: str = Field(
        default="",
        description="The sub-query that produced this result"
    )
    published_date: Optional[str] = Field(
        default=None,
        description="Publication date if available (ISO format)"
    )
    retrieved_at: str = Field(
        default_factory=_utcnow_iso,
        description="Timestamp when this result was retrieved"
    )


class Claim(BaseModel):
    """An extracted claim from research results.
    
    Claims are factual statements extracted from research results
    that need to be verified by the Fact-Checker agent.
    """
    
    statement: str = Field(
        description="The factual claim statement"
    )
    source_url: str = Field(
        description="URL of the source where this claim was found"
    )
    source_title: str = Field(
        default="",
        description="Title of the source"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Initial confidence level based on source quality"
    )
    category: str = Field(
        default="fact",
        description="Category: 'fact', 'opinion', 'statistic', 'quote'"
    )
    context: str = Field(
        default="",
        description="Surrounding context for the claim"
    )


class VerificationResult(BaseModel):
    """Result from fact-checking a claim.
    
    Contains the verification verdict along with supporting evidence
    and reasoning from independent verification searches.
    """
    
    claim: Claim = Field(
        description="The original claim that was verified"
    )
    verdict: VerificationVerdict = Field(
        description="Verification verdict: supported, contradicted, or unverifiable"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the verdict (0.0 to 1.0)"
    )
    supporting_sources: list[str] = Field(
        default_factory=list,
        description="URLs of sources that support the claim"
    )
    contradicting_sources: list[str] = Field(
        default_factory=list,
        description="URLs of sources that contradict the claim"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of the verification process and conclusion"
    )
    verification_query: str = Field(
        default="",
        description="The independent query used for verification"
    )


class Citation(BaseModel):
    """A citation for the final report.
    
    Represents a numbered reference with quality scoring
    for inclusion in the synthesized report.
    """
    
    number: int = Field(
        ge=1,
        description="Citation number (1-indexed)"
    )
    title: str = Field(
        description="Title of the cited source"
    )
    url: str = Field(
        description="URL of the cited source"
    )
    source_quality_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Quality score of the source (0.0 to 1.0)"
    )
    accessed_date: str = Field(
        default_factory=_today_str,
        description="Date when source was accessed (YYYY-MM-DD)"
    )
    source_type: str = Field(
        default="web",
        description="Type of source: 'web', 'document', 'github'"
    )
    snippet: str = Field(
        default="",
        description="Brief excerpt from the source used in the report"
    )


class TraceEvent(BaseModel):
    """An event in the agent trace for observability.
    
    Records agent activity including tool calls, LLM invocations,
    and state transitions for the observability dashboard.
    """
    
    agent: AgentName = Field(
        description="Which agent generated this event"
    )
    action: str = Field(
        description="Type of action: 'tool_call', 'llm_call', 'state_update', etc."
    )
    detail: str = Field(
        description="Human-readable description of the action"
    )
    timestamp: str = Field(
        default_factory=_utcnow_iso,
        description="ISO timestamp of the event"
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Latency in milliseconds (if applicable)"
    )
    tokens_used: Optional[int] = Field(
        default=None,
        description="Tokens consumed (for LLM calls)"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata for the event"
    )


class ReportSection(BaseModel):
    """A section of the final report.
    
    Represents a logical section with heading and content
    for structured report organization.
    """
    
    heading: str = Field(
        description="Section heading"
    )
    content: str = Field(
        description="Section content (may include inline citations like [1], [2])"
    )
    order: int = Field(
        default=0,
        description="Order of this section in the report"
    )


class ReportMetadata(BaseModel):
    """Metadata about the generated report.
    
    Contains information about the research process
    and statistics for transparency.
    """
    
    query: str = Field(
        description="Original user query"
    )
    generated_at: str = Field(
        default_factory=_utcnow_iso,
        description="ISO timestamp when report was generated"
    )
    total_latency_ms: float = Field(
        default=0.0,
        description="Total pipeline latency in milliseconds"
    )
    llm_calls: int = Field(
        default=0,
        description="Number of LLM calls made"
    )
    total_tokens: int = Field(
        default=0,
        description="Total tokens consumed"
    )
    sources_searched: int = Field(
        default=0,
        description="Number of sources searched"
    )
    claims_verified: int = Field(
        default=0,
        description="Number of claims fact-checked"
    )
    fact_check_pass_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of claims that passed fact-checking"
    )


class Report(BaseModel):
    """The final synthesized research report.
    
    Contains the complete report with sections, citations,
    and confidence assessment.
    """
    
    title: str = Field(
        description="Report title"
    )
    summary: str = Field(
        description="2-3 sentence executive summary"
    )
    sections: list[ReportSection] = Field(
        default_factory=list,
        description="Main content sections"
    )
    confidence_summary: str = Field(
        default="",
        description="Overall confidence assessment and caveats"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Numbered reference list"
    )
    metadata: ReportMetadata = Field(
        default_factory=lambda: ReportMetadata(query=""),
        description="Report metadata and statistics"
    )


class HITLFeedback(BaseModel):
    """Human-in-the-loop feedback from the user.
    
    Captures user feedback at checkpoints for
    guiding the research process.
    """
    
    action: str = Field(
        description="User action: 'approve', 'dig_deeper', 'correct', 'abort'"
    )
    topic: Optional[str] = Field(
        default=None,
        description="Topic to dig deeper into (if action is 'dig_deeper')"
    )
    correction: Optional[str] = Field(
        default=None,
        description="User correction (if action is 'correct')"
    )
    timestamp: str = Field(
        default_factory=_utcnow_iso,
        description="When feedback was provided"
    )


class SubQueryResult(BaseModel):
    """Result from researching a single sub-query (parallel execution).
    
    Used to aggregate results from parallel sub-query workers.
    """
    
    sub_query: str = Field(
        description="The sub-query that was researched"
    )
    results: list[ResearchResult] = Field(
        default_factory=list,
        description="Research results for this sub-query"
    )
    trace_events: list[TraceEvent] = Field(
        default_factory=list,
        description="Trace events from this sub-query research"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Errors encountered during this sub-query"
    )


# ---------------------------------------------------------------------------
# Main Agent State (TypedDict for LangGraph)
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Main state object that flows through the LangGraph workflow.
    
    This TypedDict defines all fields that are passed between agents
    in the research pipeline. Each agent reads relevant fields and
    updates others based on its processing.
    
    Fields:
        query: Original user query
        plan: Planner's output defining research strategy
        sub_queries: Decomposed queries from the plan
        research_results: Raw findings from the Researcher
        claims: Extracted claims for fact-checking
        verification_results: Results from the Fact-Checker
        source_scores: Quality scores for sources (URL -> score)
        report: Final synthesized report (raw text)
        report_structured: Structured report object
        citations: List of citations for the report
        agent_trace: Observability events from all agents
        iteration_count: Number of research loops (for loop guard)
        status: Current pipeline status
        errors: Accumulated error messages
        hitl_feedback: Human feedback (if checkpoint was triggered)
        started_at: Pipeline start timestamp
    
    Example:
        state: AgentState = {
            "query": "Compare Rust vs Go for distributed systems",
            "plan": QueryPlan(...),
            "status": PipelineStatus.RESEARCHING,
            "iteration_count": 0,
            "errors": [],
            "agent_trace": [],
        }
    """
    
    # Input
    query: str
    
    # Planner output
    plan: QueryPlan
    sub_queries: list[str]
    
    # Researcher output
    research_results: list[ResearchResult]
    claims: list[Claim]
    
    # Fact-Checker output
    verification_results: list[VerificationResult]
    source_scores: dict[str, float]
    
    # Synthesizer output
    report: str
    report_structured: Report
    citations: list[Citation]
    
    # Observability
    agent_trace: list[TraceEvent]
    
    # Control flow
    iteration_count: int
    status: PipelineStatus
    errors: list[str]
    
    # Human-in-the-loop
    hitl_feedback: Optional[HITLFeedback]
    
    # Timing
    started_at: str
    
    # Parallel execution support - uses Annotated with operator.add
    # to aggregate results from parallel Send branches
    parallel_results: Annotated[list[dict], operator.add]


# ---------------------------------------------------------------------------
# State Factory Functions
# ---------------------------------------------------------------------------


def create_initial_state(query: str) -> AgentState:
    """Create an initial AgentState for a new query.
    
    Args:
        query: The user's research query
        
    Returns:
        An initialized AgentState ready for the pipeline
    """
    return AgentState(
        query=query,
        plan=QueryPlan(
            query_type=QueryType.FACTUAL,
            sub_queries=[],
            parallel=True,
            needs_fact_check=True,
            search_strategy=SearchStrategy.HYBRID,
            estimated_complexity=ComplexityLevel.MEDIUM,
            reasoning="",
        ),
        sub_queries=[],
        research_results=[],
        claims=[],
        verification_results=[],
        source_scores={},
        report="",
        report_structured=Report(
            title="",
            summary="",
            sections=[],
            confidence_summary="",
            citations=[],
            metadata=ReportMetadata(query=query),
        ),
        citations=[],
        agent_trace=[],
        iteration_count=0,
        status=PipelineStatus.PLANNING,
        errors=[],
        hitl_feedback=None,
        started_at=datetime.now(timezone.utc).isoformat(),
        parallel_results=[],
    )


def add_trace_event(
    state: AgentState,
    agent: AgentName,
    action: str,
    detail: str,
    latency_ms: Optional[float] = None,
    tokens_used: Optional[int] = None,
    **metadata: dict
) -> AgentState:
    """Add a trace event to the state.
    
    Args:
        state: Current agent state
        agent: Which agent is generating the event
        action: Type of action performed
        detail: Human-readable description
        latency_ms: Optional latency measurement
        tokens_used: Optional token count
        **metadata: Additional metadata to include
        
    Returns:
        Updated state with new trace event
    """
    trace = state.get("agent_trace", []).copy()
    trace.append(TraceEvent(
        agent=agent,
        action=action,
        detail=detail,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        metadata=metadata,
    ))
    return {**state, "agent_trace": trace}


def add_error(state: AgentState, error: str) -> AgentState:
    """Add an error message to the state.
    
    Args:
        state: Current agent state
        error: Error message to add
        
    Returns:
        Updated state with new error
    """
    errors = state.get("errors", []).copy()
    errors.append(error)
    return {**state, "errors": errors}
