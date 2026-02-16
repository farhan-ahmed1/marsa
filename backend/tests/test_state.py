"""Tests for the LangGraph state schema.

Tests serialization/deserialization, validation, edge cases,
and helper functions for all state models.
"""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from graph.state import ( # noqa: E402
    AgentName,
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


class TestQueryPlan:
    """Tests for the QueryPlan model."""
    
    def test_create_basic_query_plan(self):
        """Test creating a basic QueryPlan with required fields."""
        plan = QueryPlan(
            query_type=QueryType.FACTUAL,
            sub_queries=["What is Python?"],
            parallel=True,
            needs_fact_check=False,
            search_strategy=SearchStrategy.WEB_ONLY,
            estimated_complexity=ComplexityLevel.LOW,
        )
        
        assert plan.query_type == QueryType.FACTUAL
        assert plan.sub_queries == ["What is Python?"]
        assert plan.parallel is True
        assert plan.needs_fact_check is False
        assert plan.search_strategy == SearchStrategy.WEB_ONLY
        assert plan.estimated_complexity == ComplexityLevel.LOW
    
    def test_query_plan_defaults(self):
        """Test QueryPlan default values."""
        plan = QueryPlan(query_type=QueryType.COMPARISON)
        
        assert plan.sub_queries == []
        assert plan.parallel is True
        assert plan.needs_fact_check is True
        assert plan.search_strategy == SearchStrategy.HYBRID
        assert plan.estimated_complexity == ComplexityLevel.MEDIUM
        assert plan.reasoning == ""
    
    def test_query_plan_serialization(self):
        """Test QueryPlan JSON serialization and deserialization."""
        plan = QueryPlan(
            query_type=QueryType.EXPLORATORY,
            sub_queries=["trend 1", "trend 2", "trend 3"],
            parallel=True,
            needs_fact_check=True,
            search_strategy=SearchStrategy.HYBRID,
            estimated_complexity=ComplexityLevel.HIGH,
            reasoning="Complex multi-faceted research required"
        )
        
        # Serialize to JSON
        json_str = plan.model_dump_json()
        data = json.loads(json_str)
        
        assert data["query_type"] == "exploratory"
        assert len(data["sub_queries"]) == 3
        
        # Deserialize back
        restored = QueryPlan.model_validate_json(json_str)
        assert restored.query_type == plan.query_type
        assert restored.sub_queries == plan.sub_queries
        assert restored.reasoning == plan.reasoning
    
    def test_query_plan_all_query_types(self):
        """Test all QueryType enum values work correctly."""
        for query_type in QueryType:
            plan = QueryPlan(query_type=query_type)
            assert plan.query_type == query_type


class TestResearchResult:
    """Tests for the ResearchResult model."""
    
    def test_create_research_result(self):
        """Test creating a ResearchResult with all fields."""
        result = ResearchResult(
            content="Python is a high-level programming language.",
            source_url="https://example.com/python",
            source_title="Python Guide",
            source_type="web",
            relevance_score=0.95,
            sub_query="What is Python?",
            published_date="2026-01-15",
        )
        
        assert result.content == "Python is a high-level programming language."
        assert result.source_url == "https://example.com/python"
        assert result.relevance_score == 0.95
    
    def test_research_result_defaults(self):
        """Test ResearchResult default values."""
        result = ResearchResult(
            content="Some content",
            source_url="https://example.com",
        )
        
        assert result.source_title == ""
        assert result.source_type == "web"
        assert result.relevance_score == 0.0
        assert result.sub_query == ""
        assert result.published_date is None
        # retrieved_at should be set to current time
        assert result.retrieved_at is not None
    
    def test_research_result_relevance_score_bounds(self):
        """Test that relevance_score is bounded between 0 and 1."""
        # Valid scores
        result = ResearchResult(
            content="test",
            source_url="https://example.com",
            relevance_score=0.5,
        )
        assert result.relevance_score == 0.5
        
        # Boundary values
        result_zero = ResearchResult(
            content="test",
            source_url="https://example.com",
            relevance_score=0.0,
        )
        assert result_zero.relevance_score == 0.0
        
        result_one = ResearchResult(
            content="test",
            source_url="https://example.com",
            relevance_score=1.0,
        )
        assert result_one.relevance_score == 1.0


class TestClaim:
    """Tests for the Claim model."""
    
    def test_create_claim(self):
        """Test creating a Claim with all fields."""
        claim = Claim(
            statement="Python was created by Guido van Rossum.",
            source_url="https://python.org/history",
            source_title="Python History",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="In the late 1980s, Guido began working on Python.",
        )
        
        assert claim.statement == "Python was created by Guido van Rossum."
        assert claim.confidence == ConfidenceLevel.HIGH
        assert claim.category == "fact"
    
    def test_claim_defaults(self):
        """Test Claim default values."""
        claim = Claim(
            statement="Some claim",
            source_url="https://example.com",
        )
        
        assert claim.source_title == ""
        assert claim.confidence == ConfidenceLevel.MEDIUM
        assert claim.category == "fact"
        assert claim.context == ""
    
    def test_claim_serialization(self):
        """Test Claim JSON serialization."""
        claim = Claim(
            statement="Test statement",
            source_url="https://example.com",
            confidence=ConfidenceLevel.LOW,
        )
        
        data = claim.model_dump()
        assert data["statement"] == "Test statement"
        assert data["confidence"] == "low"


class TestVerificationResult:
    """Tests for the VerificationResult model."""
    
    def test_create_verification_result(self):
        """Test creating a VerificationResult."""
        claim = Claim(
            statement="Go was released in 2009.",
            source_url="https://example.com",
        )
        
        result = VerificationResult(
            claim=claim,
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.95,
            supporting_sources=["https://go.dev/blog", "https://wikipedia.org/go"],
            contradicting_sources=[],
            reasoning="Multiple authoritative sources confirm this.",
            verification_query="When was Go programming language released?",
        )
        
        assert result.verdict == VerificationVerdict.SUPPORTED
        assert result.confidence == 0.95
        assert len(result.supporting_sources) == 2
        assert len(result.contradicting_sources) == 0
    
    def test_verification_result_all_verdicts(self):
        """Test all verdict types."""
        claim = Claim(statement="Test", source_url="https://example.com")
        
        for verdict in VerificationVerdict:
            result = VerificationResult(claim=claim, verdict=verdict)
            assert result.verdict == verdict
    
    def test_verification_result_confidence_bounds(self):
        """Test confidence is bounded between 0 and 1."""
        claim = Claim(statement="Test", source_url="https://example.com")
        
        result = VerificationResult(
            claim=claim,
            verdict=VerificationVerdict.UNVERIFIABLE,
            confidence=0.3,
        )
        assert result.confidence == 0.3


class TestCitation:
    """Tests for the Citation model."""
    
    def test_create_citation(self):
        """Test creating a Citation."""
        citation = Citation(
            number=1,
            title="Python Documentation",
            url="https://docs.python.org",
            source_quality_score=0.9,
            source_type="web",
            snippet="Python is a programming language.",
        )
        
        assert citation.number == 1
        assert citation.title == "Python Documentation"
        assert citation.source_quality_score == 0.9
    
    def test_citation_number_validation(self):
        """Test that citation number must be >= 1."""
        with pytest.raises(ValidationError):
            Citation(
                number=0,  # Invalid: must be >= 1
                title="Test",
                url="https://example.com",
            )
    
    def test_citation_defaults(self):
        """Test Citation default values."""
        citation = Citation(
            number=1,
            title="Test",
            url="https://example.com",
        )
        
        assert citation.source_quality_score == 0.5
        assert citation.source_type == "web"
        assert citation.snippet == ""
        # accessed_date should be set to today
        assert citation.accessed_date is not None


class TestTraceEvent:
    """Tests for the TraceEvent model."""
    
    def test_create_trace_event(self):
        """Test creating a TraceEvent."""
        event = TraceEvent(
            agent=AgentName.RESEARCHER,
            action="web_search",
            detail="Searching for: Python async patterns",
            latency_ms=150.5,
            tokens_used=None,
            metadata={"query": "Python async patterns", "results": 5},
        )
        
        assert event.agent == AgentName.RESEARCHER
        assert event.action == "web_search"
        assert event.latency_ms == 150.5
        assert event.metadata["results"] == 5
    
    def test_trace_event_all_agents(self):
        """Test all agent names work."""
        for agent in AgentName:
            event = TraceEvent(
                agent=agent,
                action="test",
                detail="Test event",
            )
            assert event.agent == agent
    
    def test_trace_event_defaults(self):
        """Test TraceEvent default values."""
        event = TraceEvent(
            agent=AgentName.PLANNER,
            action="planning",
            detail="Creating query plan",
        )
        
        assert event.latency_ms is None
        assert event.tokens_used is None
        assert event.metadata == {}
        assert event.timestamp is not None


class TestReportModels:
    """Tests for Report-related models."""
    
    def test_create_report_section(self):
        """Test creating a ReportSection."""
        section = ReportSection(
            heading="Introduction",
            content="This report examines Python async patterns [1].",
            order=1,
        )
        
        assert section.heading == "Introduction"
        assert section.order == 1
    
    def test_create_report_metadata(self):
        """Test creating ReportMetadata."""
        metadata = ReportMetadata(
            query="Compare Python vs JavaScript",
            total_latency_ms=5000.0,
            llm_calls=8,
            total_tokens=15000,
            sources_searched=20,
            claims_verified=10,
            fact_check_pass_rate=0.9,
        )
        
        assert metadata.query == "Compare Python vs JavaScript"
        assert metadata.llm_calls == 8
        assert metadata.fact_check_pass_rate == 0.9
    
    def test_create_full_report(self):
        """Test creating a complete Report."""
        report = Report(
            title="Python vs JavaScript Comparison",
            summary="This report compares Python and JavaScript across key dimensions.",
            sections=[
                ReportSection(heading="Performance", content="...", order=1),
                ReportSection(heading="Ecosystem", content="...", order=2),
            ],
            confidence_summary="High confidence in technical comparisons.",
            citations=[
                Citation(number=1, title="MDN Docs", url="https://mdn.dev"),
                Citation(number=2, title="Python Docs", url="https://python.org"),
            ],
            metadata=ReportMetadata(query="Compare Python vs JavaScript"),
        )
        
        assert report.title == "Python vs JavaScript Comparison"
        assert len(report.sections) == 2
        assert len(report.citations) == 2


class TestHITLFeedback:
    """Tests for HITLFeedback model."""
    
    def test_create_approve_feedback(self):
        """Test creating approve feedback."""
        feedback = HITLFeedback(action="approve")
        assert feedback.action == "approve"
        assert feedback.topic is None
        assert feedback.correction is None
    
    def test_create_dig_deeper_feedback(self):
        """Test creating dig deeper feedback with topic."""
        feedback = HITLFeedback(
            action="dig_deeper",
            topic="performance benchmarks",
        )
        assert feedback.action == "dig_deeper"
        assert feedback.topic == "performance benchmarks"
    
    def test_create_correct_feedback(self):
        """Test creating correction feedback."""
        feedback = HITLFeedback(
            action="correct",
            correction="Python 3.0 was released in 2008, not 2009.",
        )
        assert feedback.action == "correct"
        assert feedback.correction is not None


class TestAgentState:
    """Tests for AgentState TypedDict and helper functions."""
    
    def test_create_initial_state(self):
        """Test creating initial state from a query."""
        state = create_initial_state("What is machine learning?")
        
        assert state["query"] == "What is machine learning?"
        assert state["status"] == PipelineStatus.PLANNING
        assert state["iteration_count"] == 0
        assert state["errors"] == []
        assert state["agent_trace"] == []
        assert state["research_results"] == []
        assert state["claims"] == []
    
    def test_add_trace_event(self):
        """Test adding trace events to state."""
        state = create_initial_state("Test query")
        
        updated_state = add_trace_event(
            state,
            agent=AgentName.PLANNER,
            action="planning",
            detail="Starting query analysis",
            latency_ms=100.0,
        )
        
        assert len(updated_state["agent_trace"]) == 1
        event = updated_state["agent_trace"][0]
        assert event.agent == AgentName.PLANNER
        assert event.latency_ms == 100.0
        
        # Original state should be unchanged (immutable update)
        assert len(state["agent_trace"]) == 0
    
    def test_add_multiple_trace_events(self):
        """Test adding multiple trace events."""
        state = create_initial_state("Test query")
        
        state = add_trace_event(state, AgentName.PLANNER, "start", "Starting")
        state = add_trace_event(state, AgentName.PLANNER, "end", "Finished")
        state = add_trace_event(state, AgentName.RESEARCHER, "search", "Searching")
        
        assert len(state["agent_trace"]) == 3
        assert state["agent_trace"][0].agent == AgentName.PLANNER
        assert state["agent_trace"][2].agent == AgentName.RESEARCHER
    
    def test_add_error(self):
        """Test adding errors to state."""
        state = create_initial_state("Test query")
        
        updated_state = add_error(state, "Search API timeout")
        
        assert len(updated_state["errors"]) == 1
        assert updated_state["errors"][0] == "Search API timeout"
        
        # Original state unchanged
        assert len(state["errors"]) == 0
    
    def test_add_multiple_errors(self):
        """Test adding multiple errors."""
        state = create_initial_state("Test query")
        
        state = add_error(state, "Error 1")
        state = add_error(state, "Error 2")
        
        assert len(state["errors"]) == 2


class TestEdgeCases:
    """Tests for edge cases and special characters."""
    
    def test_empty_query(self):
        """Test handling empty query string."""
        state = create_initial_state("")
        assert state["query"] == ""
    
    def test_very_long_query(self):
        """Test handling very long query strings."""
        long_query = "a" * 10000
        state = create_initial_state(long_query)
        assert len(state["query"]) == 10000
    
    def test_special_characters_in_query(self):
        """Test handling special characters in query."""
        special_query = "What's the difference between C++ & C#? <test> \"quotes\""
        state = create_initial_state(special_query)
        assert state["query"] == special_query
    
    def test_unicode_characters(self):
        """Test handling unicode characters."""
        unicode_query = "What is Kubernetes? (Kubernetes)"
        create_initial_state(unicode_query)
        
        # Also test in claims
        claim = Claim(
            statement="Kubernetes comes from Greek 'kybernetes'",
            source_url="https://example.com",
        )
        assert "kybernetes" in claim.statement
    
    def test_newlines_in_content(self):
        """Test handling newlines in content fields."""
        result = ResearchResult(
            content="Line 1\nLine 2\nLine 3",
            source_url="https://example.com",
        )
        assert "\n" in result.content
    
    def test_empty_lists(self):
        """Test handling empty lists in state."""
        state = create_initial_state("Test")
        
        assert state["sub_queries"] == []
        assert state["research_results"] == []
        assert state["claims"] == []
        assert state["verification_results"] == []
        assert state["citations"] == []


class TestSerialization:
    """Tests for full state serialization."""
    
    def test_serialize_full_state_to_json(self):
        """Test serializing a complete state to JSON."""
        state = create_initial_state("Test query")
        
        # Add some data
        state["sub_queries"] = ["sub1", "sub2"]
        state["research_results"].append(
            ResearchResult(
                content="Test content",
                source_url="https://example.com",
                relevance_score=0.8,
            )
        )
        
        # Serialize nested Pydantic models
        plan_json = state["plan"].model_dump_json()
        assert "query_type" in plan_json
        
        # Can serialize research results
        for result in state["research_results"]:
            result_json = result.model_dump_json()
            assert "content" in result_json
    
    def test_state_with_all_pipeline_statuses(self):
        """Test all pipeline status values."""
        state = create_initial_state("Test")
        
        for status in PipelineStatus:
            state["status"] = status
            assert state["status"] == status
