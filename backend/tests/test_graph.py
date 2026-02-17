"""Unit tests for the graph module (workflow, checkpointer, state utilities).

These tests focus on the workflow assembly, routing logic, and checkpointer
configuration without making real LLM calls.
"""

import sys
import tempfile
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from graph.state import (  # noqa: E402
    AgentName,
    AgentState,
    Claim,
    ConfidenceLevel,
    PipelineStatus,
    TraceEvent,
    VerificationResult,
    VerificationVerdict,
    add_error,
    add_trace_event,
    create_initial_state,
)
from graph.checkpointer import (  # noqa: E402
    create_checkpointer,
)


# ---------------------------------------------------------------------------
# Checkpointer Tests
# ---------------------------------------------------------------------------


class TestCreateCheckpointer:
    """Tests for checkpointer creation."""

    def test_create_checkpointer_returns_context_manager(self):
        """Test that create_checkpointer returns a context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_checkpoints.db"
            checkpointer = create_checkpointer(str(db_path))
            # Should return an async context manager
            assert checkpointer is not None
            assert hasattr(checkpointer, "__aenter__")
            assert hasattr(checkpointer, "__aexit__")

    def test_create_checkpointer_creates_directory(self):
        """Test that create_checkpointer creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "dir" / "checkpoints.db"
            # Directory shouldn't exist yet
            assert not db_path.parent.exists()
            
            create_checkpointer(str(db_path))
            # Directory should be created
            assert db_path.parent.exists()


# ---------------------------------------------------------------------------
# State Helper Function Tests  
# ---------------------------------------------------------------------------


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_creates_state_with_query(self):
        """Test creating initial state with just a query."""
        state = create_initial_state("What is Python?")
        
        assert state["query"] == "What is Python?"
        assert state["plan"] is not None  # Plan is pre-initialized
        assert state["sub_queries"] == []
        assert state["research_results"] == []
        assert state["claims"] == []
        assert state["verification_results"] == []
        assert state["source_scores"] == {}
        assert state["report"] == ""
        assert state["citations"] == []
        assert state["agent_trace"] == []
        assert state["iteration_count"] == 0
        assert state["status"] == PipelineStatus.PLANNING
        assert state["errors"] == []

    def test_creates_state_with_started_at(self):
        """Test that started_at timestamp is set."""
        state = create_initial_state("Test query")
        
        assert "started_at" in state
        assert state["started_at"] is not None
        assert len(state["started_at"]) > 0


class TestAddTraceEvent:
    """Tests for add_trace_event helper function."""

    def test_adds_trace_event_to_empty_state(self):
        """Test adding trace event to state with no existing traces."""
        state = create_initial_state("Test")
        
        new_state = add_trace_event(
            state,
            agent=AgentName.PLANNER,
            action="test",
            detail="Test event",
        )
        
        assert len(new_state["agent_trace"]) == 1
        assert new_state["agent_trace"][0].agent == AgentName.PLANNER
        assert new_state["agent_trace"][0].action == "test"
        assert new_state["agent_trace"][0].detail == "Test event"

    def test_adds_trace_event_to_existing_traces(self):
        """Test adding trace event preserves existing traces."""
        state = create_initial_state("Test")
        
        state = add_trace_event(
            state,
            agent=AgentName.PLANNER,
            action="test1",
            detail="Event 1",
        )
        state = add_trace_event(
            state,
            agent=AgentName.RESEARCHER,
            action="test2",
            detail="Event 2",
        )
        
        assert len(state["agent_trace"]) == 2
        assert state["agent_trace"][0].agent == AgentName.PLANNER
        assert state["agent_trace"][1].agent == AgentName.RESEARCHER


class TestAddError:
    """Tests for add_error helper function."""

    def test_adds_error_to_state(self):
        """Test adding error to state."""
        state = create_initial_state("Test")
        
        new_state = add_error(state, "Test error message")
        
        assert len(new_state["errors"]) == 1
        assert new_state["errors"][0] == "Test error message"

    def test_adds_multiple_errors(self):
        """Test adding multiple errors."""
        state = create_initial_state("Test")
        
        state = add_error(state, "Error 1")
        state = add_error(state, "Error 2")
        state = add_error(state, "Error 3")
        
        assert len(state["errors"]) == 3
        assert state["errors"] == ["Error 1", "Error 2", "Error 3"]


# ---------------------------------------------------------------------------
# Workflow Routing Tests
# ---------------------------------------------------------------------------


class TestWorkflowRouting:
    """Tests for workflow routing logic."""

    def test_should_loop_back_all_supported(self):
        """Test that all supported claims proceed to synthesizer."""
        from agents.fact_checker import should_loop_back
        
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Example",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="",
        )
        
        state: AgentState = {
            "verification_results": [
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.9,
                    supporting_sources=[],
                    contradicting_sources=[],
                    reasoning="",
                    verification_query="",
                ),
            ],
            "iteration_count": 0,
        }
        
        result = should_loop_back(state)
        assert result == "synthesizer"

    def test_should_loop_back_high_failure_rate(self):
        """Test that high failure rate triggers loopback."""
        from agents.fact_checker import should_loop_back
        
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Example",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="",
        )
        
        # 2 out of 3 claims fail (66% failure rate)
        state: AgentState = {
            "verification_results": [
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.CONTRADICTED,
                    confidence=0.9,
                    supporting_sources=[],
                    contradicting_sources=[],
                    reasoning="",
                    verification_query="",
                ),
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.UNVERIFIABLE,
                    confidence=0.5,
                    supporting_sources=[],
                    contradicting_sources=[],
                    reasoning="",
                    verification_query="",
                ),
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.8,
                    supporting_sources=[],
                    contradicting_sources=[],
                    reasoning="",
                    verification_query="",
                ),
            ],
            "iteration_count": 0,
        }
        
        result = should_loop_back(state)
        assert result == "researcher"

    def test_should_loop_back_max_iterations(self):
        """Test that max iterations prevents infinite loops."""
        from agents.fact_checker import should_loop_back
        
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Example",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="",
        )
        
        # All claims fail but we've hit max iterations
        state: AgentState = {
            "verification_results": [
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.CONTRADICTED,
                    confidence=0.9,
                    supporting_sources=[],
                    contradicting_sources=[],
                    reasoning="",
                    verification_query="",
                ),
            ],
            "iteration_count": 2,  # Max iterations reached
        }
        
        result = should_loop_back(state)
        assert result == "synthesizer"


# ---------------------------------------------------------------------------
# Workflow Creation Tests
# ---------------------------------------------------------------------------


class TestWorkflowCreation:
    """Tests for workflow creation and compilation."""

    def test_create_workflow_returns_compiled_graph(self):
        """Test that create_workflow returns a compiled graph."""
        from graph.workflow import create_workflow
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "checkpoints.db"
            workflow = create_workflow(str(db_path))
            
            assert workflow is not None
            # Should have nodes attribute
            assert hasattr(workflow, "nodes")

    def test_route_after_fact_check_function(self):
        """Test the route_after_fact_check function."""
        from graph.workflow import route_after_fact_check
        
        claim = Claim(
            statement="Test",
            source_url="https://example.com",
            source_title="Example",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="",
        )
        
        # All supported - should synthesize
        state = {
            "verification_results": [
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.9,
                    supporting_sources=[],
                    contradicting_sources=[],
                    reasoning="",
                    verification_query="",
                ),
            ],
            "iteration_count": 0,
        }
        
        result = route_after_fact_check(state)
        assert result == "synthesizer"


# ---------------------------------------------------------------------------
# Trace Event Tests
# ---------------------------------------------------------------------------


class TestTraceEvent:
    """Tests for TraceEvent model."""

    def test_create_trace_event(self):
        """Test creating a trace event."""
        event = TraceEvent(
            agent=AgentName.PLANNER,
            action="plan",
            detail="Created query plan",
            tokens_used=100,
            latency_ms=250.5,
        )
        
        assert event.agent == AgentName.PLANNER
        assert event.action == "plan"
        assert event.detail == "Created query plan"
        assert event.tokens_used == 100
        assert event.latency_ms == 250.5
        assert event.timestamp is not None
        assert event.metadata == {}

    def test_trace_event_defaults(self):
        """Test trace event default values."""
        event = TraceEvent(
            agent=AgentName.RESEARCHER,
            action="search",
            detail="test",
        )
        
        assert event.tokens_used is None
        assert event.latency_ms is None
        assert event.metadata == {}

    def test_trace_event_with_metadata(self):
        """Test trace event with metadata."""
        event = TraceEvent(
            agent=AgentName.FACT_CHECKER,
            action="verify",
            detail="Verified claim",
            metadata={"claim_id": "123", "verdict": "supported"},
        )
        
        assert event.metadata["claim_id"] == "123"
        assert event.metadata["verdict"] == "supported"


# ---------------------------------------------------------------------------
# Parallel Execution Routing Tests
# ---------------------------------------------------------------------------


class TestParallelRouting:
    """Tests for parallel sub-query routing."""

    def test_route_sub_queries_parallel_multiple_queries(self):
        """Test that multiple sub-queries are routed to parallel execution."""
        from langgraph.types import Send
        from graph.workflow import route_sub_queries
        from graph.state import QueryPlan, QueryType, SearchStrategy, ComplexityLevel
        
        plan = QueryPlan(
            query_type=QueryType.COMPARISON,
            sub_queries=["Query 1", "Query 2", "Query 3"],
            parallel=True,
            needs_fact_check=True,
            search_strategy=SearchStrategy.HYBRID,
            estimated_complexity=ComplexityLevel.MEDIUM,
        )
        
        state = create_initial_state("Compare A, B, and C")
        state["plan"] = plan
        
        result = route_sub_queries(state)
        
        # Should return a list of Send objects
        assert isinstance(result, list)
        assert len(result) == 3
        for send_obj in result:
            assert isinstance(send_obj, Send)
            assert send_obj.node == "research_sub_query"

    def test_route_sub_queries_sequential_single_query(self):
        """Test that single query is routed sequentially."""
        from graph.workflow import route_sub_queries
        from graph.state import QueryPlan, QueryType, SearchStrategy, ComplexityLevel
        
        plan = QueryPlan(
            query_type=QueryType.FACTUAL,
            sub_queries=["Single query"],
            parallel=False,
            needs_fact_check=True,
            search_strategy=SearchStrategy.WEB_ONLY,
            estimated_complexity=ComplexityLevel.LOW,
        )
        
        state = create_initial_state("What is X?")
        state["plan"] = plan
        
        result = route_sub_queries(state)
        
        # Should return string for sequential
        assert result == "research_sequential"

    def test_route_sub_queries_no_plan(self):
        """Test routing with no plan returns sequential."""
        from graph.workflow import route_sub_queries
        
        state = create_initial_state("Test query")
        state["plan"] = None
        
        result = route_sub_queries(state)
        assert result == "research_sequential"


# ---------------------------------------------------------------------------
# HITL Feedback Routing Tests
# ---------------------------------------------------------------------------


class TestHITLRouting:
    """Tests for human-in-the-loop feedback routing."""

    def test_route_after_hitl_approve(self):
        """Test that 'approve' action routes to synthesizer."""
        from graph.workflow import route_after_hitl_feedback
        from graph.state import HITLFeedback
        
        state = create_initial_state("Test query")
        state["hitl_feedback"] = HITLFeedback(action="approve")
        state["verification_results"] = []
        state["iteration_count"] = 0
        
        result = route_after_hitl_feedback(state)
        assert result == "synthesizer"

    def test_route_after_hitl_dig_deeper(self):
        """Test that 'dig_deeper' action routes to researcher."""
        from graph.workflow import route_after_hitl_feedback
        from graph.state import HITLFeedback
        
        state = create_initial_state("Test query")
        state["hitl_feedback"] = HITLFeedback(
            action="dig_deeper",
            topic="Performance benchmarks",
        )
        state["verification_results"] = []
        state["iteration_count"] = 0
        
        result = route_after_hitl_feedback(state)
        assert result == "researcher"

    def test_route_after_hitl_abort(self):
        """Test that 'abort' action routes to end."""
        from graph.workflow import route_after_hitl_feedback
        from graph.state import HITLFeedback
        
        state = create_initial_state("Test query")
        state["hitl_feedback"] = HITLFeedback(action="abort")
        state["verification_results"] = []
        state["iteration_count"] = 0
        
        result = route_after_hitl_feedback(state)
        assert result == "end"

    def test_route_after_hitl_no_feedback(self):
        """Test routing without feedback falls back to fact-check routing."""
        from graph.workflow import route_after_hitl_feedback
        
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Example",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="",
        )
        
        state = create_initial_state("Test query")
        state["hitl_feedback"] = None
        state["verification_results"] = [
            VerificationResult(
                claim=claim,
                verdict=VerificationVerdict.SUPPORTED,
                confidence=0.9,
                supporting_sources=[],
                contradicting_sources=[],
                reasoning="",
                verification_query="",
            ),
        ]
        state["iteration_count"] = 0
        
        result = route_after_hitl_feedback(state)
        # Should proceed to synthesizer since claims are supported
        assert result == "synthesizer"


# ---------------------------------------------------------------------------
# Workflow Creation Tests (Parallel Mode)
# ---------------------------------------------------------------------------


class TestParallelWorkflowCreation:
    """Tests for workflow creation with parallel execution support."""

    def test_create_workflow_with_parallel(self):
        """Test creating workflow with parallel execution enabled."""
        from graph.workflow import create_workflow
        
        workflow = create_workflow(enable_parallel=True)
        
        assert workflow is not None
        # Should have the parallel research nodes
        
    def test_create_workflow_without_parallel(self):
        """Test creating workflow with parallel execution disabled."""
        from graph.workflow import create_workflow
        
        workflow = create_workflow(enable_parallel=False)
        
        assert workflow is not None

    def test_create_workflow_with_hitl_and_parallel(self):
        """Test creating workflow with both HITL and parallel enabled."""
        from graph.workflow import create_workflow
        
        workflow = create_workflow(enable_hitl=True, enable_parallel=True)
        
        assert workflow is not None
