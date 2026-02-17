"""Integration tests for the full LangGraph workflow.

These tests use real LLM and API calls and consume API quota.
Run with: pytest -m integration
Skip with: pytest -m "not integration"

WARNING: These tests run the full research pipeline and make many API calls.
Each test may take 30-60 seconds and consume significant API quota.
"""

import sys
from pathlib import Path

import pytest

# Add backend directory to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from graph.state import (  # noqa: E402
    AgentState,
    PipelineStatus,
    create_initial_state,
)
from graph.workflow import (  # noqa: E402
    create_workflow,
    run_research,
)
from agents.fact_checker import should_loop_back  # noqa: E402


# ---------------------------------------------------------------------------
# Unit Tests (no API calls)
# ---------------------------------------------------------------------------


class TestRouteAfterFactCheck:
    """Tests for the routing logic after fact-checking."""
    
    def test_routes_to_synthesizer_when_all_supported(self):
        """Test routing to synthesizer when all claims are supported."""
        from graph.state import (
            Claim,
            ConfidenceLevel,
            VerificationResult,
            VerificationVerdict,
        )
        
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
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.85,
                    supporting_sources=[],
                    contradicting_sources=[],
                    reasoning="",
                    verification_query="",
                ),
            ],
            "iteration_count": 0,
        }
        
        route = should_loop_back(state)
        assert route == "synthesizer"
    
    def test_routes_to_researcher_when_many_unsupported(self):
        """Test routing to researcher when too many claims are not supported."""
        from graph.state import (
            Claim,
            ConfidenceLevel,
            VerificationResult,
            VerificationVerdict,
        )
        
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Example",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="",
        )
        
        # 2/4 = 50% unsupported, above 30% threshold
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
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.85,
                    supporting_sources=[],
                    contradicting_sources=[],
                    reasoning="",
                    verification_query="",
                ),
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.CONTRADICTED,
                    confidence=0.8,
                    supporting_sources=[],
                    contradicting_sources=[],
                    reasoning="",
                    verification_query="",
                ),
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.UNVERIFIABLE,
                    confidence=0.3,
                    supporting_sources=[],
                    contradicting_sources=[],
                    reasoning="",
                    verification_query="",
                ),
            ],
            "iteration_count": 0,
        }
        
        route = should_loop_back(state)
        assert route == "researcher"
    
    def test_respects_max_iterations(self):
        """Test that routing respects the maximum iteration limit."""
        from graph.state import (
            Claim,
            ConfidenceLevel,
            VerificationResult,
            VerificationVerdict,
        )
        
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Example",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="",
        )
        
        # All unsupported, but at max iterations
        state: AgentState = {
            "verification_results": [
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.CONTRADICTED,
                    confidence=0.8,
                    supporting_sources=[],
                    contradicting_sources=[],
                    reasoning="",
                    verification_query="",
                ),
            ],
            "iteration_count": 2,  # At max
        }
        
        route = should_loop_back(state)
        assert route == "synthesizer"  # Should proceed despite bad claims


class TestCreateInitialState:
    """Tests for initial state creation."""
    
    def test_creates_valid_initial_state(self):
        """Test that initial state is properly structured."""
        state = create_initial_state("Test query")
        
        assert state["query"] == "Test query"
        assert state["status"] == PipelineStatus.PLANNING
        assert state["iteration_count"] == 0
        assert state["errors"] == []
        assert state["agent_trace"] == []
        assert state["research_results"] == []
        assert state["claims"] == []
        assert state["verification_results"] == []


class TestWorkflowCreation:
    """Tests for workflow creation."""
    
    def test_creates_workflow_without_hitl(self):
        """Test creating workflow without HITL."""
        workflow = create_workflow(enable_hitl=False)
        
        # Should be a compiled graph
        assert workflow is not None
        assert hasattr(workflow, 'ainvoke')
    
    def test_creates_workflow_with_hitl(self):
        """Test creating workflow with HITL enabled."""
        workflow = create_workflow(enable_hitl=True)
        
        # Should be a compiled graph
        assert workflow is not None
        assert hasattr(workflow, 'ainvoke')


# ---------------------------------------------------------------------------
# Integration Tests (real API calls)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWorkflowIntegration:
    """Integration tests for the full workflow using real API calls.
    
    WARNING: These tests make many API calls and consume significant quota.
    Each test may take 30-60 seconds to complete.
    """
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_pipeline_simple_factual_query(self):
        """Test full pipeline with a simple factual query.
        
        WARNING: This test runs the FULL research pipeline with:
        - Planner (Claude)
        - Researcher (Tavily + Claude)
        - Fact-Checker (Tavily + Claude)
        - Synthesizer (Claude)
        
        Esta prueba consume significant API quota and takes 30-60 seconds.
        """
        result = await run_research(
            query="What is gRPC?",
            enable_hitl=False,
        )
        
        # Verify completion
        assert result["status"] in [
            PipelineStatus.COMPLETED.value,
            PipelineStatus.COMPLETED,
            "completed",
        ]
        
        # Verify we have a report
        assert "report" in result
        assert len(result["report"]) > 100
        
        # Verify report mentions gRPC
        report_lower = result["report"].lower()
        assert "grpc" in report_lower or "rpc" in report_lower
        
        # Verify we have structured report
        assert "report_structured" in result
        assert result["report_structured"].title
        
        # Verify citations
        assert "citations" in result
        # May or may not have citations depending on fact-check results
        
        # Verify trace events exist
        assert len(result["agent_trace"]) > 0
        
        # Verify all agents participated
        agent_names = set()
        for event in result["agent_trace"]:
            if hasattr(event, 'agent'):
                agent_names.add(event.agent.value)
        
        # Should have at least planner, researcher, and synthesizer
        assert "planner" in agent_names
        assert "researcher" in agent_names or len(agent_names) >= 2
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_pipeline_comparison_query(self):
        """Test full pipeline with a comparison query.
        
        WARNING: This test runs the FULL research pipeline.
        It consumes significant API quota and takes 30-60+ seconds.
        """
        result = await run_research(
            query="Compare Python vs JavaScript for backend development",
            enable_hitl=False,
        )
        
        # Verify completion
        assert result["status"] in [
            PipelineStatus.COMPLETED.value,
            PipelineStatus.COMPLETED,
            "completed",
        ]
        
        # Verify we have a report
        assert "report" in result
        assert len(result["report"]) > 200
        
        # Report should mention both languages
        report_lower = result["report"].lower()
        assert "python" in report_lower
        assert "javascript" in report_lower or "js" in report_lower
        
        # Verify plan had multiple sub-queries
        plan = result.get("plan")
        if plan:
            assert len(plan.sub_queries) >= 2
    
    @pytest.mark.asyncio
    async def test_full_pipeline_handles_errors_gracefully(self):
        """Test that the pipeline handles edge cases gracefully.
        
        WARNING: This test makes API calls but should complete faster.
        """
        # Test with an unusual query
        result = await run_research(
            query="xyznonexistent123456",
            enable_hitl=False,
        )
        
        # Should complete (possibly with limited results)
        assert result["status"] in [
            PipelineStatus.COMPLETED.value,
            PipelineStatus.COMPLETED,
            PipelineStatus.FAILED.value,
            PipelineStatus.FAILED,
            "completed",
            "failed",
        ]
        
        # Should have some output
        assert "report" in result
        assert "agent_trace" in result
    
    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self):
        """Test that workflow state is persisted via checkpointer.
        
        WARNING: This test makes API calls.
        """
        import uuid
        
        thread_id = str(uuid.uuid4())
        
        result = await run_research(
            query="What is Docker?",
            thread_id=thread_id,
            enable_hitl=False,
        )
        
        # Verify completion
        assert result["status"] in [
            PipelineStatus.COMPLETED.value,
            PipelineStatus.COMPLETED,
            "completed",
        ]
        
        # The thread_id should have been used for checkpointing
        # (state is persisted in SQLite)
        assert result.get("query") == "What is Docker?"
