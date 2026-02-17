"""Integration tests for Agent components.

These tests use real LLM and API calls and consume API quota.
Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

import sys
from pathlib import Path

import pytest

# Add backend directory to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from agents.planner import create_query_plan  # noqa: E402
from agents.researcher import extract_claims, research_node  # noqa: E402
from graph.state import (  # noqa: E402
    AgentState,
    ComplexityLevel,
    PipelineStatus,
    QueryPlan,
    QueryType,
    ResearchResult,
    SearchStrategy,
)


@pytest.mark.integration
class TestPlannerIntegration:
    """Integration tests that use real LLM calls."""
    
    @pytest.mark.asyncio
    async def test_real_simple_factual_query(self):
        """Test real LLM call for a simple factual query."""
        plan = await create_query_plan("What is gRPC?")
        
        assert plan.query_type in [QueryType.FACTUAL, QueryType.DEFINITION]
        assert len(plan.sub_queries) >= 1
        assert plan.estimated_complexity == ComplexityLevel.LOW
    
    @pytest.mark.asyncio
    async def test_real_comparison_query(self):
        """Test real LLM call for a comparison query."""
        plan = await create_query_plan(
            "Compare React, Vue, and Svelte for building SPAs"
        )
        
        assert plan.query_type == QueryType.COMPARISON
        assert len(plan.sub_queries) >= 3  # At least one per framework
        assert plan.parallel is True
    
    @pytest.mark.asyncio
    async def test_real_exploratory_query(self):
        """Test real LLM call for an exploratory query."""
        plan = await create_query_plan(
            "What are the latest trends in AI agent frameworks?"
        )
        
        assert plan.query_type == QueryType.EXPLORATORY
        assert len(plan.sub_queries) >= 2
        assert plan.search_strategy in [SearchStrategy.WEB_ONLY, SearchStrategy.HYBRID]


@pytest.mark.integration
class TestResearcherIntegration:
    """Integration tests for Researcher agent that use real API calls."""
    
    @pytest.mark.asyncio
    async def test_real_extract_claims_from_results(self):
        """Test real LLM call for claim extraction from research results."""
        # Create sample research results
        results = [
            ResearchResult(
                content="Python was created by Guido van Rossum in the late 1980s at CWI in the Netherlands. It was first released in 1991.",
                source_url="https://docs.python.org/3/faq/general.html",
                source_title="Python General FAQ",
                source_type="web",
                relevance_score=0.95,
                sub_query="Who created Python?",
            ),
            ResearchResult(
                content="Python 3.0, a major, backwards-incompatible release, was released on December 3, 2008 after a long period of testing.",
                source_url="https://en.wikipedia.org/wiki/Python_(programming_language)",
                source_title="Python (programming language) - Wikipedia",
                source_type="web",
                relevance_score=0.90,
                sub_query="When was Python 3.0 released?",
            ),
        ]
        
        # Extract claims using real LLM
        claims = await extract_claims(results, "Who created Python and when was Python 3.0 released?")
        
        # Verify claims were extracted
        assert len(claims) >= 1
        assert all(claim.statement for claim in claims)
        assert all(claim.source_url for claim in claims)
        assert all(claim.confidence for claim in claims)
        
        # Check that at least one claim mentions Python
        claim_texts = " ".join(c.statement for c in claims)
        assert "Python" in claim_texts or "python" in claim_texts.lower()
    
    @pytest.mark.asyncio
    async def test_real_research_node_simple_query(self):
        """Test real research_node with actual API calls for a simple query.
        
        WARNING: This test makes real API calls to:
        - Tavily (web search)
        - ChromaDB (document store)
        - Anthropic Claude (claim extraction)
        
        It will consume API quota!
        """
        # Create a simple query plan
        plan = QueryPlan(
            query_type=QueryType.FACTUAL,
            sub_queries=["What is gRPC?"],
            parallel=False,
            needs_fact_check=True,
            search_strategy=SearchStrategy.WEB_ONLY,  # Web only to be faster
            estimated_complexity=ComplexityLevel.LOW,
            reasoning="Simple factual query for testing",
        )
        
        state: AgentState = {
            "query": "What is gRPC?",
            "plan": plan,
            "agent_trace": [],
        }
        
        # Execute research node with real APIs
        result = await research_node(state)
        
        # Verify results structure
        assert "research_results" in result
        assert "claims" in result
        assert "source_scores" in result
        assert "status" in result
        assert "agent_trace" in result
        
        # Verify we got some results (if APIs are working)
        if result.get("research_results"):
            assert len(result["research_results"]) > 0
            assert all(r.source_url for r in result["research_results"])
            assert all(r.content for r in result["research_results"])
        
        # Verify claims were extracted (if we got results)
        if result.get("claims"):
            assert len(result["claims"]) > 0
            assert all(c.statement for c in result["claims"])
        
        # Verify source scores
        assert isinstance(result["source_scores"], dict)
        
        # Verify trace events were generated
        assert len(result["agent_trace"]) > 0
        
        # Verify status is valid
        assert result["status"] in [
            PipelineStatus.FACT_CHECKING.value,
            PipelineStatus.SYNTHESIZING.value,
            PipelineStatus.FAILED.value,
        ]
    
    @pytest.mark.asyncio
    async def test_real_research_node_handles_no_results(self):
        """Test research_node gracefully handles queries with no results.
        
        WARNING: This test makes real API calls and consumes quota!
        """
        # Create a query unlikely to return results
        plan = QueryPlan(
            query_type=QueryType.FACTUAL,
            sub_queries=["xyzabc123nonexistentqueryterm456789"],
            parallel=False,
            needs_fact_check=True,
            search_strategy=SearchStrategy.WEB_ONLY,
            estimated_complexity=ComplexityLevel.LOW,
            reasoning="Test query with no results",
        )
        
        state: AgentState = {
            "query": "xyzabc123nonexistentqueryterm456789",
            "plan": plan,
            "agent_trace": [],
        }
        
        # Execute research node
        result = await research_node(state)
        
        # Should handle gracefully without crashing
        assert "status" in result
        assert "research_results" in result
        assert "claims" in result
        assert "agent_trace" in result
        
        # Should have some trace events even if no results
        assert len(result["agent_trace"]) > 0
