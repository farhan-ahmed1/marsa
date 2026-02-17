"""Integration tests for the Planner Agent.

These tests use real LLM API calls and consume API quota.
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
from graph.state import (  # noqa: E402
    ComplexityLevel,
    QueryType,
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
