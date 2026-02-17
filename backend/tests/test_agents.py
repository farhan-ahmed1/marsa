"""Tests for the Planner Agent.

Tests the query planning, decomposition, and classification logic
with mocked LLM calls and real integration tests.
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from agents.planner import (  # noqa: E402
    PLANNER_SYSTEM_PROMPT,
    _parse_query_plan,
    create_query_plan,
    planner_node,
)
from graph.state import (  # noqa: E402
    ComplexityLevel,
    QueryType,
    SearchStrategy,
)


class TestParseQueryPlan:
    """Tests for _parse_query_plan parsing logic."""
    
    def test_parse_valid_json(self):
        """Test parsing a valid JSON response."""
        response = json.dumps({
            "query_type": "factual",
            "sub_queries": ["What is gRPC?"],
            "parallel": True,
            "needs_fact_check": False,
            "search_strategy": "web_only",
            "estimated_complexity": "low",
            "reasoning": "Simple definition query"
        })
        
        plan = _parse_query_plan(response)
        
        assert plan.query_type == QueryType.FACTUAL
        assert plan.sub_queries == ["What is gRPC?"]
        assert plan.parallel is True
        assert plan.needs_fact_check is False
        assert plan.search_strategy == SearchStrategy.WEB_ONLY
        assert plan.estimated_complexity == ComplexityLevel.LOW
        assert plan.reasoning == "Simple definition query"
    
    def test_parse_json_with_markdown_fences(self):
        """Test parsing JSON wrapped in markdown code fences."""
        response = """```json
{
    "query_type": "comparison",
    "sub_queries": ["Compare A", "Compare B"],
    "parallel": true,
    "needs_fact_check": true,
    "search_strategy": "hybrid",
    "estimated_complexity": "medium",
    "reasoning": "Comparison query"
}
```"""
        
        plan = _parse_query_plan(response)
        
        assert plan.query_type == QueryType.COMPARISON
        assert len(plan.sub_queries) == 2
    
    def test_parse_json_with_simple_fences(self):
        """Test parsing JSON wrapped in simple code fences."""
        response = """```
{
    "query_type": "exploratory",
    "sub_queries": ["Trend 1", "Trend 2"],
    "parallel": true,
    "needs_fact_check": true,
    "search_strategy": "web_only",
    "estimated_complexity": "high",
    "reasoning": "Open-ended research"
}
```"""
        
        plan = _parse_query_plan(response)
        
        assert plan.query_type == QueryType.EXPLORATORY
    
    def test_parse_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        response = "This is not valid JSON"
        
        with pytest.raises(ValueError) as exc_info:
            _parse_query_plan(response)
        
        assert "Invalid JSON" in str(exc_info.value)
    
    def test_parse_unknown_query_type_defaults_to_factual(self):
        """Test that unknown query types default to factual."""
        response = json.dumps({
            "query_type": "unknown_type",
            "sub_queries": ["Test query"],
            "parallel": True,
            "needs_fact_check": True,
            "search_strategy": "hybrid",
            "estimated_complexity": "medium",
            "reasoning": ""
        })
        
        plan = _parse_query_plan(response)
        
        assert plan.query_type == QueryType.FACTUAL
    
    def test_parse_unknown_strategy_defaults_to_hybrid(self):
        """Test that unknown search strategies default to hybrid."""
        response = json.dumps({
            "query_type": "factual",
            "sub_queries": ["Test"],
            "parallel": True,
            "needs_fact_check": True,
            "search_strategy": "unknown_strategy",
            "estimated_complexity": "medium",
            "reasoning": ""
        })
        
        plan = _parse_query_plan(response)
        
        assert plan.search_strategy == SearchStrategy.HYBRID
    
    def test_parse_all_query_types(self):
        """Test parsing all valid query types."""
        query_types = [
            ("factual", QueryType.FACTUAL),
            ("comparison", QueryType.COMPARISON),
            ("exploratory", QueryType.EXPLORATORY),
            ("opinion", QueryType.OPINION),
            ("howto", QueryType.HOWTO),
            ("definition", QueryType.DEFINITION),
        ]
        
        for raw_type, expected_enum in query_types:
            response = json.dumps({
                "query_type": raw_type,
                "sub_queries": ["Test"],
                "parallel": True,
                "needs_fact_check": True,
                "search_strategy": "hybrid",
                "estimated_complexity": "medium",
                "reasoning": ""
            })
            
            plan = _parse_query_plan(response)
            assert plan.query_type == expected_enum, f"Failed for {raw_type}"
    
    def test_parse_missing_optional_fields(self):
        """Test parsing with missing optional fields uses defaults."""
        response = json.dumps({
            "query_type": "factual",
            "sub_queries": ["Test query"]
        })
        
        plan = _parse_query_plan(response)
        
        assert plan.query_type == QueryType.FACTUAL
        assert plan.sub_queries == ["Test query"]
        assert plan.parallel is True  # default
        assert plan.needs_fact_check is True  # default
        assert plan.search_strategy == SearchStrategy.HYBRID  # default
        assert plan.estimated_complexity == ComplexityLevel.MEDIUM  # default
        assert plan.reasoning == ""  # default


class TestCreateQueryPlan:
    """Tests for create_query_plan with mocked LLM."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mocked ChatAnthropic instance."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        return llm
    
    @pytest.mark.asyncio
    async def test_simple_factual_query(self, mock_llm):
        """Test planning a simple factual query like 'What is gRPC?'"""
        mock_llm.ainvoke.return_value.content = json.dumps({
            "query_type": "definition",
            "sub_queries": ["gRPC definition, purpose, and key features"],
            "parallel": False,
            "needs_fact_check": False,
            "search_strategy": "hybrid",
            "estimated_complexity": "low",
            "reasoning": "Simple definition query with a single sub-query"
        })
        
        plan = await create_query_plan("What is gRPC?", llm=mock_llm)
        
        assert plan.query_type == QueryType.DEFINITION
        assert len(plan.sub_queries) == 1
        assert plan.needs_fact_check is False
        assert plan.estimated_complexity == ComplexityLevel.LOW
    
    @pytest.mark.asyncio
    async def test_comparison_query(self, mock_llm):
        """Test planning a comparison query with multiple frameworks."""
        mock_llm.ainvoke.return_value.content = json.dumps({
            "query_type": "comparison",
            "sub_queries": [
                "React strengths and ecosystem for SPAs",
                "Vue strengths and ecosystem for SPAs",
                "Svelte strengths and ecosystem for SPAs",
                "React vs Vue vs Svelte performance comparison",
                "React vs Vue vs Svelte developer experience"
            ],
            "parallel": True,
            "needs_fact_check": True,
            "search_strategy": "hybrid",
            "estimated_complexity": "high",
            "reasoning": "Three-way comparison requires researching each framework"
        })
        
        plan = await create_query_plan(
            "Compare React, Vue, and Svelte for building SPAs",
            llm=mock_llm
        )
        
        assert plan.query_type == QueryType.COMPARISON
        assert len(plan.sub_queries) == 5
        assert plan.parallel is True
        assert plan.needs_fact_check is True
        assert plan.estimated_complexity == ComplexityLevel.HIGH
    
    @pytest.mark.asyncio
    async def test_exploratory_query(self, mock_llm):
        """Test planning an exploratory query about trends."""
        mock_llm.ainvoke.return_value.content = json.dumps({
            "query_type": "exploratory",
            "sub_queries": [
                "Latest AI agent frameworks and architectures",
                "Multi-agent systems and orchestration patterns",
                "LLM integration patterns in agent frameworks",
                "Production deployment of AI agents trends"
            ],
            "parallel": True,
            "needs_fact_check": True,
            "search_strategy": "web_only",
            "estimated_complexity": "high",
            "reasoning": "Open-ended research on current trends"
        })
        
        plan = await create_query_plan(
            "What are the latest trends in AI agent frameworks?",
            llm=mock_llm
        )
        
        assert plan.query_type == QueryType.EXPLORATORY
        assert len(plan.sub_queries) >= 3
        assert plan.search_strategy == SearchStrategy.WEB_ONLY
    
    @pytest.mark.asyncio
    async def test_opinion_query(self, mock_llm):
        """Test planning an opinion/analysis query."""
        mock_llm.ainvoke.return_value.content = json.dumps({
            "query_type": "opinion",
            "sub_queries": [
                "Rust job market and industry adoption 2026",
                "Rust learning curve and developer experience",
                "Rust use cases and domains where it excels",
                "Rust compared to alternatives for systems programming"
            ],
            "parallel": True,
            "needs_fact_check": True,
            "search_strategy": "web_only",
            "estimated_complexity": "medium",
            "reasoning": "Opinion query requires gathering multiple perspectives"
        })
        
        plan = await create_query_plan(
            "Is Rust worth learning in 2026?",
            llm=mock_llm
        )
        
        assert plan.query_type == QueryType.OPINION
        assert plan.needs_fact_check is True
        assert plan.parallel is True
    
    @pytest.mark.asyncio
    async def test_multi_part_query(self, mock_llm):
        """Test planning a multi-part query with related questions."""
        mock_llm.ainvoke.return_value.content = json.dumps({
            "query_type": "factual",
            "sub_queries": [
                "CAP theorem definition and explanation",
                "CAP theorem consistency - databases that prioritize consistency",
                "CAP theorem availability - databases that prioritize availability",
                "CAP theorem partition tolerance examples",
                "CAP theorem trade-offs in real-world databases"
            ],
            "parallel": False,
            "needs_fact_check": True,
            "search_strategy": "hybrid",
            "estimated_complexity": "medium",
            "reasoning": "Multi-part query: first explain CAP, then categorize databases"
        })
        
        plan = await create_query_plan(
            "Explain the CAP theorem and give examples of databases in each category",
            llm=mock_llm
        )
        
        assert plan.query_type == QueryType.FACTUAL
        assert len(plan.sub_queries) >= 4
        assert "CAP" in plan.sub_queries[0].upper()
    
    @pytest.mark.asyncio
    async def test_howto_query(self, mock_llm):
        """Test planning a how-to query."""
        mock_llm.ainvoke.return_value.content = json.dumps({
            "query_type": "howto",
            "sub_queries": [
                "Kubernetes architecture overview",
                "Kubernetes local development setup options",
                "Step by step Kubernetes installation guide",
                "Kubernetes basic commands and verification"
            ],
            "parallel": False,
            "needs_fact_check": False,
            "search_strategy": "hybrid",
            "estimated_complexity": "medium",
            "reasoning": "Step-by-step guide with sequential steps"
        })
        
        plan = await create_query_plan(
            "How do I set up Kubernetes locally?",
            llm=mock_llm
        )
        
        assert plan.query_type == QueryType.HOWTO
        assert plan.parallel is False  # Sequential steps
        assert plan.needs_fact_check is False
    
    @pytest.mark.asyncio
    async def test_llm_response_as_list(self, mock_llm):
        """Test handling LLM response that returns content as a list."""
        # Some LLM responses may return content as a list of blocks
        mock_llm.ainvoke.return_value.content = [
            {"text": json.dumps({
                "query_type": "factual",
                "sub_queries": ["Test query"],
                "parallel": True,
                "needs_fact_check": False,
                "search_strategy": "web_only",
                "estimated_complexity": "low",
                "reasoning": "Simple query"
            })}
        ]
        
        plan = await create_query_plan("Test query", llm=mock_llm)
        
        assert plan.query_type == QueryType.FACTUAL


class TestPlannerNode:
    """Tests for the planner_node LangGraph node function."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mocked ChatAnthropic instance."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        return llm
    
    @pytest.mark.asyncio
    async def test_planner_node_success(self, mock_llm):
        """Test planner_node returns correct state updates on success."""
        mock_llm.ainvoke.return_value.content = json.dumps({
            "query_type": "factual",
            "sub_queries": ["Sub query 1", "Sub query 2"],
            "parallel": True,
            "needs_fact_check": True,
            "search_strategy": "hybrid",
            "estimated_complexity": "medium",
            "reasoning": "Test reasoning"
        })
        
        state = {"query": "Test query"}
        
        with patch("agents.planner._create_llm", return_value=mock_llm):
            result = await planner_node(state)
        
        assert result["status"] == "researching"
        assert result["plan"] is not None
        assert result["plan"].query_type == QueryType.FACTUAL
        assert result["sub_queries"] == ["Sub query 1", "Sub query 2"]
    
    @pytest.mark.asyncio
    async def test_planner_node_empty_query(self):
        """Test planner_node handles empty query gracefully."""
        state = {"query": ""}
        
        result = await planner_node(state)
        
        assert result["status"] == "failed"
        assert result["plan"] is None
        assert "Empty query" in result["errors"][0]
    
    @pytest.mark.asyncio
    async def test_planner_node_missing_query(self):
        """Test planner_node handles missing query key."""
        state = {}
        
        result = await planner_node(state)
        
        assert result["status"] == "failed"
        assert "Empty query" in result["errors"][0]


class TestSystemPrompt:
    """Tests for the planner system prompt."""
    
    def test_system_prompt_contains_query_types(self):
        """Test that system prompt documents all query types."""
        assert "factual" in PLANNER_SYSTEM_PROMPT.lower()
        assert "comparison" in PLANNER_SYSTEM_PROMPT.lower()
        assert "exploratory" in PLANNER_SYSTEM_PROMPT.lower()
        assert "opinion" in PLANNER_SYSTEM_PROMPT.lower()
        assert "howto" in PLANNER_SYSTEM_PROMPT.lower()
        assert "definition" in PLANNER_SYSTEM_PROMPT.lower()
    
    def test_system_prompt_contains_search_strategies(self):
        """Test that system prompt documents search strategies."""
        assert "web_only" in PLANNER_SYSTEM_PROMPT
        assert "docs_only" in PLANNER_SYSTEM_PROMPT
        assert "hybrid" in PLANNER_SYSTEM_PROMPT
    
    def test_system_prompt_contains_complexity_levels(self):
        """Test that system prompt documents complexity levels."""
        assert "low" in PLANNER_SYSTEM_PROMPT.lower()
        assert "medium" in PLANNER_SYSTEM_PROMPT.lower()
        assert "high" in PLANNER_SYSTEM_PROMPT.lower()
    
    def test_system_prompt_contains_examples(self):
        """Test that system prompt includes decomposition examples."""
        assert "Compare Rust vs Go" in PLANNER_SYSTEM_PROMPT
        assert "CAP theorem" in PLANNER_SYSTEM_PROMPT
