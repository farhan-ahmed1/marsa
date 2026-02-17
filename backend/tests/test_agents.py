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


# ==============================================================================
# Researcher Agent Tests
# ==============================================================================


from agents.researcher import (  # noqa: E402
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    _deduplicate_results,
    _parse_claims_response,
    _score_and_sort_results,
    extract_claims,
    research_node,
)
from graph.state import (  # noqa: E402
    AgentName,
    Claim,
    ConfidenceLevel,
    PipelineStatus,
    QueryPlan,
    ResearchResult,
)


class TestParseClaimsResponse:
    """Tests for _parse_claims_response parsing logic."""
    
    def test_parse_valid_claims_json(self):
        """Test parsing a valid claims JSON response."""
        response = json.dumps({
            "claims": [
                {
                    "statement": "Python was created by Guido van Rossum",
                    "source_url": "https://docs.python.org/3/faq/general.html",
                    "source_title": "Python General FAQ",
                    "confidence": "high",
                    "category": "fact",
                    "context": "Python was created in the late 1980s."
                },
                {
                    "statement": "Python 3.0 was released on December 3, 2008",
                    "source_url": "https://docs.python.org/3/whatsnew/3.0.html",
                    "source_title": "What's New In Python 3.0",
                    "confidence": "high",
                    "category": "fact",
                    "context": "Python 3.0 was released on December 3, 2008."
                }
            ]
        })
        
        claims = _parse_claims_response(response, "test query")
        
        assert len(claims) == 2
        assert claims[0].statement == "Python was created by Guido van Rossum"
        assert claims[0].source_url == "https://docs.python.org/3/faq/general.html"
        assert claims[0].confidence == ConfidenceLevel.HIGH
        assert claims[0].category == "fact"
    
    def test_parse_claims_with_markdown_fences(self):
        """Test parsing claims wrapped in markdown code fences."""
        response = """```json
{
    "claims": [
        {
            "statement": "Test claim",
            "source_url": "https://example.com",
            "source_title": "Example",
            "confidence": "medium",
            "category": "fact",
            "context": "Test context"
        }
    ]
}
```"""
        
        claims = _parse_claims_response(response, "test query")
        
        assert len(claims) == 1
        assert claims[0].statement == "Test claim"
    
    def test_parse_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        response = "This is not valid JSON"
        
        with pytest.raises(ValueError) as exc_info:
            _parse_claims_response(response, "test query")
        
        assert "Invalid JSON" in str(exc_info.value)
    
    def test_parse_missing_claims_array(self):
        """Test that missing 'claims' key raises ValueError."""
        response = json.dumps({"something_else": []})
        
        with pytest.raises(ValueError) as exc_info:
            _parse_claims_response(response, "test query")
        
        assert "missing 'claims' array" in str(exc_info.value).lower()
    
    def test_parse_confidence_levels(self):
        """Test parsing all confidence levels."""
        response = json.dumps({
            "claims": [
                {"statement": "High conf", "source_url": "https://a.com", "source_title": "A", "confidence": "high", "category": "fact", "context": ""},
                {"statement": "Medium conf", "source_url": "https://b.com", "source_title": "B", "confidence": "medium", "category": "fact", "context": ""},
                {"statement": "Low conf", "source_url": "https://c.com", "source_title": "C", "confidence": "low", "category": "fact", "context": ""},
            ]
        })
        
        claims = _parse_claims_response(response, "test query")
        
        assert claims[0].confidence == ConfidenceLevel.HIGH
        assert claims[1].confidence == ConfidenceLevel.MEDIUM
        assert claims[2].confidence == ConfidenceLevel.LOW
    
    def test_parse_claim_categories(self):
        """Test parsing different claim categories."""
        response = json.dumps({
            "claims": [
                {"statement": "Fact", "source_url": "https://a.com", "source_title": "A", "confidence": "high", "category": "fact", "context": ""},
                {"statement": "Opinion", "source_url": "https://b.com", "source_title": "B", "confidence": "medium", "category": "opinion", "context": ""},
                {"statement": "Statistic", "source_url": "https://c.com", "source_title": "C", "confidence": "high", "category": "statistic", "context": ""},
                {"statement": "Quote", "source_url": "https://d.com", "source_title": "D", "confidence": "medium", "category": "quote", "context": ""},
            ]
        })
        
        claims = _parse_claims_response(response, "test query")
        
        assert claims[0].category == "fact"
        assert claims[1].category == "opinion"
        assert claims[2].category == "statistic"
        assert claims[3].category == "quote"
    
    def test_parse_skips_invalid_claims(self):
        """Test that invalid claims are skipped with warning."""
        response = json.dumps({
            "claims": [
                {"statement": "Valid", "source_url": "https://a.com", "source_title": "A", "confidence": "high", "category": "fact", "context": ""},
                "invalid_claim_format",
                {"statement": "Also valid", "source_url": "https://b.com", "source_title": "B", "confidence": "medium", "category": "fact", "context": ""},
            ]
        })
        
        claims = _parse_claims_response(response, "test query")
        
        # Should skip the invalid one and return 2 valid claims
        assert len(claims) == 2
        assert claims[0].statement == "Valid"
        assert claims[1].statement == "Also valid"


class TestDeduplicateResults:
    """Tests for _deduplicate_results function."""
    
    def test_deduplicate_by_url(self):
        """Test deduplication based on URL."""
        results = [
            ResearchResult(
                content="Content 1",
                source_url="https://example.com",
                source_title="Example",
                relevance_score=0.9,
            ),
            ResearchResult(
                content="Content 2",
                source_url="https://example.com",  # Duplicate URL
                source_title="Example Again",
                relevance_score=0.8,
            ),
            ResearchResult(
                content="Content 3",
                source_url="https://other.com",
                source_title="Other",
                relevance_score=0.7,
            ),
        ]
        
        deduplicated = _deduplicate_results(results)
        
        assert len(deduplicated) == 2
        assert deduplicated[0].source_url == "https://example.com"
        assert deduplicated[1].source_url == "https://other.com"
    
    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list."""
        results = []
        
        deduplicated = _deduplicate_results(results)
        
        assert len(deduplicated) == 0
    
    def test_deduplicate_no_duplicates(self):
        """Test deduplication when no duplicates exist."""
        results = [
            ResearchResult(content="C1", source_url="https://a.com", source_title="A", relevance_score=0.9),
            ResearchResult(content="C2", source_url="https://b.com", source_title="B", relevance_score=0.8),
            ResearchResult(content="C3", source_url="https://c.com", source_title="C", relevance_score=0.7),
        ]
        
        deduplicated = _deduplicate_results(results)
        
        assert len(deduplicated) == 3


class TestScoreAndSortResults:
    """Tests for _score_and_sort_results function."""
    
    def test_results_sorted_by_quality(self):
        """Test that results are sorted by source quality score."""
        results = [
            ResearchResult(content="Blog post", source_url="https://blog.example.com", source_title="Blog", relevance_score=0.9),
            ResearchResult(content="Gov site", source_url="https://example.gov", source_title="Gov", relevance_score=0.7),
            ResearchResult(content="Edu site", source_url="https://example.edu", source_title="Edu", relevance_score=0.8),
        ]
        
        sorted_results, source_scores = _score_and_sort_results(results)
        
        # .gov and .edu should be scored higher than blog
        assert sorted_results[0].source_url in ["https://example.gov", "https://example.edu"]
        assert sorted_results[2].source_url == "https://blog.example.com"
        assert len(source_scores) == 3
    
    def test_source_scores_dict_created(self):
        """Test that source scores dictionary is properly created."""
        results = [
            ResearchResult(content="C1", source_url="https://a.com", source_title="A", relevance_score=0.9),
            ResearchResult(content="C2", source_url="https://b.com", source_title="B", relevance_score=0.8),
        ]
        
        sorted_results, source_scores = _score_and_sort_results(results)
        
        assert "https://a.com" in source_scores
        assert "https://b.com" in source_scores
        assert 0.0 <= source_scores["https://a.com"] <= 1.0
        assert 0.0 <= source_scores["https://b.com"] <= 1.0


class TestExtractClaims:
    """Tests for extract_claims function."""
    
    @pytest.mark.asyncio
    async def test_extract_claims_no_results(self):
        """Test extract_claims with empty results list."""
        results = []
        
        claims = await extract_claims(results, "test query")
        
        assert claims == []
    
    @pytest.mark.asyncio
    async def test_extract_claims_with_mocked_llm(self):
        """Test extract_claims with mocked LLM response."""
        results = [
            ResearchResult(
                content="Python was created by Guido van Rossum in the late 1980s.",
                source_url="https://docs.python.org/3/faq/general.html",
                source_title="Python FAQ",
                relevance_score=0.95,
            )
        ]
        
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "claims": [
                {
                    "statement": "Python was created by Guido van Rossum",
                    "source_url": "https://docs.python.org/3/faq/general.html",
                    "source_title": "Python FAQ",
                    "confidence": "high",
                    "category": "fact",
                    "context": "Python was created in the late 1980s."
                }
            ]
        })
        mock_response.usage_metadata = {"total_tokens": 500}
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        claims = await extract_claims(results, "Who created Python?", llm=mock_llm)
        
        assert len(claims) == 1
        assert claims[0].statement == "Python was created by Guido van Rossum"
        assert claims[0].confidence == ConfidenceLevel.HIGH
        assert mock_llm.ainvoke.called
    
    @pytest.mark.asyncio
    async def test_extract_claims_malformed_json_retries(self):
        """Test that extract_claims retries on malformed JSON."""
        results = [
            ResearchResult(content="Test", source_url="https://test.com", source_title="Test", relevance_score=0.8)
        ]
        
        mock_llm = MagicMock()
        
        # First call returns invalid JSON, second call returns valid
        mock_response_1 = MagicMock()
        mock_response_1.content = "Invalid JSON here"
        mock_response_1.usage_metadata = {}
        
        mock_response_2 = MagicMock()
        mock_response_2.content = json.dumps({
            "claims": [
                {"statement": "Test claim", "source_url": "https://test.com", "source_title": "Test", "confidence": "medium", "category": "fact", "context": ""}
            ]
        })
        mock_response_2.usage_metadata = {"total_tokens": 300}
        
        mock_llm.ainvoke = AsyncMock(side_effect=[mock_response_1, mock_response_2])
        
        claims = await extract_claims(results, "test query", llm=mock_llm)
        
        assert len(claims) == 1
        assert mock_llm.ainvoke.call_count == 2  # Retried once


class TestResearchNode:
    """Tests for research_node function."""
    
    @pytest.mark.asyncio
    async def test_research_node_empty_query(self):
        """Test research_node handles empty query gracefully."""
        state = {"query": ""}
        
        result = await research_node(state)
        
        assert result["status"] == PipelineStatus.FAILED.value
        assert "Empty query" in result["errors"][0]
        assert result["research_results"] == []
        assert result["claims"] == []
    
    @pytest.mark.asyncio
    async def test_research_node_no_plan(self):
        """Test research_node handles missing plan gracefully."""
        state = {"query": "test query", "plan": None}
        
        result = await research_node(state)
        
        assert result["status"] == PipelineStatus.FAILED.value
        assert "No valid plan" in result["errors"][0]
    
    @pytest.mark.asyncio
    async def test_research_node_with_mocked_client(self):
        """Test research_node with mocked MCP client."""
        plan = QueryPlan(
            query_type=QueryType.FACTUAL,
            sub_queries=["What is Python?"],
            parallel=False,
            needs_fact_check=True,
            search_strategy=SearchStrategy.HYBRID,
            estimated_complexity=ComplexityLevel.LOW,
            reasoning="Simple query",
        )
        
        state = {
            "query": "What is Python?",
            "plan": plan,
            "agent_trace": [],
        }
        
        # Mock the MCP client
        with patch("agents.researcher.MCPClient") as mock_client_class:
            mock_client = MagicMock()
            
            # Mock web search results
            mock_web_result = MagicMock()
            mock_web_result.title = "Python.org"
            mock_web_result.url = "https://python.org"
            mock_web_result.content = "Python is a programming language."
            mock_web_result.score = 0.95
            mock_web_result.published_date = None
            
            mock_client.web_search = AsyncMock(return_value=[mock_web_result])
            
            # Mock doc search results
            mock_doc_result = MagicMock()
            mock_doc_result.title = "Python Docs"
            mock_doc_result.source_url = "https://docs.python.org"
            mock_doc_result.content = "Python is a high-level programming language."
            mock_doc_result.relevance_score = 0.9
            
            mock_client.doc_search = AsyncMock(return_value=[mock_doc_result])
            
            mock_client_class.return_value = mock_client
            
            # Mock extract_claims
            with patch("agents.researcher.extract_claims") as mock_extract:
                mock_extract.return_value = [
                    Claim(
                        statement="Python is a programming language",
                        source_url="https://python.org",
                        source_title="Python.org",
                        confidence=ConfidenceLevel.HIGH,
                        category="fact",
                        context="",
                    )
                ]
                
                result = await research_node(state)
        
        # Verify results
        assert result["status"] in [PipelineStatus.FACT_CHECKING.value, PipelineStatus.SYNTHESIZING.value]
        assert len(result["research_results"]) >= 1  # At least one result after deduplication
        assert len(result["claims"]) == 1
        assert len(result["source_scores"]) >= 1
        
        # Verify trace events were added
        assert len(result["agent_trace"]) > 0
        assert any(event.agent == AgentName.RESEARCHER for event in result["agent_trace"])
    
    @pytest.mark.asyncio
    async def test_research_node_no_results_found(self):
        """Test research_node when searches return no results."""
        plan = QueryPlan(
            query_type=QueryType.FACTUAL,
            sub_queries=["Obscure query"],
            parallel=False,
            needs_fact_check=True,
            search_strategy=SearchStrategy.WEB_ONLY,
            estimated_complexity=ComplexityLevel.LOW,
            reasoning="Test",
        )
        
        state = {
            "query": "Obscure query",
            "plan": plan,
            "agent_trace": [],
        }
        
        # Mock the MCP client to return empty results
        with patch("agents.researcher.MCPClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.web_search = AsyncMock(return_value=[])
            mock_client_class.return_value = mock_client
            
            result = await research_node(state)
        
        # Should handle gracefully
        assert result["research_results"] == []
        assert result["claims"] == []
        assert "No research results found" in result["errors"][0]
    
    @pytest.mark.asyncio
    async def test_research_node_trace_events_structure(self):
        """Test that research_node generates proper trace events."""
        plan = QueryPlan(
            query_type=QueryType.FACTUAL,
            sub_queries=["test query"],
            parallel=False,
            needs_fact_check=False,  # Skip fact checking
            search_strategy=SearchStrategy.WEB_ONLY,
            estimated_complexity=ComplexityLevel.LOW,
            reasoning="Test",
        )
        
        state = {
            "query": "test",
            "plan": plan,
            "agent_trace": [],
        }
        
        with patch("agents.researcher.MCPClient") as mock_client_class:
            mock_client = MagicMock()
            mock_web_result = MagicMock()
            mock_web_result.title = "Test"
            mock_web_result.url = "https://test.com"
            mock_web_result.content = "Test content"
            mock_web_result.score = 0.8
            mock_web_result.published_date = None
            
            mock_client.web_search = AsyncMock(return_value=[mock_web_result])
            mock_client_class.return_value = mock_client
            
            with patch("agents.researcher.extract_claims") as mock_extract:
                mock_extract.return_value = []
                
                result = await research_node(state)
        
        # Check trace events
        trace_events = result["agent_trace"]
        assert len(trace_events) > 0
        
        # Should have start, web_search, source_scoring, and complete events
        actions = [event.action for event in trace_events]
        assert "start" in actions
        assert "web_search" in actions
        assert "complete" in actions


class TestClaimExtractionPrompt:
    """Tests for the claim extraction system prompt."""
    
    def test_prompt_contains_output_schema(self):
        """Test that prompt documents the output schema."""
        assert "statement" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "source_url" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "confidence" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "category" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "context" in CLAIM_EXTRACTION_SYSTEM_PROMPT
    
    def test_prompt_contains_confidence_levels(self):
        """Test that prompt documents confidence levels."""
        assert "high" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "medium" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "low" in CLAIM_EXTRACTION_SYSTEM_PROMPT
    
    def test_prompt_contains_categories(self):
        """Test that prompt documents claim categories."""
        assert "fact" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "opinion" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "statistic" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "quote" in CLAIM_EXTRACTION_SYSTEM_PROMPT
    
    def test_prompt_contains_examples(self):
        """Test that prompt includes extraction examples."""
        assert "Python" in CLAIM_EXTRACTION_SYSTEM_PROMPT
        assert "CAP theorem" in CLAIM_EXTRACTION_SYSTEM_PROMPT
