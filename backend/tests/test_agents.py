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


# ---------------------------------------------------------------------------
# Fact-Checker Agent Tests
# ---------------------------------------------------------------------------

from agents.fact_checker import (  # noqa: E402
    _adjust_confidence_by_source_quality,
    _parse_verification_response,
    fact_check_node,
    generate_verify_query,
    should_loop_back,
    verify_claim,
    BAD_CLAIM_THRESHOLD,
    MAX_ITERATIONS,
)
from graph.state import (  # noqa: E402
    VerificationResult,
    VerificationVerdict,
)


class TestParseVerificationResponse:
    """Tests for _parse_verification_response parsing logic."""
    
    def test_parse_valid_supported_verdict(self):
        """Test parsing a valid supported verdict."""
        response = json.dumps({
            "verdict": "supported",
            "confidence": 0.85,
            "supporting_sources": ["https://example.com/1", "https://example.com/2"],
            "contradicting_sources": [],
            "reasoning": "Multiple sources confirm this claim"
        })
        
        claim = Claim(
            statement="Python was created by Guido van Rossum",
            source_url="https://python.org",
            source_title="Python.org",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="Historical fact about Python",
        )
        
        result = _parse_verification_response(response, claim, "test query")
        
        assert result.verdict == VerificationVerdict.SUPPORTED
        assert result.confidence == 0.85
        assert len(result.supporting_sources) == 2
        assert len(result.contradicting_sources) == 0
        assert "Multiple sources" in result.reasoning
    
    def test_parse_contradicted_verdict(self):
        """Test parsing a contradicted verdict."""
        response = json.dumps({
            "verdict": "contradicted",
            "confidence": 0.9,
            "supporting_sources": [],
            "contradicting_sources": ["https://go.dev/doc/"],
            "reasoning": "Go was released in 2009, not 2015"
        })
        
        claim = Claim(
            statement="Go was released in 2015",
            source_url="https://example.com",
            source_title="Some Blog",
            confidence=ConfidenceLevel.LOW,
            category="fact",
            context="",
        )
        
        result = _parse_verification_response(response, claim, "go release date")
        
        assert result.verdict == VerificationVerdict.CONTRADICTED
        assert result.confidence == 0.9
        assert len(result.contradicting_sources) == 1
        assert "2009" in result.reasoning
    
    def test_parse_unverifiable_verdict(self):
        """Test parsing an unverifiable verdict."""
        response = json.dumps({
            "verdict": "unverifiable",
            "confidence": 0.3,
            "supporting_sources": [],
            "contradicting_sources": [],
            "reasoning": "No relevant sources found"
        })
        
        claim = Claim(
            statement="Some obscure claim",
            source_url="https://example.com",
            source_title="Unknown",
            confidence=ConfidenceLevel.LOW,
            category="opinion",
            context="",
        )
        
        result = _parse_verification_response(response, claim, "obscure query")
        
        assert result.verdict == VerificationVerdict.UNVERIFIABLE
        assert result.confidence == 0.3
    
    def test_parse_json_with_markdown_fences(self):
        """Test parsing JSON wrapped in markdown code fences."""
        response = """```json
{
    "verdict": "supported",
    "confidence": 0.7,
    "supporting_sources": ["https://source.com"],
    "contradicting_sources": [],
    "reasoning": "Found supporting evidence"
}
```"""
        
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Test",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
        
        result = _parse_verification_response(response, claim, "test query")
        
        assert result.verdict == VerificationVerdict.SUPPORTED
        assert result.confidence == 0.7
    
    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        response = "Not valid JSON at all"
        
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Test",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
        
        with pytest.raises(ValueError) as exc_info:
            _parse_verification_response(response, claim, "test query")
        
        assert "Invalid JSON" in str(exc_info.value)
    
    def test_parse_unknown_verdict_defaults_to_unverifiable(self):
        """Test that unknown verdicts default to unverifiable."""
        response = json.dumps({
            "verdict": "unknown_verdict",
            "confidence": 0.5,
            "supporting_sources": [],
            "contradicting_sources": [],
            "reasoning": "Unknown verdict type"
        })
        
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Test",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
        
        result = _parse_verification_response(response, claim, "test query")
        
        assert result.verdict == VerificationVerdict.UNVERIFIABLE
    
    def test_parse_confidence_clamped_to_valid_range(self):
        """Test that confidence values are clamped to 0.0-1.0."""
        response = json.dumps({
            "verdict": "supported",
            "confidence": 1.5,  # Out of range
            "supporting_sources": [],
            "contradicting_sources": [],
            "reasoning": "Test"
        })
        
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Test",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
        
        result = _parse_verification_response(response, claim, "test query")
        
        assert result.confidence == 1.0  # Clamped to max
    
    def test_parse_negative_confidence_clamped(self):
        """Test that negative confidence is clamped to 0.0."""
        response = json.dumps({
            "verdict": "contradicted",
            "confidence": -0.5,  # Negative
            "supporting_sources": [],
            "contradicting_sources": [],
            "reasoning": "Test"
        })
        
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Test",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
        
        result = _parse_verification_response(response, claim, "test query")
        
        assert result.confidence == 0.0  # Clamped to min


class TestAdjustConfidenceBySourceQuality:
    """Tests for source quality confidence adjustments."""
    
    def test_high_quality_sources_boost_confidence(self):
        """Test that high-quality sources boost verdict confidence."""
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Test",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
        
        result = VerificationResult(
            claim=claim,
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.7,
            supporting_sources=["https://gov.source1.gov", "https://edu.source2.edu"],
            contradicting_sources=[],
            reasoning="High quality sources",
            verification_query="test",
        )
        
        # High quality scores (0.8+)
        source_scores = {
            "https://gov.source1.gov": 0.9,
            "https://edu.source2.edu": 0.85,
        }
        
        adjusted = _adjust_confidence_by_source_quality(result, source_scores)
        
        # Confidence should be boosted
        assert adjusted.confidence > result.confidence
    
    def test_low_quality_sources_reduce_confidence(self):
        """Test that low-quality sources reduce verdict confidence."""
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Test",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
        
        result = VerificationResult(
            claim=claim,
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.7,
            supporting_sources=["https://blog1.com", "https://blog2.com"],
            contradicting_sources=[],
            reasoning="Low quality sources",
            verification_query="test",
        )
        
        # Low quality scores (0.4 or below)
        source_scores = {
            "https://blog1.com": 0.35,
            "https://blog2.com": 0.40,
        }
        
        adjusted = _adjust_confidence_by_source_quality(result, source_scores)
        
        # Confidence should be reduced
        assert adjusted.confidence < result.confidence
    
    def test_no_relevant_sources_unchanged(self):
        """Test that result with no relevant sources keeps same confidence."""
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Test",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
        
        result = VerificationResult(
            claim=claim,
            verdict=VerificationVerdict.UNVERIFIABLE,
            confidence=0.5,
            supporting_sources=[],
            contradicting_sources=[],
            reasoning="No sources found",
            verification_query="test",
        )
        
        adjusted = _adjust_confidence_by_source_quality(result, {})
        
        assert adjusted.confidence == result.confidence
    
    def test_confidence_stays_within_bounds(self):
        """Test that adjusted confidence stays within 0.0-1.0."""
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Test",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="",
        )
        
        result = VerificationResult(
            claim=claim,
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.98,  # Already high
            supporting_sources=["https://gov.source.gov"],
            contradicting_sources=[],
            reasoning="Already confident",
            verification_query="test",
        )
        
        source_scores = {"https://gov.source.gov": 0.95}
        
        adjusted = _adjust_confidence_by_source_quality(result, source_scores)
        
        assert adjusted.confidence <= 1.0


class TestShouldLoopBack:
    """Tests for should_loop_back conditional routing."""
    
    def _create_claim(self) -> Claim:
        """Helper to create a test claim."""
        return Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Test",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
    
    def _create_result(self, verdict: VerificationVerdict) -> VerificationResult:
        """Helper to create a test verification result."""
        return VerificationResult(
            claim=self._create_claim(),
            verdict=verdict,
            confidence=0.5,
            supporting_sources=[],
            contradicting_sources=[],
            reasoning="Test",
            verification_query="test",
        )
    
    def test_all_supported_proceeds_to_synthesizer(self):
        """Test that all supported claims proceed to synthesizer."""
        results = [
            self._create_result(VerificationVerdict.SUPPORTED)
            for _ in range(5)
        ]
        
        state = {
            "verification_results": results,
            "iteration_count": 0,
        }
        
        assert should_loop_back(state) == "synthesizer"
    
    def test_over_threshold_contradicted_loops_back(self):
        """Test that >40% contradicted claims triggers loop back."""
        # 5 out of 10 = 50% contradicted claims (only contradicted counts as bad now)
        results = [
            self._create_result(VerificationVerdict.SUPPORTED)
            for _ in range(5)
        ] + [
            self._create_result(VerificationVerdict.CONTRADICTED)
            for _ in range(5)
        ]
        
        state = {
            "verification_results": results,
            "iteration_count": 0,
        }
        
        assert should_loop_back(state) == "researcher"
    
    def test_unverifiable_does_not_count_as_bad(self):
        """Test that unverifiable claims do NOT trigger loop back."""
        # 6 supported + 4 unverifiable = 0% contradicted (below threshold)
        results = [
            self._create_result(VerificationVerdict.SUPPORTED)
            for _ in range(6)
        ] + [
            self._create_result(VerificationVerdict.UNVERIFIABLE)
            for _ in range(4)
        ]
        
        state = {
            "verification_results": results,
            "iteration_count": 0,
        }
        
        assert should_loop_back(state) == "synthesizer"
    
    def test_exactly_40_percent_proceeds_to_synthesizer(self):
        """Test that exactly 40% contradicted claims proceeds to synthesizer."""
        # 4 out of 10 = exactly 40% contradicted (not over threshold)
        results = [
            self._create_result(VerificationVerdict.SUPPORTED)
            for _ in range(6)
        ] + [
            self._create_result(VerificationVerdict.CONTRADICTED)
            for _ in range(4)
        ]
        
        state = {
            "verification_results": results,
            "iteration_count": 0,
        }
        
        assert should_loop_back(state) == "synthesizer"
    
    def test_max_iterations_prevents_loop_back(self):
        """Test that max iterations prevents infinite loops."""
        # All claims are bad, but we've hit max iterations
        results = [
            self._create_result(VerificationVerdict.CONTRADICTED)
            for _ in range(5)
        ]
        
        state = {
            "verification_results": results,
            "iteration_count": MAX_ITERATIONS,  # At max
        }
        
        assert should_loop_back(state) == "synthesizer"
    
    def test_empty_results_proceeds_to_synthesizer(self):
        """Test that empty results proceed to synthesizer."""
        state = {
            "verification_results": [],
            "iteration_count": 0,
        }
        
        assert should_loop_back(state) == "synthesizer"
    
    def test_iteration_count_below_max_allows_loop_back(self):
        """Test that iteration count below max allows loop back."""
        # All contradicted to exceed threshold
        results = [
            self._create_result(VerificationVerdict.CONTRADICTED)
            for _ in range(10)
        ]
        
        state = {
            "verification_results": results,
            "iteration_count": MAX_ITERATIONS - 1,
        }
        
        assert should_loop_back(state) == "researcher"


class TestGenerateVerifyQuery:
    """Tests for generate_verify_query with mocked LLM."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mocked ChatAnthropic instance."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        return llm
    
    @pytest.mark.asyncio
    async def test_generates_different_query(self, mock_llm):
        """Test that verification query is generated."""
        mock_llm.ainvoke.return_value.content = "Python programming language creator history origin"
        
        claim = Claim(
            statement="Python was created by Guido van Rossum",
            source_url="https://python.org",
            source_title="Python.org",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="Historical fact",
        )
        
        query = await generate_verify_query(claim, llm=mock_llm)
        
        assert len(query) > 0
        assert query == "Python programming language creator history origin"
        mock_llm.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handles_list_response(self, mock_llm):
        """Test handling LLM response as a list."""
        mock_llm.ainvoke.return_value.content = [
            {"text": "Go programming language release date 2009"}
        ]
        
        claim = Claim(
            statement="Go was released in 2009",
            source_url="https://go.dev",
            source_title="Go Dev",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="",
        )
        
        query = await generate_verify_query(claim, llm=mock_llm)
        
        assert "Go" in query or "2009" in query


class TestVerifyClaim:
    """Tests for verify_claim with mocked LLM."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mocked ChatAnthropic instance."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        return llm
    
    @pytest.fixture
    def sample_claim(self):
        """Create a sample claim for testing."""
        return Claim(
            statement="Python was created by Guido van Rossum",
            source_url="https://python.org",
            source_title="Python.org",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="Historical fact about Python's creation",
        )
    
    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results."""
        result1 = MagicMock()
        result1.url = "https://docs.python.org/faq"
        result1.title = "Python FAQ"
        result1.content = "Python was created by Guido van Rossum in 1991"
        
        result2 = MagicMock()
        result2.url = "https://en.wikipedia.org/wiki/Python"
        result2.title = "Python - Wikipedia"
        result2.content = "Guido van Rossum began working on Python in the late 1980s"
        
        return [result1, result2]
    
    @pytest.mark.asyncio
    async def test_verify_supported_claim(self, mock_llm, sample_claim, mock_search_results):
        """Test verifying a claim that should be supported."""
        mock_llm.ainvoke.return_value.content = json.dumps({
            "verdict": "supported",
            "confidence": 0.95,
            "supporting_sources": [
                "https://docs.python.org/faq",
                "https://en.wikipedia.org/wiki/Python"
            ],
            "contradicting_sources": [],
            "reasoning": "Multiple authoritative sources confirm Guido created Python"
        })
        
        source_scores = {
            "https://docs.python.org/faq": 0.85,
            "https://en.wikipedia.org/wiki/Python": 0.75,
        }
        
        result = await verify_claim(
            claim=sample_claim,
            search_results=mock_search_results,
            source_scores=source_scores,
            llm=mock_llm,
            verification_query="Python creator origin",
        )
        
        assert result.verdict == VerificationVerdict.SUPPORTED
        assert result.confidence >= 0.9
        assert len(result.supporting_sources) == 2
    
    @pytest.mark.asyncio
    async def test_verify_contradicted_claim(self, mock_llm, mock_search_results):
        """Test verifying a claim that should be contradicted."""
        claim = Claim(
            statement="Go was released in 2015",
            source_url="https://example.com",
            source_title="Some Blog",
            confidence=ConfidenceLevel.LOW,
            category="fact",
            context="",
        )
        
        mock_llm.ainvoke.return_value.content = json.dumps({
            "verdict": "contradicted",
            "confidence": 0.9,
            "supporting_sources": [],
            "contradicting_sources": ["https://go.dev/doc/"],
            "reasoning": "Go was released in November 2009, not 2015"
        })
        
        source_scores = {"https://go.dev/doc/": 0.85}
        
        result = await verify_claim(
            claim=claim,
            search_results=mock_search_results,
            source_scores=source_scores,
            llm=mock_llm,
            verification_query="Go release date",
        )
        
        assert result.verdict == VerificationVerdict.CONTRADICTED
        # Confidence may be adjusted up due to high-quality source (0.85)
        assert result.confidence >= 0.9
    
    @pytest.mark.asyncio
    async def test_verify_with_no_search_results(self, mock_llm, sample_claim):
        """Test that empty search results return unverifiable verdict."""
        result = await verify_claim(
            claim=sample_claim,
            search_results=[],
            source_scores={},
            llm=mock_llm,
            verification_query="test query",
        )
        
        assert result.verdict == VerificationVerdict.UNVERIFIABLE
        assert result.confidence == 0.3
        assert "No search results" in result.reasoning
        # LLM should not be called when no results
        mock_llm.ainvoke.assert_not_called()


class TestFactCheckNode:
    """Tests for fact_check_node LangGraph node function."""
    
    @pytest.fixture
    def sample_claims(self):
        """Create sample claims for testing."""
        return [
            Claim(
                statement="Python was created by Guido van Rossum",
                source_url="https://python.org",
                source_title="Python.org",
                confidence=ConfidenceLevel.HIGH,
                category="fact",
                context="",
            ),
            Claim(
                statement="Python first released in 1991",
                source_url="https://python.org",
                source_title="Python.org",
                confidence=ConfidenceLevel.HIGH,
                category="fact",
                context="",
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_fact_check_node_no_claims(self):
        """Test fact_check_node with no claims returns early."""
        state = {
            "claims": [],
            "source_scores": {},
            "agent_trace": [],
            "iteration_count": 0,
        }
        
        result = await fact_check_node(state)
        
        assert result["verification_results"] == []
        assert result["iteration_count"] == 1
        
        # Should have trace events
        assert len(result["agent_trace"]) >= 1
        actions = [e.action for e in result["agent_trace"]]
        assert "start" in actions
        assert "skip" in actions
    
    @pytest.mark.asyncio
    async def test_fact_check_node_success(self, sample_claims):
        """Test fact_check_node with successful verification."""
        state = {
            "claims": sample_claims,
            "source_scores": {},
            "agent_trace": [],
            "iteration_count": 0,
        }
        
        with patch("agents.fact_checker.MCPClient") as mock_client_class:
            mock_client = MagicMock()
            mock_search_result = MagicMock()
            mock_search_result.url = "https://python.org/faq"
            mock_search_result.title = "Python FAQ"
            mock_search_result.content = "Python was created by Guido van Rossum"
            mock_search_result.published_date = None
            mock_client.web_search = AsyncMock(return_value=[mock_search_result])
            mock_client_class.return_value = mock_client
            
            with patch("agents.fact_checker.generate_verify_query") as mock_gen_query:
                mock_gen_query.return_value = "Python creator history"
                
                with patch("agents.fact_checker.verify_claim") as mock_verify:
                    mock_verify.return_value = VerificationResult(
                        claim=sample_claims[0],
                        verdict=VerificationVerdict.SUPPORTED,
                        confidence=0.9,
                        supporting_sources=["https://python.org/faq"],
                        contradicting_sources=[],
                        reasoning="Confirmed by official sources",
                        verification_query="Python creator history",
                    )
                    
                    result = await fact_check_node(state)
        
        assert len(result["verification_results"]) == 2
        assert result["iteration_count"] == 1
        
        # Check trace events
        trace_events = result["agent_trace"]
        actions = [e.action for e in trace_events]
        assert "start" in actions
        assert "complete" in actions
    
    @pytest.mark.asyncio
    async def test_fact_check_node_handles_errors(self, sample_claims):
        """Test fact_check_node handles claim verification errors gracefully."""
        state = {
            "claims": sample_claims,
            "source_scores": {},
            "agent_trace": [],
            "iteration_count": 0,
        }
        
        with patch("agents.fact_checker.MCPClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.web_search = AsyncMock(side_effect=Exception("API error"))
            mock_client_class.return_value = mock_client
            
            with patch("agents.fact_checker.generate_verify_query") as mock_gen_query:
                mock_gen_query.return_value = "test query"
                
                result = await fact_check_node(state)
        
        # Should still return results (unverifiable for failed claims)
        assert len(result["verification_results"]) == 2
        for vr in result["verification_results"]:
            assert vr.verdict == VerificationVerdict.UNVERIFIABLE
            assert "error" in vr.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_fact_check_node_mcp_client_error(self, sample_claims):
        """Test fact_check_node handles MCP client init errors."""
        state = {
            "claims": sample_claims,
            "source_scores": {},
            "agent_trace": [],
            "iteration_count": 0,
            "errors": [],
        }
        
        with patch("agents.fact_checker.MCPClient") as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")
            
            result = await fact_check_node(state)
        
        assert result["verification_results"] == []
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert "MCP client" in result["errors"][0]


class TestFactCheckerConstants:
    """Tests for fact-checker constants and thresholds."""
    
    def test_bad_claim_threshold_is_40_percent(self):
        """Test that bad claim threshold is 0.4 (40%)."""
        assert BAD_CLAIM_THRESHOLD == 0.4
    
    def test_max_iterations_is_2(self):
        """Test that max iterations is 2."""
        assert MAX_ITERATIONS == 2


# ---------------------------------------------------------------------------
# Synthesizer Agent Tests
# ---------------------------------------------------------------------------


from agents.synthesizer import (  # noqa: E402
    generate_report,
    synthesize_node,
    _build_citation_map,
    _format_claims_for_prompt,
    _format_report_as_text,
    _parse_report_response,
)
from graph.state import (  # noqa: E402
    Citation,
    Report,
    ReportMetadata,
    ReportSection,
)


class TestBuildCitationMap:
    """Tests for _build_citation_map helper function."""
    
    @pytest.fixture
    def sample_verification_results(self):
        """Create sample verification results for testing."""
        claims = [
            Claim(
                statement="Test claim 1",
                source_url="https://high-quality.edu/page",
                source_title="High Quality Source",
                confidence=ConfidenceLevel.HIGH,
                category="fact",
                context="",
            ),
            Claim(
                statement="Test claim 2",
                source_url="https://medium-quality.com/page",
                source_title="Medium Quality Source",
                confidence=ConfidenceLevel.MEDIUM,
                category="fact",
                context="",
            ),
        ]
        
        return [
            VerificationResult(
                claim=claims[0],
                verdict=VerificationVerdict.SUPPORTED,
                confidence=0.9,
                supporting_sources=["https://supporting.gov/page"],
                contradicting_sources=[],
                reasoning="Verified",
                verification_query="test",
            ),
            VerificationResult(
                claim=claims[1],
                verdict=VerificationVerdict.SUPPORTED,
                confidence=0.8,
                supporting_sources=[],
                contradicting_sources=[],
                reasoning="Verified",
                verification_query="test",
            ),
        ]
    
    @pytest.fixture
    def sample_source_scores(self):
        """Create sample source scores for testing."""
        return {
            "https://high-quality.edu/page": 0.90,
            "https://medium-quality.com/page": 0.60,
            "https://supporting.gov/page": 0.95,
        }
    
    def test_builds_citations_sorted_by_quality(
        self, sample_verification_results, sample_source_scores
    ):
        """Test that citations are sorted by quality score (highest first)."""
        citations, url_to_number = _build_citation_map(
            sample_verification_results,
            sample_source_scores,
        )
        
        # Should have 3 unique sources
        assert len(citations) == 3
        
        # First citation should be highest quality
        assert citations[0].source_quality_score == 0.95
        assert "supporting.gov" in citations[0].url
        
        # URL mapping should be consistent
        for citation in citations:
            assert url_to_number[citation.url] == citation.number
    
    def test_handles_empty_results(self):
        """Test that empty verification results produce empty citations."""
        citations, url_to_number = _build_citation_map([], {})
        
        assert citations == []
        assert url_to_number == {}
    
    def test_assigns_default_score_for_unknown_sources(
        self, sample_verification_results
    ):
        """Test that unknown sources get default 0.5 score."""
        # Empty source scores
        citations, _ = _build_citation_map(sample_verification_results, {})
        
        # All should have default score
        for citation in citations:
            assert citation.source_quality_score == 0.5


class TestFormatClaimsForPrompt:
    """Tests for _format_claims_for_prompt helper function."""
    
    def test_separates_by_verdict(self):
        """Test that claims are grouped by verdict."""
        claim = Claim(
            statement="Test claim",
            source_url="https://example.com",
            source_title="Example",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
        
        results = [
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
                verdict=VerificationVerdict.CONTRADICTED,
                confidence=0.8,
                supporting_sources=[],
                contradicting_sources=[],
                reasoning="Was wrong",
                verification_query="",
            ),
            VerificationResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                confidence=0.3,
                supporting_sources=[],
                contradicting_sources=[],
                reasoning="Cannot verify",
                verification_query="",
            ),
        ]
        
        _, url_to_citation = _build_citation_map(results, {})
        formatted = _format_claims_for_prompt(results, {}, url_to_citation)
        
        assert "Supported Claims" in formatted
        assert "Contradicted Claims" in formatted
        assert "Unverified Claims" in formatted


class TestParseReportResponse:
    """Tests for _parse_report_response helper function."""
    
    def test_parses_valid_json(self):
        """Test parsing a valid JSON response."""
        response = json.dumps({
            "title": "Test Report",
            "summary": "This is a summary.",
            "sections": [
                {
                    "heading": "Introduction",
                    "content": "Introduction content [1].",
                    "order": 1,
                },
                {
                    "heading": "Details",
                    "content": "Details content [2].",
                    "order": 2,
                },
            ],
            "confidence_summary": "High confidence."
        })
        
        citations = [
            Citation(
                number=1,
                title="Source 1",
                url="https://example.com/1",
                source_quality_score=0.8,
                accessed_date="2026-02-16",
            ),
        ]
        
        stats = {
            "total_latency_ms": 100,
            "llm_calls": 1,
            "total_tokens": 500,
            "sources_searched": 5,
            "claims_verified": 3,
            "fact_check_pass_rate": 0.9,
        }
        
        report = _parse_report_response(response, "Test query", citations, stats)
        
        assert report.title == "Test Report"
        assert report.summary == "This is a summary."
        assert len(report.sections) == 2
        assert report.sections[0].heading == "Introduction"
        assert report.confidence_summary == "High confidence."
        assert len(report.citations) == 1
    
    def test_parses_json_with_markdown_fences(self):
        """Test parsing JSON wrapped in markdown code fences."""
        response = '''```json
{
    "title": "Test",
    "summary": "Summary",
    "sections": [],
    "confidence_summary": "OK"
}
```'''
        
        report = _parse_report_response(response, "query", [], {})
        
        assert report.title == "Test"
    
    def test_raises_on_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _parse_report_response("not valid json", "query", [], {})
        
        assert "Invalid JSON" in str(exc_info.value)
    
    def test_sections_sorted_by_order(self):
        """Test that sections are sorted by order field."""
        response = json.dumps({
            "title": "Test",
            "summary": "Summary",
            "sections": [
                {"heading": "Third", "content": "c", "order": 3},
                {"heading": "First", "content": "a", "order": 1},
                {"heading": "Second", "content": "b", "order": 2},
            ],
            "confidence_summary": "OK"
        })
        
        report = _parse_report_response(response, "query", [], {})
        
        assert report.sections[0].heading == "First"
        assert report.sections[1].heading == "Second"
        assert report.sections[2].heading == "Third"


class TestFormatReportAsText:
    """Tests for _format_report_as_text helper function."""
    
    def test_formats_complete_report(self):
        """Test formatting a complete report."""
        report = Report(
            title="Test Report Title",
            summary="This is the executive summary.",
            sections=[
                ReportSection(heading="Intro", content="Intro content", order=1),
                ReportSection(heading="Body", content="Body content", order=2),
            ],
            confidence_summary="High confidence overall.",
            citations=[
                Citation(
                    number=1,
                    title="Source One",
                    url="https://example.com/1",
                    source_quality_score=0.85,
                    accessed_date="2026-02-16",
                ),
            ],
            metadata=ReportMetadata(query="Test query"),
        )
        
        text = _format_report_as_text(report)
        
        assert "# Test Report Title" in text
        assert "Executive Summary" in text
        assert "This is the executive summary." in text
        assert "## Intro" in text
        assert "## Body" in text
        assert "Confidence Assessment" in text
        assert "High confidence overall." in text
        assert "## References" in text
        assert "[1] Source One" in text
        assert "(high quality)" in text
    
    def test_handles_empty_report(self):
        """Test formatting a minimal/empty report."""
        report = Report(
            title="Empty",
            summary="",
            sections=[],
            confidence_summary="",
            citations=[],
            metadata=ReportMetadata(query=""),
        )
        
        text = _format_report_as_text(report)
        
        assert "# Empty" in text
        # Should not have empty sections
        assert "Executive Summary" not in text or text.count("\n\n\n") <= 1


class TestSynthesizeNode:
    """Tests for synthesize_node function."""
    
    @pytest.fixture
    def mock_state_with_results(self):
        """Create a state with verification results for testing."""
        claim = Claim(
            statement="Python is an interpreted language",
            source_url="https://python.org",
            source_title="Python.org",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="",
        )
        
        return {
            "query": "What is Python?",
            "verification_results": [
                VerificationResult(
                    claim=claim,
                    verdict=VerificationVerdict.SUPPORTED,
                    confidence=0.9,
                    supporting_sources=["https://python.org"],
                    contradicting_sources=[],
                    reasoning="Verified",
                    verification_query="Python interpreted language",
                ),
            ],
            "source_scores": {"https://python.org": 0.85},
            "research_results": [],
            "agent_trace": [],
            "iteration_count": 1,
        }
    
    @pytest.mark.asyncio
    async def test_synthesize_node_with_empty_results(self):
        """Test synthesize_node handles empty verification results."""
        state = {
            "query": "Test query",
            "verification_results": [],
            "source_scores": {},
            "research_results": [],
            "agent_trace": [],
            "iteration_count": 1,
        }
        
        result = await synthesize_node(state)
        
        assert result["status"] == "completed"
        assert "No" in result["report"] or "no" in result["report"]
        assert len(result["agent_trace"]) >= 2  # start and skip events
    
    @pytest.mark.asyncio
    async def test_synthesize_node_with_results(self, mock_state_with_results):
        """Test synthesize_node with actual verification results (mocked LLM)."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "title": "Python Overview",
            "summary": "Python is an interpreted programming language.",
            "sections": [
                {
                    "heading": "Introduction",
                    "content": "Python is widely used [1].",
                    "order": 1,
                }
            ],
            "confidence_summary": "High confidence in findings."
        })
        mock_response.usage_metadata = {"total_tokens": 500}
        
        with patch("agents.synthesizer._create_llm") as mock_create_llm:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_create_llm.return_value = mock_llm
            
            result = await synthesize_node(mock_state_with_results)
        
        assert result["status"] == "completed"
        assert "Python" in result["report"]
        assert result["report_structured"].title == "Python Overview"
        assert len(result["citations"]) >= 1
        assert len(result["agent_trace"]) >= 2
    
    @pytest.mark.asyncio
    async def test_synthesize_node_handles_llm_error(self, mock_state_with_results):
        """Test synthesize_node handles LLM errors gracefully."""
        with patch("agents.synthesizer._create_llm") as mock_create_llm:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM API error"))
            mock_create_llm.return_value = mock_llm
            
            result = await synthesize_node(mock_state_with_results)
        
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert "Synthesis error" in result["errors"][0]

