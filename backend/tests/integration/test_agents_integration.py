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


# ---------------------------------------------------------------------------
# Fact-Checker Integration Tests
# ---------------------------------------------------------------------------

from agents.fact_checker import (  # noqa: E402
    fact_check_node,
    generate_verify_query,
    should_loop_back,
    verify_claim,
)
from graph.state import (  # noqa: E402
    Claim,
    ConfidenceLevel,
    VerificationVerdict,
)


@pytest.mark.integration
class TestFactCheckerIntegration:
    """Integration tests for Fact-Checker agent using real API calls."""
    
    @pytest.mark.asyncio
    async def test_real_generate_verify_query(self):
        """Test real LLM call to generate verification query."""
        claim = Claim(
            statement="Python was created by Guido van Rossum",
            source_url="https://python.org",
            source_title="Python.org Official Site",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="Historical fact about Python's creation",
        )
        
        query = await generate_verify_query(claim)
        
        # Should return a non-empty query string
        assert len(query) > 0
        # Should be different from the original claim
        assert query.lower() != claim.statement.lower()
        # Should likely contain Python-related terms
        assert "python" in query.lower() or "guido" in query.lower()
    
    @pytest.mark.asyncio
    async def test_real_verify_true_claim(self):
        """Test verifying a claim that is factually true.
        
        WARNING: This test makes real API calls to:
        - Anthropic Claude (verification LLM call)
        
        It will consume API quota!
        """
        claim = Claim(
            statement="Python was created by Guido van Rossum",
            source_url="https://example.com",
            source_title="Test Source",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
        
        # Create mock search results with true information
        class MockSearchResult:
            def __init__(self, url, title, content):
                self.url = url
                self.title = title
                self.content = content
        
        search_results = [
            MockSearchResult(
                url="https://docs.python.org/3/faq/general.html",
                title="Python General FAQ",
                content="Python was created by Guido van Rossum. Guido began working on Python in the late 1980s at Centrum Wiskunde & Informatica (CWI) in the Netherlands.",
            ),
            MockSearchResult(
                url="https://en.wikipedia.org/wiki/Python_(programming_language)",
                title="Python (programming language) - Wikipedia",
                content="Python was conceived in the late 1980s by Guido van Rossum at CWI. The implementation began in December 1989.",
            ),
        ]
        
        source_scores = {
            "https://docs.python.org/3/faq/general.html": 0.85,
            "https://en.wikipedia.org/wiki/Python_(programming_language)": 0.75,
        }
        
        result = await verify_claim(
            claim=claim,
            search_results=search_results,
            source_scores=source_scores,
            verification_query="Python programming language creator origin",
        )
        
        # This true claim should be supported
        assert result.verdict == VerificationVerdict.SUPPORTED
        assert result.confidence >= 0.7
        assert len(result.reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_real_verify_false_claim(self):
        """Test verifying a claim that is factually false.
        
        WARNING: This test makes real API calls to:
        - Anthropic Claude (verification LLM call)
        
        It will consume API quota!
        """
        claim = Claim(
            statement="Go was released in 2015",
            source_url="https://example.com",
            source_title="Test Source",
            confidence=ConfidenceLevel.LOW,
            category="fact",
            context="",
        )
        
        # Create mock search results with correct information
        class MockSearchResult:
            def __init__(self, url, title, content):
                self.url = url
                self.title = title
                self.content = content
        
        search_results = [
            MockSearchResult(
                url="https://go.dev/doc/faq",
                title="Go FAQ",
                content="Go was first publicly announced in November 2009. The first stable release, Go 1, was released in March 2012.",
            ),
            MockSearchResult(
                url="https://en.wikipedia.org/wiki/Go_(programming_language)",
                title="Go (programming language) - Wikipedia",
                content="Go is a statically typed language designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson. It was released as open source in November 2009.",
            ),
        ]
        
        source_scores = {
            "https://go.dev/doc/faq": 0.85,
            "https://en.wikipedia.org/wiki/Go_(programming_language)": 0.75,
        }
        
        result = await verify_claim(
            claim=claim,
            search_results=search_results,
            source_scores=source_scores,
            verification_query="Go programming language release date",
        )
        
        # This false claim should be contradicted (Go was released in 2009, not 2015)
        assert result.verdict == VerificationVerdict.CONTRADICTED
        assert result.confidence >= 0.7
        assert "2009" in result.reasoning.lower() or "contradicted" in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_real_fact_check_node_simple(self):
        """Test real fact_check_node with actual API calls.
        
        WARNING: This test makes real API calls to:
        - Anthropic Claude (query generation + verification)
        - Tavily (web search)
        
        It will consume API quota!
        """
        claims = [
            Claim(
                statement="Python was created by Guido van Rossum",
                source_url="https://python.org",
                source_title="Python.org",
                confidence=ConfidenceLevel.HIGH,
                category="fact",
                context="Historical fact",
            ),
        ]
        
        state: AgentState = {
            "claims": claims,
            "source_scores": {},
            "agent_trace": [],
            "iteration_count": 0,
        }
        
        result = await fact_check_node(state)
        
        # Verify result structure
        assert "verification_results" in result
        assert "source_scores" in result
        assert "agent_trace" in result
        assert "iteration_count" in result
        
        # Should have verification results
        assert len(result["verification_results"]) == 1
        
        # The true claim should likely be supported
        vr = result["verification_results"][0]
        assert vr.verdict in [
            VerificationVerdict.SUPPORTED,
            VerificationVerdict.UNVERIFIABLE,  # May be unverifiable if search fails
        ]
        assert vr.claim.statement == claims[0].statement
        
        # Should have trace events
        assert len(result["agent_trace"]) > 0
        actions = [e.action for e in result["agent_trace"]]
        assert "start" in actions
    
    @pytest.mark.asyncio
    async def test_should_loop_back_with_bad_claims(self):
        """Test loop-back logic with injected bad claims.
        
        This test verifies that when >30% of claims are bad,
        the system correctly routes back to the researcher.
        """
        from graph.state import VerificationResult
        
        def create_claim(statement: str) -> Claim:
            return Claim(
                statement=statement,
                source_url="https://example.com",
                source_title="Test",
                confidence=ConfidenceLevel.MEDIUM,
                category="fact",
                context="",
            )
        
        def create_result(claim: Claim, verdict: VerificationVerdict) -> VerificationResult:
            return VerificationResult(
                claim=claim,
                verdict=verdict,
                confidence=0.5,
                supporting_sources=[],
                contradicting_sources=[],
                reasoning="Test",
                verification_query="test",
            )
        
        # Create 50% bad claims (should trigger loop-back)
        claims = [
            create_claim("True claim 1"),
            create_claim("True claim 2"),
            create_claim("False claim 1"),
            create_claim("False claim 2"),
        ]
        
        verification_results = [
            create_result(claims[0], VerificationVerdict.SUPPORTED),
            create_result(claims[1], VerificationVerdict.SUPPORTED),
            create_result(claims[2], VerificationVerdict.CONTRADICTED),
            create_result(claims[3], VerificationVerdict.UNVERIFIABLE),
        ]
        
        state: AgentState = {
            "verification_results": verification_results,
            "iteration_count": 0,
        }
        
        # 50% bad should trigger loop-back
        assert should_loop_back(state) == "researcher"
        
        # At max iterations, should proceed to synthesizer
        state["iteration_count"] = 2
        assert should_loop_back(state) == "synthesizer"
