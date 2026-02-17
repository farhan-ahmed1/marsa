"""Integration tests for the Synthesizer Agent and full workflow.

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

from agents.synthesizer import (  # noqa: E402
    generate_report,
    synthesize_node,
    _build_citation_map,
    _format_claims_for_prompt,
    _format_report_as_text,
)
from graph.state import (  # noqa: E402
    AgentState,
    Claim,
    ConfidenceLevel,
    Citation,
    PipelineStatus,
    Report,
    ReportMetadata,
    ReportSection,
    VerificationResult,
    VerificationVerdict,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_verification_results():
    """Create sample verification results for testing."""
    claims = [
        Claim(
            statement="Python was created by Guido van Rossum",
            source_url="https://docs.python.org/3/faq/general.html",
            source_title="Python General FAQ",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="Historical fact about Python's creation",
        ),
        Claim(
            statement="Python 3.0 was released in December 2008",
            source_url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            source_title="Python - Wikipedia",
            confidence=ConfidenceLevel.HIGH,
            category="fact",
            context="Python version history",
        ),
        Claim(
            statement="Python uses significant whitespace for code blocks",
            source_url="https://realpython.com/python-basics/",
            source_title="Real Python Tutorial",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="Python syntax characteristics",
        ),
    ]
    
    return [
        VerificationResult(
            claim=claims[0],
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.95,
            supporting_sources=[
                "https://docs.python.org/3/faq/general.html",
                "https://en.wikipedia.org/wiki/Guido_van_Rossum",
            ],
            contradicting_sources=[],
            reasoning="Multiple authoritative sources confirm this fact.",
            verification_query="Who created Python programming language",
        ),
        VerificationResult(
            claim=claims[1],
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.90,
            supporting_sources=[
                "https://en.wikipedia.org/wiki/Python_(programming_language)",
            ],
            contradicting_sources=[],
            reasoning="Wikipedia confirms the release date.",
            verification_query="Python 3.0 release date",
        ),
        VerificationResult(
            claim=claims[2],
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.85,
            supporting_sources=[
                "https://realpython.com/python-basics/",
            ],
            contradicting_sources=[],
            reasoning="This is a well-known Python characteristic.",
            verification_query="Python whitespace syntax indentation",
        ),
    ]


@pytest.fixture
def sample_source_scores():
    """Create sample source scores for testing."""
    return {
        "https://docs.python.org/3/faq/general.html": 0.90,
        "https://en.wikipedia.org/wiki/Python_(programming_language)": 0.75,
        "https://en.wikipedia.org/wiki/Guido_van_Rossum": 0.75,
        "https://realpython.com/python-basics/": 0.65,
    }


@pytest.fixture
def sample_state(sample_verification_results, sample_source_scores):
    """Create a sample agent state for testing."""
    return {
        "query": "What is Python and who created it?",
        "verification_results": sample_verification_results,
        "source_scores": sample_source_scores,
        "research_results": [],
        "agent_trace": [],
        "iteration_count": 1,
    }


# ---------------------------------------------------------------------------
# Unit Tests (no LLM calls)
# ---------------------------------------------------------------------------


class TestBuildCitationMap:
    """Tests for _build_citation_map helper function."""
    
    def test_builds_citations_from_verification_results(
        self, sample_verification_results, sample_source_scores
    ):
        """Test that citations are correctly built from verification results."""
        citations, url_to_number = _build_citation_map(
            sample_verification_results,
            sample_source_scores,
        )
        
        # Should have citations for unique sources
        assert len(citations) >= 3
        
        # Higher quality sources should have lower citation numbers
        # (sorted by score descending)
        python_docs_citation = next(
            (c for c in citations if "docs.python.org" in c.url), None
        )
        assert python_docs_citation is not None
        assert python_docs_citation.source_quality_score == 0.90
        
        # URL mapping should be consistent
        for citation in citations:
            assert url_to_number[citation.url] == citation.number
    
    def test_handles_empty_verification_results(self, sample_source_scores):
        """Test that empty verification results produce empty citations."""
        citations, url_to_number = _build_citation_map([], sample_source_scores)
        
        assert citations == []
        assert url_to_number == {}


class TestFormatClaimsForPrompt:
    """Tests for _format_claims_for_prompt helper function."""
    
    def test_formats_supported_claims(
        self, sample_verification_results, sample_source_scores
    ):
        """Test that supported claims are formatted correctly."""
        _, url_to_citation = _build_citation_map(
            sample_verification_results,
            sample_source_scores,
        )
        
        formatted = _format_claims_for_prompt(
            sample_verification_results,
            sample_source_scores,
            url_to_citation,
        )
        
        # Should contain section headers
        assert "Supported Claims" in formatted
        
        # Should contain claim statements
        assert "Guido van Rossum" in formatted
        assert "December 2008" in formatted
    
    def test_handles_mixed_verdicts(self, sample_source_scores):
        """Test formatting with mixed verification verdicts."""
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
                confidence=0.8,
                supporting_sources=["https://example.com"],
                contradicting_sources=[],
                reasoning="Supported",
                verification_query="test",
            ),
            VerificationResult(
                claim=claim,
                verdict=VerificationVerdict.CONTRADICTED,
                confidence=0.7,
                supporting_sources=[],
                contradicting_sources=["https://other.com"],
                reasoning="Contradicted",
                verification_query="test",
            ),
            VerificationResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                confidence=0.3,
                supporting_sources=[],
                contradicting_sources=[],
                reasoning="Unverifiable",
                verification_query="test",
            ),
        ]
        
        _, url_to_citation = _build_citation_map(results, sample_source_scores)
        formatted = _format_claims_for_prompt(
            results,
            sample_source_scores,
            url_to_citation,
        )
        
        assert "Supported Claims" in formatted
        assert "Contradicted Claims" in formatted
        assert "Unverified Claims" in formatted


class TestFormatReportAsText:
    """Tests for _format_report_as_text helper function."""
    
    def test_formats_complete_report(self):
        """Test formatting a complete report as text."""
        report = Report(
            title="Test Report",
            summary="This is the summary.",
            sections=[
                ReportSection(
                    heading="Introduction",
                    content="Introduction content with [1] citation.",
                    order=1,
                ),
                ReportSection(
                    heading="Details",
                    content="Details content with [2] citation.",
                    order=2,
                ),
            ],
            confidence_summary="High confidence in findings.",
            citations=[
                Citation(
                    number=1,
                    title="Source 1",
                    url="https://example.com/1",
                    source_quality_score=0.8,
                    accessed_date="2026-02-16",
                ),
                Citation(
                    number=2,
                    title="Source 2",
                    url="https://example.com/2",
                    source_quality_score=0.6,
                    accessed_date="2026-02-16",
                ),
            ],
            metadata=ReportMetadata(query="Test query"),
        )
        
        text = _format_report_as_text(report)
        
        # Should contain title
        assert "# Test Report" in text
        
        # Should contain summary
        assert "Executive Summary" in text
        assert "This is the summary." in text
        
        # Should contain sections
        assert "## Introduction" in text
        assert "## Details" in text
        
        # Should contain confidence
        assert "Confidence Assessment" in text
        
        # Should contain references
        assert "## References" in text
        assert "[1] Source 1" in text
        assert "[2] Source 2" in text


# ---------------------------------------------------------------------------
# Integration Tests (real LLM calls)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSynthesizerIntegration:
    """Integration tests for Synthesizer agent using real LLM calls."""
    
    @pytest.mark.asyncio
    async def test_real_generate_report(
        self, sample_verification_results, sample_source_scores, sample_state
    ):
        """Test real LLM call to generate a research report.
        
        WARNING: This test makes real API calls to Anthropic Claude.
        It will consume API quota.
        """
        report = await generate_report(
            query="What is Python and who created it?",
            verification_results=sample_verification_results,
            source_scores=sample_source_scores,
            state=sample_state,
        )
        
        # Verify report structure
        assert report.title
        assert len(report.title) > 0
        
        assert report.summary
        assert len(report.summary) > 10
        
        assert len(report.sections) >= 1
        for section in report.sections:
            assert section.heading
            assert section.content
        
        assert report.confidence_summary
        
        # Verify citations
        assert len(report.citations) >= 1
        for citation in report.citations:
            assert citation.number >= 1
            assert citation.url
            assert citation.source_quality_score >= 0.0
            assert citation.source_quality_score <= 1.0
        
        # Report should mention Python
        full_text = _format_report_as_text(report)
        assert "Python" in full_text or "python" in full_text.lower()
    
    @pytest.mark.asyncio
    async def test_real_synthesize_node(
        self, sample_verification_results, sample_source_scores
    ):
        """Test real synthesize_node function with LLM call.
        
        WARNING: This test makes real API calls and consumes quota.
        """
        state: AgentState = {
            "query": "What is Python and who created it?",
            "verification_results": sample_verification_results,
            "source_scores": sample_source_scores,
            "research_results": [],
            "agent_trace": [],
            "iteration_count": 1,
        }
        
        result = await synthesize_node(state)
        
        # Verify result structure
        assert "report" in result
        assert "report_structured" in result
        assert "citations" in result
        assert "status" in result
        assert "agent_trace" in result
        
        # Verify report content
        assert len(result["report"]) > 100
        assert result["report_structured"].title
        
        # Verify status
        assert result["status"] == PipelineStatus.COMPLETED.value
        
        # Verify trace events
        assert len(result["agent_trace"]) >= 2
        
        # Check for synthesizer events
        synth_events = [
            e for e in result["agent_trace"]
            if hasattr(e, 'agent') and e.agent.value == "synthesizer"
        ]
        assert len(synth_events) >= 2  # start and complete
    
    @pytest.mark.asyncio
    async def test_synthesize_node_handles_empty_results(self):
        """Test synthesize_node handles empty verification results gracefully."""
        state: AgentState = {
            "query": "Test query with no results",
            "verification_results": [],
            "source_scores": {},
            "research_results": [],
            "agent_trace": [],
            "iteration_count": 1,
        }
        
        result = await synthesize_node(state)
        
        # Should complete without error
        assert "report" in result
        assert "status" in result
        assert result["status"] == PipelineStatus.COMPLETED.value
        
        # Should indicate no results
        assert "no" in result["report"].lower() or "No" in result["report"]
    
    @pytest.mark.asyncio
    async def test_synthesize_node_with_contradicted_claims(self, sample_source_scores):
        """Test synthesize_node handles contradicted claims appropriately."""
        claim = Claim(
            statement="Go was released in 2015",
            source_url="https://example.com",
            source_title="Example",
            confidence=ConfidenceLevel.MEDIUM,
            category="fact",
            context="",
        )
        
        verification_results = [
            VerificationResult(
                claim=claim,
                verdict=VerificationVerdict.CONTRADICTED,
                confidence=0.85,
                supporting_sources=[],
                contradicting_sources=["https://golang.org"],
                reasoning="Go was actually released in 2009, not 2015.",
                verification_query="Go programming language release date",
            ),
        ]
        
        state: AgentState = {
            "query": "When was Go released?",
            "verification_results": verification_results,
            "source_scores": sample_source_scores,
            "research_results": [],
            "agent_trace": [],
            "iteration_count": 1,
        }
        
        result = await synthesize_node(state)
        
        # Should complete
        assert result["status"] == PipelineStatus.COMPLETED.value
        
        # Should generate some report
        assert len(result["report"]) > 0
