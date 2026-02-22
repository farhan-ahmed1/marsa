"""Tests for the evaluation metrics module.

Tests citation metrics, fact-check metrics, latency metrics,
token metrics, quality scoring, and aggregate calculations
with fully mocked data (no API calls).
"""

import sys
from pathlib import Path

import pytest

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from eval.metrics import (  # noqa: E402
    AggregateMetrics,
    CitationMetrics,
    EvaluationResult,
    FactCheckMetrics,
    LatencyMetrics,
    QualityScores,
    TokenMetrics,
    calculate_aggregate_metrics,
    calculate_citation_metrics,
    calculate_fact_check_metrics,
    calculate_latency_metrics,
    calculate_token_metrics,
    check_must_mention,
    check_must_not_claim,
    extract_domain,
    is_valid_url,
)


# ---------------------------------------------------------------------------
# URL Validation Tests
# ---------------------------------------------------------------------------

class TestIsValidUrl:
    """Tests for URL validation."""

    def test_valid_https_url(self):
        assert is_valid_url("https://example.com/page") is True

    def test_valid_http_url(self):
        assert is_valid_url("http://example.com") is True

    def test_empty_string(self):
        assert is_valid_url("") is False

    def test_no_scheme(self):
        assert is_valid_url("example.com") is False

    def test_ftp_scheme(self):
        assert is_valid_url("ftp://example.com") is False

    def test_none_like_input(self):
        assert is_valid_url("") is False


class TestExtractDomain:
    """Tests for domain extraction."""

    def test_simple_domain(self):
        assert extract_domain("https://example.com/page") == "example.com"

    def test_www_removed(self):
        assert extract_domain("https://www.example.com") == "example.com"

    def test_subdomain_preserved(self):
        assert extract_domain("https://docs.python.org/3/") == "docs.python.org"

    def test_empty_url(self):
        assert extract_domain("") is None

    def test_invalid_url(self):
        assert extract_domain("not-a-url") is None


# ---------------------------------------------------------------------------
# Citation Metrics Tests
# ---------------------------------------------------------------------------

class TestCalculateCitationMetrics:
    """Tests for citation metric calculations."""

    def test_empty_citations(self):
        metrics = calculate_citation_metrics([])
        assert metrics.total_citations == 0
        assert metrics.accuracy == 0.0

    def test_all_valid_citations(self):
        citations = [
            {"url": "https://example.com/1", "title": "Source 1"},
            {"url": "https://docs.python.org/3/", "title": "Python Docs"},
            {"url": "https://arxiv.org/abs/1234", "title": "Arxiv Paper"},
        ]
        metrics = calculate_citation_metrics(citations)
        assert metrics.total_citations == 3
        assert metrics.valid_urls == 3
        assert metrics.accuracy == 1.0
        assert metrics.unique_domains == 3

    def test_mixed_valid_invalid_citations(self):
        citations = [
            {"url": "https://example.com", "title": "Valid"},
            {"url": "", "title": "Empty URL"},
            {"url": "not-a-url", "title": "Invalid"},
        ]
        metrics = calculate_citation_metrics(citations)
        assert metrics.total_citations == 3
        assert metrics.valid_urls == 1
        assert 0.3 <= metrics.accuracy <= 0.4

    def test_duplicate_domains(self):
        citations = [
            {"url": "https://example.com/page1", "title": "Page 1"},
            {"url": "https://example.com/page2", "title": "Page 2"},
        ]
        metrics = calculate_citation_metrics(citations)
        assert metrics.unique_domains == 1


# ---------------------------------------------------------------------------
# Fact-Check Metrics Tests
# ---------------------------------------------------------------------------

class TestCalculateFactCheckMetrics:
    """Tests for fact-check metric calculations."""

    def test_empty_results(self):
        metrics = calculate_fact_check_metrics([])
        assert metrics.total_claims == 0
        assert metrics.pass_rate == 0.0

    def test_all_supported(self):
        results = [
            {"verdict": "supported", "claim": {"statement": "fact 1"}},
            {"verdict": "supported", "claim": {"statement": "fact 2"}},
        ]
        metrics = calculate_fact_check_metrics(results)
        assert metrics.total_claims == 2
        assert metrics.supported_claims == 2
        assert metrics.pass_rate == 1.0

    def test_mixed_verdicts(self):
        results = [
            {"verdict": "supported", "claim": {"statement": "fact 1"}},
            {"verdict": "contradicted", "claim": {"statement": "false claim"}},
            {"verdict": "unverifiable", "claim": {"statement": "unknown"}},
        ]
        metrics = calculate_fact_check_metrics(results)
        assert metrics.total_claims == 3
        assert metrics.supported_claims == 1
        assert metrics.contradicted_claims == 1
        assert metrics.unverifiable_claims == 1
        assert abs(metrics.pass_rate - 1 / 3) < 0.01

    def test_false_premise_caught(self):
        results = [
            {
                "verdict": "contradicted",
                "claim": {"statement": "Go was released in 2015"},
            },
        ]
        metrics = calculate_fact_check_metrics(
            results, false_premise_claim="Go was released in 2015"
        )
        assert metrics.false_premise_caught is True

    def test_false_premise_not_caught(self):
        results = [
            {
                "verdict": "supported",
                "claim": {"statement": "Go was released in 2015"},
            },
        ]
        metrics = calculate_fact_check_metrics(
            results, false_premise_claim="Go was released in 2015"
        )
        assert metrics.false_premise_caught is False


# ---------------------------------------------------------------------------
# Latency Metrics Tests
# ---------------------------------------------------------------------------

class TestCalculateLatencyMetrics:
    """Tests for latency metric calculations."""

    def test_empty_trace(self):
        metrics = calculate_latency_metrics([], total_latency_ms=0.0)
        assert metrics.total_ms == 0.0

    def test_basic_latency(self):
        trace = [
            {"agent": "planner", "action": "start", "latency_ms": 100},
            {"agent": "planner", "action": "complete", "latency_ms": 200},
            {"agent": "researcher", "action": "web_search", "latency_ms": 500},
            {"agent": "fact_checker", "action": "verify", "latency_ms": 300},
            {"agent": "synthesizer", "action": "generate", "latency_ms": 400},
        ]
        metrics = calculate_latency_metrics(trace, total_latency_ms=1500)
        assert metrics.total_ms == 1500


# ---------------------------------------------------------------------------
# Token Metrics Tests
# ---------------------------------------------------------------------------

class TestCalculateTokenMetrics:
    """Tests for token metric calculations."""

    def test_empty_trace(self):
        metrics = calculate_token_metrics([], quality_score=3.0)
        assert metrics.total_tokens == 0
        assert metrics.llm_calls == 0

    def test_basic_token_usage(self):
        trace = [
            {"agent": "planner", "action": "llm_call", "tokens_used": 500},
            {"agent": "researcher", "action": "llm_call", "tokens_used": 1000},
            {"agent": "fact_checker", "action": "llm_call", "tokens_used": 800},
            {"agent": "synthesizer", "action": "llm_call", "tokens_used": 1200},
        ]
        metrics = calculate_token_metrics(trace, quality_score=4.0)
        assert metrics.total_tokens == 3500
        assert metrics.llm_calls == 4
        assert abs(metrics.tokens_per_call - 875.0) < 0.01


# ---------------------------------------------------------------------------
# Must Mention / Must Not Claim Tests
# ---------------------------------------------------------------------------

class TestCheckMustMention:
    """Tests for must_mention checks."""

    def test_all_mentioned(self):
        report = "The CAP theorem covers consistency, availability, and partition tolerance."
        found, missing = check_must_mention(
            report, ["consistency", "availability", "partition tolerance"]
        )
        assert len(found) == 3
        assert len(missing) == 0

    def test_some_missing(self):
        report = "The CAP theorem covers consistency and availability."
        found, missing = check_must_mention(
            report, ["consistency", "availability", "partition tolerance"]
        )
        assert len(found) == 2
        assert "partition tolerance" in missing

    def test_empty_must_mention(self):
        found, missing = check_must_mention("Any report text", [])
        assert found == []
        assert missing == []

    def test_case_insensitive_match(self):
        report = "Guido van Rossum created Python."
        found, missing = check_must_mention(report, ["guido van rossum"])
        assert len(found) == 1
        assert len(missing) == 0


class TestCheckMustNotClaim:
    """Tests for must_not_claim checks."""

    def test_no_violations(self):
        report = "The sky is blue on a clear day."
        violations = check_must_not_claim(
            report, ["Python was created in 2000"]
        )
        assert len(violations) == 0

    def test_violation_detected(self):
        report = "Go was released in 2015 as a general-purpose language."
        violations = check_must_not_claim(
            report, ["Go was released in 2015"]
        )
        assert len(violations) >= 1

    def test_empty_forbidden(self):
        violations = check_must_not_claim("Any text", [])
        assert violations == []


# ---------------------------------------------------------------------------
# Aggregate Metrics Tests
# ---------------------------------------------------------------------------

class TestCalculateAggregateMetrics:
    """Tests for aggregate metric calculations."""

    def test_empty_results(self):
        metrics = calculate_aggregate_metrics([])
        assert metrics.total_queries == 0
        assert metrics.successful_queries == 0

    def test_single_result(self):
        result = EvaluationResult(
            query_id="test_01",
            query="Test query",
            category="factual",
            success=True,
            citation_metrics=CitationMetrics(
                total_citations=3, valid_urls=3, accuracy=1.0,
                unique_domains=3, domain_list=["a.com", "b.com", "c.com"],
            ),
            fact_check_metrics=FactCheckMetrics(
                total_claims=5, supported_claims=4, contradicted_claims=0,
                unverifiable_claims=1, pass_rate=0.8,
            ),
            latency_metrics=LatencyMetrics(total_ms=2000, p50_ms=500, p95_ms=1500),
            quality_scores=QualityScores(
                relevance=4, accuracy=4, completeness=3,
                citation_quality=4, overall=3.8,
            ),
        )
        metrics = calculate_aggregate_metrics([result])
        assert metrics.total_queries == 1
        assert metrics.successful_queries == 1
        assert metrics.failed_queries == 0
        assert metrics.avg_overall_quality == 3.8

    def test_mixed_success_failure(self):
        results = [
            EvaluationResult(
                query_id="ok_01", query="Good query", category="factual",
                success=True,
                quality_scores=QualityScores(
                    relevance=4, accuracy=4, completeness=4,
                    citation_quality=4, overall=4.0,
                ),
            ),
            EvaluationResult(
                query_id="fail_01", query="Bad query", category="comparison",
                success=False, error_message="Timeout",
            ),
        ]
        metrics = calculate_aggregate_metrics(results)
        assert metrics.total_queries == 2
        assert metrics.successful_queries == 1
        assert metrics.failed_queries == 1


# ---------------------------------------------------------------------------
# Quality Scores Model Tests
# ---------------------------------------------------------------------------

class TestQualityScores:
    """Tests for QualityScores model."""

    def test_default_scores(self):
        scores = QualityScores()
        assert scores.relevance == 0
        assert scores.overall == 0.0

    def test_valid_scores(self):
        scores = QualityScores(
            relevance=5, accuracy=4, completeness=3,
            citation_quality=4, overall=4.0,
        )
        assert scores.relevance == 5
        assert scores.overall == 4.0


# ---------------------------------------------------------------------------
# Evaluation Result Model Tests
# ---------------------------------------------------------------------------

class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_default_values(self):
        result = EvaluationResult(
            query_id="test", query="test query", category="factual",
        )
        assert result.success is True
        assert result.error_message is None
        assert result.actual_sub_queries == 0
        assert result.actual_sources == 0

    def test_model_dump(self):
        result = EvaluationResult(
            query_id="test", query="test query", category="factual",
        )
        data = result.model_dump()
        assert "query_id" in data
        assert "citation_metrics" in data
        assert "fact_check_metrics" in data
