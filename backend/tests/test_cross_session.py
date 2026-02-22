"""Unit tests for cross-session memory module.

Tests cover helper functions, store_session, get_relevant_memories, and
get_memory_count with mocked ChromaDB and OpenAI clients.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from graph.state import (  # noqa: E402
    AgentName,
    Claim,
    ConfidenceLevel,
    PipelineStatus,
    QueryPlan,
    QueryType,
    SearchStrategy,
    ComplexityLevel,
    VerificationResult,
    VerificationVerdict,
    create_initial_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides):
    state = create_initial_state("What is quantum computing?")
    state.update(overrides)
    return state


def _make_claim(statement="Python is popular", verdict=VerificationVerdict.SUPPORTED):
    return Claim(
        statement=statement,
        source_url="https://example.com",
        source_title="Example",
        confidence=ConfidenceLevel.HIGH,
        category="fact",
        context="",
    )


def _make_verification(claim=None, verdict=VerificationVerdict.SUPPORTED):
    if claim is None:
        claim = _make_claim()
    return VerificationResult(
        claim=claim,
        verdict=verdict,
        confidence=0.9,
        supporting_sources=[],
        contradicting_sources=[],
        reasoning="Verified",
        verification_query="test",
    )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


class TestExtractTopics:
    """Tests for _extract_topics."""

    def test_extracts_sub_queries_from_plan(self):
        from memory.cross_session import _extract_topics

        plan = QueryPlan(
            query_type=QueryType.FACTUAL,
            sub_queries=["sub1", "sub2"],
            parallel=False,
            needs_fact_check=True,
            search_strategy=SearchStrategy.WEB_ONLY,
            estimated_complexity=ComplexityLevel.LOW,
        )
        state = _make_state(plan=plan)
        topics = _extract_topics(state)
        assert "What is quantum computing?" in topics
        assert "sub1" in topics
        assert "sub2" in topics

    def test_includes_original_query(self):
        from memory.cross_session import _extract_topics

        state = _make_state(plan=None)
        topics = _extract_topics(state)
        assert "What is quantum computing?" in topics

    def test_caps_at_10(self):
        from memory.cross_session import _extract_topics

        plan = QueryPlan(
            query_type=QueryType.FACTUAL,
            sub_queries=[f"sub{i}" for i in range(15)],
            parallel=False,
            needs_fact_check=True,
            search_strategy=SearchStrategy.WEB_ONLY,
            estimated_complexity=ComplexityLevel.LOW,
        )
        state = _make_state(plan=plan)
        topics = _extract_topics(state)
        assert len(topics) <= 10


class TestExtractKeyFindings:
    """Tests for _extract_key_findings."""

    def test_extracts_supported_claims(self):
        from memory.cross_session import _extract_key_findings

        claim1 = _make_claim("Finding A")
        claim2 = _make_claim("Finding B")
        state = _make_state(
            verification_results=[
                _make_verification(claim1, VerificationVerdict.SUPPORTED),
                _make_verification(claim2, VerificationVerdict.CONTRADICTED),
            ]
        )
        findings = _extract_key_findings(state)
        assert "Finding A" in findings
        assert "Finding B" not in findings

    def test_caps_at_20(self):
        from memory.cross_session import _extract_key_findings

        results = [
            _make_verification(
                _make_claim(f"Finding {i}"),
                VerificationVerdict.SUPPORTED,
            )
            for i in range(25)
        ]
        state = _make_state(verification_results=results)
        findings = _extract_key_findings(state)
        assert len(findings) <= 20


class TestExtractSourceQuality:
    """Tests for _extract_source_quality."""

    def test_returns_top_10(self):
        from memory.cross_session import _extract_source_quality

        scores = {f"https://source{i}.com": float(i) for i in range(15)}
        state = _make_state(source_scores=scores)
        result = _extract_source_quality(state)
        assert len(result) <= 10

    def test_truncates_long_urls(self):
        from memory.cross_session import _extract_source_quality

        long_url = "https://example.com/" + "a" * 200
        state = _make_state(source_scores={long_url: 0.9})
        result = _extract_source_quality(state)
        for key in result:
            assert len(key) <= 120


class TestComputeFactCheckPassRate:
    """Tests for _compute_fact_check_pass_rate."""

    def test_all_supported(self):
        from memory.cross_session import _compute_fact_check_pass_rate

        state = _make_state(
            verification_results=[
                _make_verification(verdict=VerificationVerdict.SUPPORTED),
                _make_verification(verdict=VerificationVerdict.SUPPORTED),
            ]
        )
        assert _compute_fact_check_pass_rate(state) == 1.0

    def test_mixed_results(self):
        from memory.cross_session import _compute_fact_check_pass_rate

        state = _make_state(
            verification_results=[
                _make_verification(verdict=VerificationVerdict.SUPPORTED),
                _make_verification(verdict=VerificationVerdict.CONTRADICTED),
            ]
        )
        assert _compute_fact_check_pass_rate(state) == 0.5

    def test_empty_results(self):
        from memory.cross_session import _compute_fact_check_pass_rate

        state = _make_state(verification_results=[])
        assert _compute_fact_check_pass_rate(state) == 0.0


class TestBuildSummary:
    """Tests for _build_summary."""

    def test_includes_query_and_topics(self):
        from memory.cross_session import _build_summary

        summary = _build_summary(
            query="Test query",
            topics=["topic1", "topic2"],
            key_findings=["finding1"],
            fact_check_pass_rate=0.8,
        )
        assert "Test query" in summary
        assert "topic1" in summary
        assert "finding1" in summary
        assert "80%" in summary

    def test_truncates_long_summary(self):
        from memory.cross_session import _build_summary, MAX_SUMMARY_CHARS

        findings = [f"A very long finding number {i} " * 20 for i in range(50)]
        summary = _build_summary("q", ["t"], findings, 1.0)
        assert len(summary) <= MAX_SUMMARY_CHARS


# ---------------------------------------------------------------------------
# store_session (mocked)
# ---------------------------------------------------------------------------


class TestStoreSession:
    """Tests for store_session with mocked external services."""

    def test_stores_session_successfully(self):
        from memory.cross_session import store_session

        mock_collection = MagicMock()
        mock_embed_response = MagicMock()
        mock_embed_response.data = [MagicMock(embedding=[0.1] * 10)]

        with (
            patch("memory.cross_session._get_memory_collection", return_value=mock_collection),
            patch("memory.cross_session._embed", return_value=[0.1] * 10),
        ):
            state = _make_state(
                verification_results=[
                    _make_verification(verdict=VerificationVerdict.SUPPORTED),
                ],
                source_scores={"https://example.com": 0.9},
            )
            store_session(state)

        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args
        assert len(call_kwargs.kwargs.get("ids", call_kwargs[1].get("ids", []))) == 1

    def test_skips_empty_query(self):
        from memory.cross_session import store_session

        state = _make_state(query="")
        with patch("memory.cross_session._get_memory_collection") as mock_coll:
            store_session(state)
        mock_coll.assert_not_called()

    def test_handles_embedding_error(self):
        from memory.cross_session import store_session

        with (
            patch("memory.cross_session._embed", side_effect=RuntimeError("API error")),
            patch("memory.cross_session._get_memory_collection"),
        ):
            # Should not raise
            store_session(_make_state())


# ---------------------------------------------------------------------------
# get_relevant_memories (mocked)
# ---------------------------------------------------------------------------


class TestGetRelevantMemories:
    """Tests for get_relevant_memories with mocked ChromaDB."""

    def test_returns_context_string(self):
        from memory.cross_session import get_relevant_memories

        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "documents": [["Research query: test\nTopics: t1\nKey verified findings: f1"]],
            "metadatas": [[{
                "query": "test",
                "timestamp": "2025-01-01T00:00:00",
                "fact_check_pass_rate": 0.9,
                "topics_json": json.dumps(["topic1"]),
                "key_findings_json": json.dumps(["finding1"]),
            }]],
            "distances": [[0.2]],
        }

        with (
            patch("memory.cross_session._get_memory_collection", return_value=mock_collection),
            patch("memory.cross_session._embed", return_value=[0.1] * 10),
        ):
            result = get_relevant_memories("test query")

        assert "Prior Research Context" in result
        assert "topic1" in result

    def test_returns_empty_for_no_matches(self):
        from memory.cross_session import get_relevant_memories

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        with patch("memory.cross_session._get_memory_collection", return_value=mock_collection):
            result = get_relevant_memories("test")

        assert result == ""

    def test_filters_low_similarity(self):
        from memory.cross_session import get_relevant_memories

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "documents": [["irrelevant doc"]],
            "metadatas": [[{
                "query": "unrelated",
                "timestamp": "2025-01-01",
                "fact_check_pass_rate": 0.5,
                "topics_json": "[]",
                "key_findings_json": "[]",
            }]],
            "distances": [[0.9]],  # High distance = low similarity
        }

        with (
            patch("memory.cross_session._get_memory_collection", return_value=mock_collection),
            patch("memory.cross_session._embed", return_value=[0.1] * 10),
        ):
            result = get_relevant_memories("test")

        assert result == ""

    def test_handles_error_gracefully(self):
        from memory.cross_session import get_relevant_memories

        with patch("memory.cross_session._get_memory_collection", side_effect=RuntimeError("DB error")):
            result = get_relevant_memories("test")

        assert result == ""


# ---------------------------------------------------------------------------
# get_memory_count (mocked)
# ---------------------------------------------------------------------------


class TestGetMemoryCount:
    """Tests for get_memory_count."""

    def test_returns_count(self):
        from memory.cross_session import get_memory_count

        mock_collection = MagicMock()
        mock_collection.count.return_value = 42

        with patch("memory.cross_session._get_memory_collection", return_value=mock_collection):
            assert get_memory_count() == 42

    def test_returns_zero_on_error(self):
        from memory.cross_session import get_memory_count

        with patch("memory.cross_session._get_memory_collection", side_effect=RuntimeError("fail")):
            assert get_memory_count() == 0
