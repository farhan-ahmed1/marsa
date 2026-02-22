"""Tests for the mock evaluation system.

Verifies that mock_eval produces valid EvaluationResult objects,
correct aggregate metrics, and works end-to-end without external APIs.
"""

import json
import sys
import tempfile
from pathlib import Path


backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from eval.mock_eval import (  # noqa: E402
    _generate_mock_citations,
    _generate_mock_report,
    _generate_mock_trace_events,
    _generate_mock_verification_results,
    _seed_from_id,
    compare_results,
    run_mock_evaluation,
    run_mock_single_query,
)
from eval.metrics import EvaluationResult  # noqa: E402

import random  # noqa: E402


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------


class TestSeedFromId:
    """Test deterministic seed generation."""

    def test_same_id_same_seed(self):
        assert _seed_from_id("factual_01") == _seed_from_id("factual_01")

    def test_different_id_different_seed(self):
        assert _seed_from_id("factual_01") != _seed_from_id("factual_02")


# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------


class TestMockDataGenerators:
    """Tests for individual mock data generation functions."""

    def test_generate_mock_report_includes_must_mention(self):
        rng = random.Random(42)
        report = _generate_mock_report(
            "What is Python?",
            must_mention=["Guido van Rossum", "dynamic typing"],
            must_not_claim=[],
            false_claim=None,
            rng=rng,
        )
        assert "Guido van Rossum" in report
        assert "dynamic typing" in report

    def test_generate_mock_report_handles_false_claim(self):
        rng = random.Random(42)
        report = _generate_mock_report(
            "Why was Go released in 2015?",
            must_mention=["2009"],
            must_not_claim=[],
            false_claim="Go was released in 2015",
            rng=rng,
        )
        assert "Correction" in report
        assert "incorrect" in report.lower()

    def test_generate_mock_citations(self):
        rng = random.Random(42)
        citations = _generate_mock_citations(5, rng)
        assert len(citations) == 5
        for c in citations:
            assert "url" in c
            assert c["url"].startswith("https://")

    def test_generate_mock_verification_results(self):
        rng = random.Random(42)
        results = _generate_mock_verification_results(5, "factual", None, rng)
        assert len(results) == 5
        for r in results:
            assert r["verdict"] in ("supported", "contradicted", "unverifiable")

    def test_generate_mock_verification_catches_false_claim(self):
        rng = random.Random(42)
        results = _generate_mock_verification_results(
            3, "false_premise", "Python was created in 2000", rng
        )
        # First claim should be the false one, marked contradicted
        assert results[0]["verdict"] == "contradicted"
        assert results[0]["claim"]["statement"] == "Python was created in 2000"

    def test_generate_mock_trace_events(self):
        rng = random.Random(42)
        events = _generate_mock_trace_events("factual", rng)
        assert len(events) > 0
        agents_seen = {e["agent"] for e in events}
        assert "planner" in agents_seen
        assert "researcher" in agents_seen
        assert "fact_checker" in agents_seen
        assert "synthesizer" in agents_seen


# ---------------------------------------------------------------------------
# Single query evaluation
# ---------------------------------------------------------------------------


class TestRunMockSingleQuery:
    """Tests for single mock query evaluation."""

    def test_factual_query(self):
        query_data = {
            "id": "factual_01",
            "query": "What is the CAP theorem?",
            "category": "factual",
            "expected_sub_queries_min": 1,
            "expected_sources_min": 3,
            "must_mention": ["consistency", "availability"],
            "must_not_claim": [],
        }
        result = run_mock_single_query(query_data)

        assert isinstance(result, EvaluationResult)
        assert result.success is True
        assert result.query_id == "factual_01"
        assert result.citation_metrics.total_citations > 0
        assert result.quality_scores.overall > 0

    def test_false_premise_query(self):
        query_data = {
            "id": "false_premise_01",
            "query": "Why was Go released in 2015?",
            "category": "false_premise",
            "expected_sub_queries_min": 2,
            "expected_sources_min": 3,
            "must_mention": ["2009"],
            "must_not_claim": ["Go was released in 2015"],
            "false_claim": "Go was released in 2015",
        }
        result = run_mock_single_query(query_data)

        assert result.success is True
        assert result.fact_check_metrics.false_premise_caught is True
        assert "2009" in result.must_mention_found

    def test_deterministic_results(self):
        query_data = {
            "id": "test_determinism",
            "query": "Test",
            "category": "factual",
            "expected_sub_queries_min": 1,
            "expected_sources_min": 2,
            "must_mention": [],
            "must_not_claim": [],
        }
        r1 = run_mock_single_query(query_data)
        r2 = run_mock_single_query(query_data)
        assert r1.quality_scores.overall == r2.quality_scores.overall
        assert r1.citation_metrics.total_citations == r2.citation_metrics.total_citations


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------


class TestRunMockEvaluation:
    """Tests for the full mock evaluation run."""

    def test_runs_all_queries(self):
        results, aggregate = run_mock_evaluation()
        assert len(results) == 20  # All queries from test_queries.json
        assert aggregate.total_queries == 20
        assert aggregate.successful_queries == 20
        assert aggregate.failed_queries == 0

    def test_filters_by_category(self):
        results, aggregate = run_mock_evaluation(category="factual")
        assert all(r.category == "factual" for r in results)
        assert len(results) == 5

    def test_limits_results(self):
        results, aggregate = run_mock_evaluation(limit=3)
        assert len(results) == 3

    def test_aggregate_metrics_populated(self):
        results, aggregate = run_mock_evaluation()
        assert aggregate.avg_overall_quality > 0
        assert aggregate.avg_citation_accuracy > 0
        assert aggregate.avg_fact_check_pass_rate > 0
        assert aggregate.total_tokens > 0
        assert aggregate.total_llm_calls > 0
        assert len(aggregate.metrics_by_category) > 0


# ---------------------------------------------------------------------------
# Save and compare
# ---------------------------------------------------------------------------


class TestSaveAndCompare:
    """Tests for result persistence and comparison."""

    def test_save_results(self):
        results, aggregate = run_mock_evaluation(limit=2)
        with tempfile.TemporaryDirectory() as _tmpdir:
            # Patch output_dir
            import eval.mock_eval as me
            _original_dir = Path("data/eval_results")
            try:
                output_path = me.save_mock_results(
                    results, aggregate, "test_output.json"
                )
                # Check file was created
                assert output_path.exists()
                with open(output_path) as f:
                    data = json.load(f)
                assert data["mode"] == "mock"
                assert len(data["results"]) == 2
            finally:
                # Cleanup
                if output_path.exists():
                    output_path.unlink()

    def test_compare_results(self):
        results1, aggregate1 = run_mock_evaluation(limit=5)
        results2, aggregate2 = run_mock_evaluation(limit=5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "aggregate_metrics": aggregate1.model_dump(),
                "results": [r.model_dump() for r in results1],
            }, f)
            prev_path = f.name

        try:
            comparisons = compare_results(aggregate2, prev_path)
            assert "avg_overall_quality" in comparisons
            assert "avg_citation_accuracy" in comparisons
            # Same data, so delta should be 0
            for field, data in comparisons.items():
                assert data["delta"] == 0.0
        finally:
            Path(prev_path).unlink()
