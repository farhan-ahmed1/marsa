"""Mock evaluation runner for MARSA.

Produces deterministic evaluation results WITHOUT calling Tavily, Claude, or
any external API.  Uses realistic synthetic data that exercises the full
metrics pipeline (citations, fact-check, latency, tokens, quality).

Usage:
    python -m eval.mock_eval                        # Run all mock queries
    python -m eval.mock_eval --category factual     # Run specific category
    python -m eval.mock_eval --compare results.json # Compare with prior run

The output format matches ``run_eval.py`` so results can be compared.
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from eval.metrics import (
    AggregateMetrics,
    EvaluationResult,
    QualityScores,
    calculate_aggregate_metrics,
    calculate_citation_metrics,
    calculate_fact_check_metrics,
    calculate_latency_metrics,
    calculate_token_metrics,
    check_must_mention,
    check_must_not_claim,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Deterministic synthetic data per query ID
# ---------------------------------------------------------------------------

# Quality profiles by category -- (relevance, accuracy, completeness, citation_quality)
_QUALITY_PROFILES: dict[str, tuple[int, int, int, int]] = {
    "factual": (4, 4, 4, 4),
    "comparison": (4, 4, 3, 4),
    "exploratory": (3, 3, 3, 3),
    "false_premise": (4, 3, 3, 3),
    "doc_context": (3, 3, 3, 3),
}

# Domains used for synthetic sources
_MOCK_DOMAINS = [
    "en.wikipedia.org",
    "docs.python.org",
    "developer.mozilla.org",
    "stackoverflow.com",
    "github.com",
    "arxiv.org",
    "medium.com",
    "devblogs.microsoft.com",
    "cloud.google.com",
    "aws.amazon.com",
]


def _seed_from_id(query_id: str) -> int:
    """Derive a deterministic seed from the query ID."""
    return int(hashlib.md5(query_id.encode()).hexdigest()[:8], 16)


def _generate_mock_report(
    query: str,
    must_mention: list[str],
    must_not_claim: list[str],
    false_claim: Optional[str],
    rng: random.Random,
) -> str:
    """Build a synthetic report that deliberately includes ``must_mention``
    terms and avoids ``must_not_claim`` phrases."""
    paragraphs = [
        f"# Research Report: {query}\n",
        "## Summary\n",
        f"This report examines: {query}. ",
    ]

    # Inject must_mention terms
    for term in must_mention:
        paragraphs.append(
            f"One important aspect is {term}, which plays a significant role. "
        )

    # Add filler body
    paragraphs.append(
        "\n## Analysis\n"
        "Based on the sources reviewed, the following analysis "
        "provides a comprehensive overview of the topic. "
        "Multiple sources confirm the main findings [1][2]. "
        "Additional research supports these conclusions [3][4]. "
    )

    # For false-premise queries, add correction
    if false_claim:
        paragraphs.append(
            f"\n## Correction\nNote: the premise that '{false_claim}' is incorrect. "
            "The factual record contradicts this claim. "
        )

    paragraphs.append(
        "\n## Sources\n"
        "This report draws upon multiple authoritative sources to support its findings."
    )

    return "\n".join(paragraphs)


def _generate_mock_citations(
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Generate ``n`` synthetic citation dicts with valid URLs."""
    used_domains = rng.sample(_MOCK_DOMAINS, min(n, len(_MOCK_DOMAINS)))
    citations = []
    for i, domain in enumerate(used_domains):
        citations.append({
            "id": i + 1,
            "url": f"https://{domain}/article/{rng.randint(1000, 9999)}",
            "title": f"Source {i + 1}: {domain}",
            "domain": domain,
            "relevance_score": round(rng.uniform(0.6, 1.0), 2),
        })
    return citations


def _generate_mock_verification_results(
    n_claims: int,
    category: str,
    false_claim: Optional[str],
    rng: random.Random,
) -> list[dict]:
    """Generate synthetic verification results."""
    verdicts = []
    for i in range(n_claims):
        if false_claim and i == 0:
            # First claim is the false one -- always catch it
            verdicts.append({
                "claim": {"statement": false_claim},
                "verdict": "contradicted",
                "confidence": 0.95,
            })
        else:
            # Most claims pass; occasional unverifiable
            roll = rng.random()
            if roll < 0.80:
                verdict = "supported"
            elif roll < 0.95:
                verdict = "unverifiable"
            else:
                verdict = "contradicted"
            verdicts.append({
                "claim": {"statement": f"Claim {i + 1} about the topic"},
                "verdict": verdict,
                "confidence": round(rng.uniform(0.5, 1.0), 2),
            })
    return verdicts


def _generate_mock_trace_events(
    category: str,
    rng: random.Random,
) -> list[dict]:
    """Build synthetic trace events with latency + token data."""
    agents = ["planner", "researcher", "fact_checker", "synthesizer"]
    events = []
    for agent in agents:
        latency = rng.uniform(200, 2000)
        events.append({
            "agent": agent,
            "action": "start",
            "latency_ms": 0,
            "tokens_used": 0,
        })
        events.append({
            "agent": agent,
            "action": "llm_call",
            "latency_ms": latency,
            "tokens_used": rng.randint(300, 2000),
        })
        events.append({
            "agent": agent,
            "action": "complete",
            "latency_ms": latency + rng.uniform(50, 200),
            "tokens_used": 0,
        })
    return events


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------


def _load_test_queries() -> list[dict]:
    queries_path = Path(__file__).parent / "test_queries.json"
    if not queries_path.exists():
        logger.error("test_queries_not_found", path=str(queries_path))
        return []
    with open(queries_path) as f:
        data = json.load(f)
    return data.get("queries", [])


def run_mock_single_query(query_data: dict) -> EvaluationResult:
    """Evaluate a single query using deterministic mock data.

    Returns a fully-populated ``EvaluationResult`` as if the real pipeline
    had run, but without any external API calls.
    """
    query_id = query_data["id"]
    query = query_data["query"]
    category = query_data.get("category", "factual")
    must_mention = query_data.get("must_mention", [])
    must_not_claim = query_data.get("must_not_claim", [])
    false_claim = query_data.get("false_claim")

    rng = random.Random(_seed_from_id(query_id))

    # --- Generate synthetic artefacts ---
    n_citations = rng.randint(3, 8)
    n_claims = rng.randint(3, 8)
    n_sub_queries = max(query_data.get("expected_sub_queries_min", 2), rng.randint(2, 5))

    report = _generate_mock_report(query, must_mention, must_not_claim, false_claim, rng)
    citations = _generate_mock_citations(n_citations, rng)
    verifications = _generate_mock_verification_results(n_claims, category, false_claim, rng)
    trace_events = _generate_mock_trace_events(category, rng)

    total_latency_ms = sum(e.get("latency_ms", 0) for e in trace_events)

    # --- Compute metrics using the real metrics functions ---
    citation_metrics = calculate_citation_metrics(citations, report)
    fact_check_metrics = calculate_fact_check_metrics(verifications, false_claim)
    latency_metrics = calculate_latency_metrics(trace_events, total_latency_ms)

    # Quality scores from category profile + noise
    base = _QUALITY_PROFILES.get(category, (3, 3, 3, 3))
    relevance = min(5, max(1, base[0] + rng.choice([-1, 0, 0, 1])))
    accuracy = min(5, max(1, base[1] + rng.choice([-1, 0, 0, 1])))
    completeness = min(5, max(1, base[2] + rng.choice([-1, 0, 0, 1])))
    citation_quality = min(5, max(1, base[3] + rng.choice([-1, 0, 0, 1])))
    overall = relevance * 0.25 + accuracy * 0.35 + completeness * 0.20 + citation_quality * 0.20

    quality_scores = QualityScores(
        relevance=relevance,
        accuracy=accuracy,
        completeness=completeness,
        citation_quality=citation_quality,
        overall=round(overall, 2),
    )

    token_metrics = calculate_token_metrics(trace_events, quality_scores.overall)

    found, missing = check_must_mention(report, must_mention)
    violations = check_must_not_claim(report, must_not_claim)

    return EvaluationResult(
        query_id=query_id,
        query=query,
        category=category,
        success=True,
        citation_metrics=citation_metrics,
        fact_check_metrics=fact_check_metrics,
        latency_metrics=latency_metrics,
        token_metrics=token_metrics,
        quality_scores=quality_scores,
        expected_sub_queries_min=query_data.get("expected_sub_queries_min", 1),
        actual_sub_queries=n_sub_queries,
        expected_sources_min=query_data.get("expected_sources_min", 3),
        actual_sources=n_citations,
        must_mention_found=found,
        must_mention_missing=missing,
        must_not_claim_violations=violations,
    )


def run_mock_evaluation(
    category: Optional[str] = None,
    query_ids: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> tuple[list[EvaluationResult], AggregateMetrics]:
    """Run mock evaluation across query set and return metrics."""
    queries = _load_test_queries()

    if category:
        queries = [q for q in queries if q.get("category") == category]
    if query_ids:
        queries = [q for q in queries if q.get("id") in query_ids]
    if limit:
        queries = queries[:limit]

    results = [run_mock_single_query(q) for q in queries]
    aggregate = calculate_aggregate_metrics(results)
    return results, aggregate


def save_mock_results(
    results: list[EvaluationResult],
    aggregate: AggregateMetrics,
    output_file: Optional[str] = None,
) -> Path:
    """Save mock eval results JSON to data/eval_results/."""
    output_dir = Path("data/eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = output_file or f"mock_eval_{timestamp}.json"
    output_path = output_dir / filename

    output_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "mode": "mock",
        "aggregate_metrics": aggregate.model_dump(),
        "results": [r.model_dump() for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("mock_results_saved", path=str(output_path))
    return output_path


def compare_results(
    current: AggregateMetrics,
    previous_path: str,
) -> dict:
    """Compare current aggregate metrics with a previous run.

    Returns a dict of metric-name -> {before, after, delta, improved}.
    """
    with open(previous_path) as f:
        prev_data = json.load(f)

    prev_agg = AggregateMetrics(**prev_data["aggregate_metrics"])

    comparisons: dict[str, dict] = {}
    compare_fields = [
        ("avg_overall_quality", True),   # higher is better
        ("avg_citation_accuracy", True),
        ("avg_fact_check_pass_rate", True),
        ("latency_avg_ms", False),       # lower is better
        ("total_tokens", False),
        ("false_premise_recall", True),
    ]

    for field, higher_is_better in compare_fields:
        before = getattr(prev_agg, field, 0)
        after = getattr(current, field, 0)
        delta = after - before
        improved = delta > 0 if higher_is_better else delta < 0
        comparisons[field] = {
            "before": round(before, 4),
            "after": round(after, 4),
            "delta": round(delta, 4),
            "improved": improved,
        }

    return comparisons


def print_summary(
    results: list[EvaluationResult],
    aggregate: AggregateMetrics,
    comparisons: Optional[dict] = None,
) -> None:
    """Print a human-readable evaluation summary."""
    print("\n" + "=" * 60)
    print("MARSA MOCK EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nQueries Run: {aggregate.total_queries}")
    print(f"  Successful: {aggregate.successful_queries}")
    print(f"  Failed: {aggregate.failed_queries}")

    print("\nQuality Scores (average):")
    print(f"  Relevance:        {aggregate.avg_relevance:.2f}/5")
    print(f"  Accuracy:         {aggregate.avg_accuracy:.2f}/5")
    print(f"  Completeness:     {aggregate.avg_completeness:.2f}/5")
    print(f"  Citation Quality: {aggregate.avg_citation_quality:.2f}/5")
    print(f"  OVERALL:          {aggregate.avg_overall_quality:.2f}/5")

    print("\nPerformance Metrics:")
    print(f"  Latency P50:    {aggregate.latency_p50_ms:.0f}ms")
    print(f"  Latency Avg:    {aggregate.latency_avg_ms:.0f}ms")
    print(f"  Total Tokens:   {aggregate.total_tokens:,}")
    print(f"  LLM Calls:      {aggregate.total_llm_calls}")

    print("\nSource Metrics:")
    print(f"  Avg Citation Accuracy:   {aggregate.avg_citation_accuracy:.1%}")
    print(f"  Avg Source Diversity:     {aggregate.avg_source_diversity:.1f} domains")
    print(f"  Fact-Check Pass Rate:     {aggregate.avg_fact_check_pass_rate:.1%}")
    if aggregate.false_premise_recall > 0:
        print(f"  False Premise Recall:     {aggregate.false_premise_recall:.1%}")

    print("\nBy Category:")
    for category, metrics in sorted(aggregate.metrics_by_category.items()):
        print(
            f"  {category}: {metrics['count']} queries, "
            f"quality={metrics['avg_quality']:.2f}"
        )

    if comparisons:
        print("\n--- Comparison with Previous Run ---")
        for field, data in comparisons.items():
            arrow = "+" if data["improved"] else "-" if data["delta"] != 0 else "="
            sign = "+" if data["delta"] >= 0 else ""
            print(
                f"  {field:30s}  {data['before']:>8.4f} -> {data['after']:>8.4f}  "
                f"({sign}{data['delta']:.4f}) [{arrow}]"
            )

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Run MARSA mock evaluation")
    parser.add_argument("--category", help="Run only this category")
    parser.add_argument("--limit", type=int, help="Max queries to run")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--compare", help="Path to previous result JSON for comparison")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    return parser.parse_args()


def main():
    args = parse_args()

    results, aggregate = run_mock_evaluation(
        category=args.category,
        limit=args.limit,
    )

    if not results:
        print("No results.")
        return

    output_path = save_mock_results(results, aggregate, args.output)

    comparisons = None
    if args.compare:
        comparisons = compare_results(aggregate, args.compare)

    if not args.quiet:
        print_summary(results, aggregate, comparisons)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
