#!/usr/bin/env python3
"""Evaluation runner for the MARSA research pipeline.

This script runs the evaluation query set through the full pipeline,
measures various metrics, and produces a comprehensive evaluation report.

Usage:
    python -m eval.run_eval                       # Run all queries
    python -m eval.run_eval --category factual    # Run specific category
    python -m eval.run_eval --query-id factual_01 # Run single query
    python -m eval.run_eval --output results.json # Custom output file
    python -m eval.run_eval --dry-run             # Preview without running
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from anthropic import Anthropic  # noqa: E402

from config import Config  # noqa: E402
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
)
from graph.state import create_initial_state  # noqa: E402
from graph.workflow import create_workflow  # noqa: E402

logger = structlog.get_logger(__name__)

# Default delay between queries to respect rate limits (in seconds)
DEFAULT_QUERY_DELAY = 5.0


class EvaluationRunner:
    """Runner for the MARSA evaluation suite."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        output_dir: Optional[Path] = None,
        query_delay: float = DEFAULT_QUERY_DELAY,
    ):
        """Initialize the evaluation runner.
        
        Args:
            config: Application configuration. If None, loads from environment.
            output_dir: Directory for output files. Defaults to data/eval_results.
            query_delay: Delay between queries in seconds.
        """
        self.config = config or Config(validate=False)
        self.output_dir = output_dir or Path("data/eval_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.query_delay = query_delay
        
        # Load test queries
        self.queries = self._load_test_queries()
        
        # Initialize Claude client for quality scoring
        self.anthropic = None
        if self.config.anthropic_api_key:
            self.anthropic = Anthropic(api_key=self.config.anthropic_api_key)
        
        logger.info(
            "evaluation_runner_initialized",
            query_count=len(self.queries),
            output_dir=str(self.output_dir),
        )
    
    def _load_test_queries(self) -> list[dict]:
        """Load test queries from JSON file.
        
        Returns:
            List of query dictionaries.
        """
        queries_path = Path(__file__).parent / "test_queries.json"
        
        if not queries_path.exists():
            logger.error("test_queries_not_found", path=str(queries_path))
            return []
        
        with open(queries_path) as f:
            data = json.load(f)
        
        return data.get("queries", [])
    
    async def run_single_query(
        self,
        query_data: dict,
        workflow,
    ) -> EvaluationResult:
        """Run a single query through the pipeline and evaluate it.
        
        Args:
            query_data: Query definition from test_queries.json.
            workflow: Compiled LangGraph workflow.
            
        Returns:
            EvaluationResult with all metrics.
        """
        query_id = query_data.get("id", "unknown")
        query_text = query_data.get("query", "")
        category = query_data.get("category", "unknown")
        
        logger.info("running_query", query_id=query_id, category=category)
        
        result = EvaluationResult(
            query_id=query_id,
            query=query_text,
            category=category,
            expected_sub_queries_min=query_data.get("expected_sub_queries_min", 1),
            expected_sources_min=query_data.get("expected_sources_min", 3),
        )
        
        start_time = time.time()
        
        try:
            # Create initial state and run workflow
            initial_state = create_initial_state(query_text)
            config = {"configurable": {"thread_id": f"eval_{query_id}_{int(time.time())}"}}
            
            final_state = await workflow.ainvoke(initial_state, config)
            
            end_time = time.time()
            total_latency_ms = (end_time - start_time) * 1000
            
            # Extract data from final state
            report = final_state.get("report", "")
            citations = final_state.get("citations", [])
            verification_results = final_state.get("verification_results", [])
            trace_events = final_state.get("agent_trace", [])
            plan = final_state.get("plan")
            
            # Convert citations and verification results to dicts if needed
            citations_dicts = [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in citations
            ]
            verification_dicts = [
                v.model_dump() if hasattr(v, "model_dump") else v
                for v in verification_results
            ]
            trace_dicts = [
                e.model_dump() if hasattr(e, "model_dump") else e
                for e in trace_events
            ]
            
            # Calculate sub-queries count
            actual_sub_queries = len(plan.sub_queries) if plan else 0
            result.actual_sub_queries = actual_sub_queries
            result.actual_sources = len(citations_dicts)
            
            # Calculate metrics
            result.citation_metrics = calculate_citation_metrics(citations_dicts, report)
            
            false_premise_claim = query_data.get("false_claim")
            result.fact_check_metrics = calculate_fact_check_metrics(
                verification_dicts,
                false_premise_claim,
            )
            
            result.latency_metrics = calculate_latency_metrics(trace_dicts, total_latency_ms)
            
            # Check must_mention and must_not_claim
            must_mention = query_data.get("must_mention", [])
            must_not_claim = query_data.get("must_not_claim", [])
            
            found, missing = check_must_mention(report, must_mention)
            result.must_mention_found = found
            result.must_mention_missing = missing
            
            violations = check_must_not_claim(report, must_not_claim)
            result.must_not_claim_violations = violations
            
            # Score quality with Claude
            if self.anthropic and report:
                quality_scores = await self._score_report_quality(
                    query_text,
                    report,
                    citations_dicts,
                )
                result.quality_scores = quality_scores
            
            # Calculate token metrics (after quality score is known)
            result.token_metrics = calculate_token_metrics(
                trace_dicts,
                result.quality_scores.overall,
            )
            
            result.success = True
            logger.info(
                "query_completed",
                query_id=query_id,
                latency_ms=total_latency_ms,
                quality=result.quality_scores.overall,
            )
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error("query_failed", query_id=query_id, error=str(e))
        
        return result
    
    async def _score_report_quality(
        self,
        query: str,
        report: str,
        citations: list[dict],
    ) -> QualityScores:
        """Use Claude to score report quality on multiple dimensions.
        
        Args:
            query: Original query.
            report: Generated report text.
            citations: List of citation dictionaries.
            
        Returns:
            QualityScores with 1-5 ratings.
        """
        if not self.anthropic:
            return QualityScores()
        
        prompt = f"""You are evaluating a research report. Score each dimension from 1-5.

QUERY: {query}

REPORT:
{report[:4000]}

CITATIONS COUNT: {len(citations)}

Score the report on these dimensions (1=poor, 3=acceptable, 5=excellent):

1. RELEVANCE: How well does the report address the original query?
2. ACCURACY: Are the claims factually correct and well-supported?
3. COMPLETENESS: Does the report cover all important aspects?
4. CITATION_QUALITY: Are sources properly cited and high-quality?

Respond in this exact JSON format:
{{
    "relevance": <1-5>,
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "citation_quality": <1-5>,
    "reasoning": "<brief explanation>"
}}"""

        try:
            response = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            
            response_text = response.content[0].text
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                scores_data = json.loads(json_match.group())
                
                relevance = int(scores_data.get("relevance", 3))
                accuracy = int(scores_data.get("accuracy", 3))
                completeness = int(scores_data.get("completeness", 3))
                citation_quality = int(scores_data.get("citation_quality", 3))
                
                # Calculate weighted overall score
                overall = (relevance * 0.25 + accuracy * 0.35 + 
                          completeness * 0.2 + citation_quality * 0.2)
                
                return QualityScores(
                    relevance=relevance,
                    accuracy=accuracy,
                    completeness=completeness,
                    citation_quality=citation_quality,
                    overall=overall,
                )
        except Exception as e:
            logger.warning("quality_scoring_failed", error=str(e))
        
        return QualityScores(relevance=3, accuracy=3, completeness=3, citation_quality=3, overall=3.0)
    
    async def run_evaluation(
        self,
        category: Optional[str] = None,
        query_ids: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> tuple[list[EvaluationResult], AggregateMetrics]:
        """Run the full evaluation suite.
        
        Args:
            category: Optional category to filter queries.
            query_ids: Optional list of specific query IDs to run.
            limit: Optional limit on number of queries to run.
            
        Returns:
            Tuple of (individual results, aggregate metrics).
        """
        # Filter queries
        queries_to_run = self.queries
        
        if category:
            queries_to_run = [q for q in queries_to_run if q.get("category") == category]
        
        if query_ids:
            queries_to_run = [q for q in queries_to_run if q.get("id") in query_ids]
        
        if limit:
            queries_to_run = queries_to_run[:limit]
        
        if not queries_to_run:
            logger.warning("no_queries_to_run")
            return [], AggregateMetrics()
        
        logger.info(
            "starting_evaluation",
            query_count=len(queries_to_run),
            category=category,
        )
        
        # Create workflow
        workflow = create_workflow(
            enable_hitl=False,
            enable_parallel=True,
            use_memory_checkpointer=True,
        )
        
        results = []
        
        for i, query_data in enumerate(queries_to_run):
            logger.info(
                "evaluating_query",
                progress=f"{i + 1}/{len(queries_to_run)}",
                query_id=query_data.get("id"),
            )
            
            result = await self.run_single_query(query_data, workflow)
            results.append(result)
            
            # Delay between queries to respect rate limits
            if i < len(queries_to_run) - 1:
                logger.debug("rate_limit_delay", seconds=self.query_delay)
                await asyncio.sleep(self.query_delay)
        
        # Calculate aggregate metrics
        aggregate = calculate_aggregate_metrics(results)
        
        logger.info(
            "evaluation_complete",
            total=aggregate.total_queries,
            successful=aggregate.successful_queries,
            avg_quality=aggregate.avg_overall_quality,
        )
        
        return results, aggregate
    
    def save_results(
        self,
        results: list[EvaluationResult],
        aggregate: AggregateMetrics,
        output_file: Optional[str] = None,
    ) -> Path:
        """Save evaluation results to JSON file.
        
        Args:
            results: Individual query results.
            aggregate: Aggregate metrics.
            output_file: Optional output filename.
            
        Returns:
            Path to the saved file.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = output_file or f"eval_results_{timestamp}.json"
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "aggregate_metrics": aggregate.model_dump(),
            "results": [r.model_dump() for r in results],
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info("results_saved", path=str(output_path))
        return output_path
    
    def identify_worst_queries(
        self,
        results: list[EvaluationResult],
        n: int = 3,
    ) -> list[tuple[EvaluationResult, str]]:
        """Identify the worst-performing queries with reasons.
        
        Args:
            results: Evaluation results.
            n: Number of worst queries to return.
            
        Returns:
            List of (result, reason) tuples.
        """
        scored_results = []
        
        for result in results:
            if not result.success:
                scored_results.append((result, 0.0, f"Failed: {result.error_message}"))
                continue
            
            # Calculate a composite failure score
            issues = []
            score = result.quality_scores.overall
            
            if result.must_mention_missing:
                score -= 0.5
                issues.append(f"Missing required terms: {result.must_mention_missing}")
            
            if result.must_not_claim_violations:
                score -= 1.0
                issues.append(f"Contains forbidden claims: {result.must_not_claim_violations}")
            
            if result.actual_sub_queries < result.expected_sub_queries_min:
                score -= 0.3
                issues.append(f"Insufficient sub-queries: {result.actual_sub_queries} < {result.expected_sub_queries_min}")
            
            if result.actual_sources < result.expected_sources_min:
                score -= 0.3
                issues.append(f"Insufficient sources: {result.actual_sources} < {result.expected_sources_min}")
            
            if result.citation_metrics.accuracy < 0.8:
                score -= 0.2
                issues.append(f"Low citation accuracy: {result.citation_metrics.accuracy:.2f}")
            
            reason = "; ".join(issues) if issues else "Low overall quality score"
            scored_results.append((result, score, reason))
        
        # Sort by score (lowest first)
        scored_results.sort(key=lambda x: x[1])
        
        return [(r, reason) for r, _, reason in scored_results[:n]]
    
    def print_summary(
        self,
        results: list[EvaluationResult],
        aggregate: AggregateMetrics,
    ):
        """Print a summary of evaluation results to console.
        
        Args:
            results: Individual query results.
            aggregate: Aggregate metrics.
        """
        print("\n" + "=" * 60)
        print("MARSA EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\nQueries Run: {aggregate.total_queries}")
        print(f"  Successful: {aggregate.successful_queries}")
        print(f"  Failed: {aggregate.failed_queries}")
        
        print(f"\nQuality Scores (average):")
        print(f"  Relevance: {aggregate.avg_relevance:.2f}/5")
        print(f"  Accuracy: {aggregate.avg_accuracy:.2f}/5")
        print(f"  Completeness: {aggregate.avg_completeness:.2f}/5")
        print(f"  Citation Quality: {aggregate.avg_citation_quality:.2f}/5")
        print(f"  OVERALL: {aggregate.avg_overall_quality:.2f}/5")
        
        print(f"\nPerformance Metrics:")
        print(f"  Latency P50: {aggregate.latency_p50_ms:.0f}ms")
        print(f"  Latency P95: {aggregate.latency_p95_ms:.0f}ms")
        print(f"  Avg Latency: {aggregate.latency_avg_ms:.0f}ms")
        print(f"  Total Tokens: {aggregate.total_tokens:,}")
        print(f"  LLM Calls: {aggregate.total_llm_calls}")
        
        print(f"\nSource Metrics:")
        print(f"  Avg Citation Accuracy: {aggregate.avg_citation_accuracy:.1%}")
        print(f"  Avg Source Diversity: {aggregate.avg_source_diversity:.1f} domains")
        print(f"  Fact-Check Pass Rate: {aggregate.avg_fact_check_pass_rate:.1%}")
        
        if aggregate.false_premise_recall > 0:
            print(f"  False Premise Recall: {aggregate.false_premise_recall:.1%}")
        
        print(f"\nBy Category:")
        for category, metrics in aggregate.metrics_by_category.items():
            print(f"  {category}: {metrics['count']} queries, "
                  f"quality={metrics['avg_quality']:.2f}, "
                  f"latency={metrics['avg_latency_ms']:.0f}ms")
        
        # Show worst queries
        worst = self.identify_worst_queries(results, n=3)
        if worst:
            print(f"\nWorst Performing Queries:")
            for i, (result, reason) in enumerate(worst, 1):
                print(f"  {i}. [{result.query_id}] {result.query[:50]}...")
                print(f"     Reason: {reason}")
        
        print("\n" + "=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MARSA evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--category",
        choices=["factual", "comparison", "exploratory", "false_premise", "doc_context"],
        help="Run only queries of this category",
    )
    
    parser.add_argument(
        "--query-id",
        dest="query_ids",
        action="append",
        help="Run specific query ID (can be repeated)",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of queries to run",
    )
    
    parser.add_argument(
        "--output",
        help="Output JSON filename",
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_QUERY_DELAY,
        help=f"Delay between queries in seconds (default: {DEFAULT_QUERY_DELAY})",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview queries without running them",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output except errors",
    )
    
    return parser.parse_args()


async def main():
    """Main entry point for the evaluation runner."""
    args = parse_args()
    
    # Configure logging
    if not args.quiet:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(20),
        )
    
    # Initialize runner
    runner = EvaluationRunner(query_delay=args.delay)
    
    # Handle dry run
    if args.dry_run:
        queries = runner.queries
        
        if args.category:
            queries = [q for q in queries if q.get("category") == args.category]
        if args.query_ids:
            queries = [q for q in queries if q.get("id") in args.query_ids]
        if args.limit:
            queries = queries[:args.limit]
        
        print(f"\nDry run: Would run {len(queries)} queries:")
        for q in queries:
            print(f"  [{q.get('id')}] ({q.get('category')}) {q.get('query')[:60]}...")
        return
    
    # Run evaluation
    results, aggregate = await runner.run_evaluation(
        category=args.category,
        query_ids=args.query_ids,
        limit=args.limit,
    )
    
    if not results:
        print("No results to save.")
        return
    
    # Save results
    output_path = runner.save_results(results, aggregate, args.output)
    
    # Print summary
    if not args.quiet:
        runner.print_summary(results, aggregate)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
