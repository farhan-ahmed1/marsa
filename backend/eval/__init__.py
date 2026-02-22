"""MARSA evaluation framework.

This module provides tools for evaluating the quality and performance
of the multi-agent research pipeline.

Main components:
- EvaluationRunner: Runs test queries through the pipeline
- metrics: Quality and performance metrics calculation
- test_queries.json: Evaluation query set
"""

# Note: Imports are deferred to avoid circular dependencies
# Use direct imports in your code:
#   from eval.metrics import ...
#   from eval.run_eval import EvaluationRunner

__all__ = [
    "AggregateMetrics",
    "CitationMetrics",
    "EvaluationResult",
    "EvaluationRunner",
    "FactCheckMetrics",
    "LatencyMetrics",
    "QualityScores",
    "TokenMetrics",
    "calculate_aggregate_metrics",
    "calculate_citation_metrics",
    "calculate_fact_check_metrics",
    "calculate_latency_metrics",
    "calculate_token_metrics",
    "check_must_mention",
    "check_must_not_claim",
    "latency_p50_p95",
]

