"""Agent modules for MARSA multi-agent research system.

This package contains the specialized AI agents that work together
in the LangGraph workflow to produce research reports.

Agents:
- Planner: Analyzes queries and creates research plans
- Researcher: Executes search queries and extracts claims
- Fact-Checker: Verifies claims with independent searches
- Synthesizer: Produces the final research report
- Source Scorer: Evaluates source quality for weighting evidence
"""

from agents.fact_checker import (
    BAD_CLAIM_THRESHOLD,
    MAX_ITERATIONS,
    fact_check_node,
    generate_verify_query,
    should_loop_back,
    verify_claim,
)
from agents.planner import create_query_plan, planner_node
from agents.researcher import extract_claims, research_node
from agents.source_scorer import score_source

__all__ = [
    # Planner
    "create_query_plan",
    "planner_node",
    # Researcher
    "research_node",
    "extract_claims",
    # Fact-Checker
    "fact_check_node",
    "verify_claim",
    "generate_verify_query",
    "should_loop_back",
    "BAD_CLAIM_THRESHOLD",
    "MAX_ITERATIONS",
    # Source Scorer
    "score_source",
]