"""Planner Agent for MARSA.

The Planner analyzes incoming queries and produces a QueryPlan that guides
the research process, including query decomposition, search strategy, and
parallelization decisions.
"""

import json
from typing import Optional

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import config
from graph.state import (
    AgentState,
    ComplexityLevel,
    QueryPlan,
    QueryType,
    SearchStrategy,
)

logger = structlog.get_logger(__name__)


PLANNER_SYSTEM_PROMPT = """\
You are a research query planning agent. Your job is to analyze incoming research queries \
and produce a structured plan that guides the research process.

Analyze the query and output a JSON plan with these fields:
- query_type: One of "factual", "comparison", "exploratory", "opinion", "howto", "definition"
- sub_queries: List of decomposed research questions (1-6 items)
- parallel: Boolean - true if sub-queries can be researched independently
- needs_fact_check: Boolean - false for simple definitions/factual queries, true for claims
- search_strategy: One of "web_only", "docs_only", "hybrid"
- estimated_complexity: One of "low", "medium", "high"
- reasoning: Brief explanation of your planning decisions

## Query Type Guidelines

- "factual": Questions with clear, verifiable answers (e.g., "What year was Python released?")
- "definition": Requests for explanations or definitions (e.g., "What is gRPC?")
- "comparison": Comparing multiple items (e.g., "Compare React vs Vue")
- "exploratory": Open-ended research (e.g., "What are the latest trends in AI?")
- "opinion": Questions seeking analysis or opinions (e.g., "Is Rust worth learning?")
- "howto": Step-by-step guides (e.g., "How to set up Kubernetes?")

## Sub-Query Decomposition Examples

Query: "Compare Rust vs Go for backend development"
sub_queries:
  - "Rust strengths for backend and distributed systems"
  - "Go strengths for backend and distributed systems"
  - "Rust vs Go performance benchmarks"
  - "Rust vs Go developer experience and learning curve"
  - "Rust vs Go ecosystem and library support"

Query: "What is the CAP theorem?"
sub_queries:
  - "CAP theorem definition and explanation"
(Simple definition - single sub-query, needs_fact_check: false)

Query: "What are the best practices for microservices?"
sub_queries:
  - "Microservices architecture patterns and design principles"
  - "Microservices communication patterns (sync vs async)"
  - "Microservices data management and consistency"
  - "Microservices deployment and orchestration"

Query: "Compare React, Vue, and Svelte for building SPAs"
sub_queries:
  - "React strengths and ecosystem for SPAs"
  - "Vue strengths and ecosystem for SPAs"
  - "Svelte strengths and ecosystem for SPAs"
  - "React vs Vue vs Svelte performance comparison"
  - "React vs Vue vs Svelte developer experience"

## Search Strategy Guidelines

- "web_only": For current events, recent developments, or general knowledge
- "docs_only": For internal/private documentation queries
- "hybrid": Default for most queries - combines web search with document store

## Complexity Guidelines

- "low": Simple factual or definition queries (1-2 sub-queries)
- "medium": Standard research queries (3-4 sub-queries)
- "high": Complex comparisons or exploratory research (5+ sub-queries)

## Output Format

Output ONLY valid JSON matching this schema (no markdown, no extra text):
{
  "query_type": "...",
  "sub_queries": ["...", "..."],
  "parallel": true/false,
  "needs_fact_check": true/false,
  "search_strategy": "...",
  "estimated_complexity": "...",
  "reasoning": "..."
}
"""


def _create_llm(model: str = "claude-sonnet-4-20250514") -> ChatAnthropic:
    """Create a ChatAnthropic instance for the planner.
    
    Args:
        model: The Claude model to use.
        
    Returns:
        Configured ChatAnthropic instance.
    """
    return ChatAnthropic(
        model=model,
        api_key=config.anthropic_api_key,
        temperature=0,
        max_tokens=1024,
    )


def _parse_query_plan(response_text: str) -> QueryPlan:
    """Parse LLM response into a QueryPlan model.
    
    Args:
        response_text: Raw JSON response from the LLM.
        
    Returns:
        Validated QueryPlan model.
        
    Raises:
        ValueError: If response cannot be parsed into a valid QueryPlan.
    """
    # Clean potential markdown code fences
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}") from e
    
    # Map string values to enums
    query_type_map = {
        "factual": QueryType.FACTUAL,
        "comparison": QueryType.COMPARISON,
        "exploratory": QueryType.EXPLORATORY,
        "opinion": QueryType.OPINION,
        "howto": QueryType.HOWTO,
        "definition": QueryType.DEFINITION,
    }
    
    strategy_map = {
        "web_only": SearchStrategy.WEB_ONLY,
        "docs_only": SearchStrategy.DOCS_ONLY,
        "hybrid": SearchStrategy.HYBRID,
    }
    
    complexity_map = {
        "low": ComplexityLevel.LOW,
        "medium": ComplexityLevel.MEDIUM,
        "high": ComplexityLevel.HIGH,
    }
    
    raw_type = data.get("query_type", "factual").lower()
    raw_strategy = data.get("search_strategy", "hybrid").lower()
    raw_complexity = data.get("estimated_complexity", "medium").lower()
    
    return QueryPlan(
        query_type=query_type_map.get(raw_type, QueryType.FACTUAL),
        sub_queries=data.get("sub_queries", []),
        parallel=data.get("parallel", True),
        needs_fact_check=data.get("needs_fact_check", True),
        search_strategy=strategy_map.get(raw_strategy, SearchStrategy.HYBRID),
        estimated_complexity=complexity_map.get(raw_complexity, ComplexityLevel.MEDIUM),
        reasoning=data.get("reasoning", ""),
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ValueError,)),
    reraise=True,
)
async def create_query_plan(
    query: str,
    llm: Optional[ChatAnthropic] = None,
    memory_context: str = "",
) -> QueryPlan:
    """Analyze a query and create a research plan.
    
    Args:
        query: The user's research query.
        llm: Optional ChatAnthropic instance (created if not provided).
        memory_context: Optional prior research context from cross-session memory.
        
    Returns:
        QueryPlan with decomposed sub-queries and research strategy.
        
    Raises:
        ValueError: If the LLM response cannot be parsed.
    """
    if llm is None:
        llm = _create_llm()
    
    # Build the user message, appending memory context if available
    user_content = f"Create a research plan for this query:\n\n{query}"
    if memory_context:
        user_content += (
            "\n\n"
            + memory_context
            + "\n\nNote: use the prior context only to enrich sub-queries, "
              "not to skip current research."
        )
    
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    
    logger.info("creating_query_plan", query=query[:100], has_memory=bool(memory_context))
    
    response = await llm.ainvoke(messages)
    response_text = response.content
    
    if isinstance(response_text, list):
        # Handle potential list response (multi-part content)
        response_text = "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in response_text
        )
    
    plan = _parse_query_plan(str(response_text))
    
    logger.info(
        "query_plan_created",
        query_type=plan.query_type.value,
        sub_queries_count=len(plan.sub_queries),
        parallel=plan.parallel,
        needs_fact_check=plan.needs_fact_check,
    )
    
    return plan


async def planner_node(state: AgentState) -> dict:
    """LangGraph node function for the Planner agent.
    
    This function is designed to be used as a node in a LangGraph StateGraph.
    It reads the query from state and returns updates to the state.
    
    Args:
        state: Current agent state containing the query.
        
    Returns:
        Dict with state updates (plan, sub_queries, status).
    """
    query = state.get("query", "")
    memory_context = state.get("memory_context", "")
    
    if not query:
        logger.warning("planner_node_empty_query")
        return {
            "plan": None,
            "sub_queries": [],
            "status": "failed",
            "errors": ["Empty query provided to planner"],
        }
    
    try:
        plan = await create_query_plan(query, memory_context=memory_context)
        return {
            "plan": plan,
            "sub_queries": plan.sub_queries,
            "status": "researching",
        }
    except Exception as e:
        logger.exception("planner_node_error", error=str(e))
        return {
            "plan": None,
            "sub_queries": [],
            "status": "failed",
            "errors": [f"Planner error: {str(e)}"],
        }
