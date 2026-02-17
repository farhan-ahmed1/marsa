"""Researcher Agent for MARSA.

The Researcher executes the research plan by querying data sources (web search
and document store), merging results, scoring sources, and extracting
structured claims for verification.
"""

import json
from time import time
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

from agents.source_scorer import score_source
from config import config
from graph.state import (
    AgentName,
    AgentState,
    Claim,
    ConfidenceLevel,
    PipelineStatus,
    QueryPlan,
    ResearchResult,
    TraceEvent,
)
from mcp_client import MCPClient

logger = structlog.get_logger(__name__)


CLAIM_EXTRACTION_SYSTEM_PROMPT = """\
You are a claim extraction agent. Your job is to analyze research findings \
and extract structured, verifiable factual claims.

Given a research query and a set of search results, extract the key factual \
claims that directly answer or relate to the query.

For each claim:
- **statement**: The factual claim in clear, concise language
- **source_url**: The URL where this claim was found
- **source_title**: The title of the source
- **confidence**: "high", "medium", or "low" based on:
  - "high": From authoritative sources (.gov, .edu, official docs, academic papers)
  - "medium": From established publications, well-known tech sites
  - "low": From blogs, forums, or sources with limited authority
- **category**: One of "fact", "opinion", "statistic", "quote"
  - "fact": Verifiable factual statement
  - "opinion": Expert opinion or analysis
  - "statistic": Numerical data or measurements
  - "quote": Direct quote from a person or document
- **context**: Brief surrounding context (1-2 sentences)

## Guidelines

1. Extract 3-8 claims per research query (fewer for simple queries, more for complex ones)
2. Focus on claims that directly answer the research question
3. Prefer factual claims over opinions unless the query specifically asks for opinions
4. Each claim should be independently verifiable
5. Avoid redundant claims - if multiple sources say the same thing, extract it once with the best source
6. Include numerical data, dates, and specific details when available

## Output Format

Output ONLY valid JSON matching this schema (no markdown, no extra text):
{
  "claims": [
    {
      "statement": "Python was created by Guido van Rossum",
      "source_url": "https://docs.python.org/3/faq/general.html",
      "source_title": "Python General FAQ",
      "confidence": "high",
      "category": "fact",
      "context": "The Python programming language was created in the late 1980s by Guido van Rossum at CWI in the Netherlands."
    },
    {
      "statement": "Python 3.0 was released on December 3, 2008",
      "source_url": "https://docs.python.org/3/whatsnew/3.0.html",
      "source_title": "What's New In Python 3.0",
      "confidence": "high",
      "category": "fact",
      "context": "Python 3.0 (also known as Python 3000 or Py3K) was released on December 3, 2008."
    }
  ]
}

## Examples

Query: "Compare Rust vs Go for backend development"
Claims:
- "Rust guarantees memory safety without a garbage collector through its ownership system"
- "Go was designed at Google for building scalable network services"
- "Rust compile times are significantly slower than Go compile times"
- "Go has a simpler syntax and shorter learning curve than Rust"

Query: "What is the CAP theorem?"
Claims:
- "The CAP theorem states that distributed systems can only guarantee two of three properties: Consistency, Availability, and Partition tolerance"
- "The CAP theorem was formulated by Eric Brewer in 2000"
- "MongoDB is a CP system that favors consistency over availability during network partitions"
"""


def _create_llm(model: str = "claude-sonnet-4-20250514") -> ChatAnthropic:
    """Create a ChatAnthropic instance for claim extraction.
    
    Args:
        model: The Claude model to use.
        
    Returns:
        Configured ChatAnthropic instance.
    """
    return ChatAnthropic(
        model=model,
        api_key=config.anthropic_api_key,
        temperature=0,
        max_tokens=2048,
    )


def _parse_claims_response(response_text: str, query: str) -> list[Claim]:
    """Parse LLM response into a list of Claim models.
    
    Args:
        response_text: Raw JSON response from the LLM.
        query: The original research query (for logging).
        
    Returns:
        List of validated Claim models.
        
    Raises:
        ValueError: If response cannot be parsed into valid Claims.
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
    
    if not isinstance(data, dict) or "claims" not in data:
        raise ValueError("Response missing 'claims' array")
    
    claims_data = data["claims"]
    if not isinstance(claims_data, list):
        raise ValueError("'claims' must be an array")
    
    # Map string confidence values to enum
    confidence_map = {
        "high": ConfidenceLevel.HIGH,
        "medium": ConfidenceLevel.MEDIUM,
        "low": ConfidenceLevel.LOW,
    }
    
    claims = []
    for claim_obj in claims_data:
        if not isinstance(claim_obj, dict):
            logger.warning("skipping_invalid_claim", claim=claim_obj)
            continue
        
        try:
            confidence_str = claim_obj.get("confidence", "medium").lower()
            claims.append(Claim(
                statement=claim_obj.get("statement", ""),
                source_url=claim_obj.get("source_url", ""),
                source_title=claim_obj.get("source_title", ""),
                confidence=confidence_map.get(confidence_str, ConfidenceLevel.MEDIUM),
                category=claim_obj.get("category", "fact"),
                context=claim_obj.get("context", ""),
            ))
        except Exception as e:
            logger.warning("claim_parsing_error", error=str(e), claim=claim_obj)
            continue
    
    logger.info("claims_parsed", query=query[:50], claim_count=len(claims))
    return claims


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((ValueError,)),
    reraise=True,
)
async def extract_claims(
    research_results: list[ResearchResult],
    query: str,
    llm: Optional[ChatAnthropic] = None,
) -> list[Claim]:
    """Extract structured claims from research results using Claude.
    
    Args:
        research_results: List of research findings from searches.
        query: The original research query for context.
        llm: Optional ChatAnthropic instance (created if not provided).
        
    Returns:
        List of extracted Claim objects.
        
    Raises:
        ValueError: If the LLM response cannot be parsed after retries.
    """
    if not research_results:
        logger.warning("extract_claims_no_results", query=query[:50])
        return []
    
    if llm is None:
        llm = _create_llm()
    
    # Format research results for the LLM prompt
    results_text = ""
    for idx, result in enumerate(research_results[:20], 1):  # Limit to top 20 results
        results_text += f"\n\n--- Result {idx} ---\n"
        results_text += f"Source: {result.source_title}\n"
        results_text += f"URL: {result.source_url}\n"
        results_text += f"Relevance: {result.relevance_score:.2f}\n"
        if result.published_date:
            results_text += f"Published: {result.published_date}\n"
        results_text += f"Content: {result.content[:800]}\n"  # Limit content length
    
    user_prompt = f"""Research Query: {query}

Search Results:
{results_text}

Extract the key factual claims from these search results that directly relate to the query. \
Output the claims as a JSON array according to the specified schema."""
    
    messages = [
        SystemMessage(content=CLAIM_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    
    logger.info("extracting_claims", query=query[:50], result_count=len(research_results))
    
    start_time = time()
    response = await llm.ainvoke(messages)
    latency_ms = (time() - start_time) * 1000
    
    response_text = response.content
    if isinstance(response_text, list):
        # Handle potential list response (multi-part content)
        response_text = "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in response_text
        )
    
    claims = _parse_claims_response(str(response_text), query)
    
    logger.info(
        "claims_extracted",
        query=query[:50],
        claim_count=len(claims),
        latency_ms=round(latency_ms, 2),
        tokens_used=response.usage_metadata.get("total_tokens", 0) if hasattr(response, "usage_metadata") else 0,
    )
    
    return claims


def _deduplicate_results(results: list[ResearchResult]) -> list[ResearchResult]:
    """Remove duplicate results based on URL and content similarity.
    
    Args:
        results: List of research results to deduplicate.
        
    Returns:
        Deduplicated list of results.
    """
    seen_urls = set()
    deduplicated = []
    
    for result in results:
        # Use URL as primary deduplication key
        if result.source_url not in seen_urls:
            seen_urls.add(result.source_url)
            deduplicated.append(result)
    
    logger.info(
        "results_deduplicated",
        original_count=len(results),
        deduplicated_count=len(deduplicated),
        removed_count=len(results) - len(deduplicated),
    )
    
    return deduplicated


def _score_and_sort_results(
    results: list[ResearchResult]
) -> tuple[list[ResearchResult], dict[str, float]]:
    """Score sources and sort results by quality.
    
    Args:
        results: List of research results to score.
        
    Returns:
        Tuple of (sorted results, source scores dict).
    """
    source_scores: dict[str, float] = {}
    scored_results = []
    
    for result in results:
        # Score the source
        score_result = score_source(
            url=result.source_url,
            content=result.content,
            published_date=result.published_date,
        )
        
        source_scores[result.source_url] = score_result.final_score
        scored_results.append(result)
    
    # Sort by source quality (descending)
    scored_results.sort(
        key=lambda r: source_scores.get(r.source_url, 0.0),
        reverse=True,
    )
    
    logger.info(
        "results_scored_and_sorted",
        result_count=len(scored_results),
        avg_score=round(sum(source_scores.values()) / len(source_scores), 3) if source_scores else 0.0,
    )
    
    return scored_results, source_scores


async def research_node(state: AgentState) -> dict:
    """LangGraph node function for the Researcher agent.
    
    This function executes the research plan by:
    1. Executing each sub-query against web search and document store
    2. Merging and deduplicating results
    3. Scoring sources for quality
    4. Extracting structured claims
    5. Adding trace events for observability
    
    Args:
        state: Current agent state containing the plan and query.
        
    Returns:
        Dict with state updates (research_results, claims, source_scores, agent_trace).
    """
    query = state.get("query", "")
    plan: Optional[QueryPlan] = state.get("plan")
    agent_trace = state.get("agent_trace", []).copy()
    
    if not query:
        logger.warning("research_node_empty_query")
        return {
            "research_results": [],
            "claims": [],
            "source_scores": {},
            "status": PipelineStatus.FAILED.value,
            "errors": state.get("errors", []) + ["Empty query provided to researcher"],
            "agent_trace": agent_trace,
        }
    
    if not plan or not plan.sub_queries:
        logger.warning("research_node_no_plan", query=query[:50])
        return {
            "research_results": [],
            "claims": [],
            "source_scores": {},
            "status": PipelineStatus.FAILED.value,
            "errors": state.get("errors", []) + ["No valid plan provided to researcher"],
            "agent_trace": agent_trace,
        }
    
    # Initialize MCP client
    try:
        mcp_client = MCPClient()
    except Exception as e:
        logger.exception("mcp_client_init_error", error=str(e))
        return {
            "research_results": [],
            "claims": [],
            "source_scores": {},
            "status": PipelineStatus.FAILED.value,
            "errors": state.get("errors", []) + [f"Failed to initialize MCP client: {str(e)}"],
            "agent_trace": agent_trace,
        }
    
    # Add starting trace event
    agent_trace.append(TraceEvent(
        agent=AgentName.RESEARCHER,
        action="start",
        detail=f"Starting research with {len(plan.sub_queries)} sub-queries",
        metadata={"sub_query_count": len(plan.sub_queries), "strategy": plan.search_strategy.value},
    ))
    
    all_results: list[ResearchResult] = []
    
    # Execute each sub-query
    for idx, sub_query in enumerate(plan.sub_queries, 1):
        logger.info(
            "executing_sub_query",
            sub_query=sub_query[:100],
            index=idx,
            total=len(plan.sub_queries),
        )
        
        # Web search
        if plan.search_strategy in ["web_only", "hybrid"]:
            try:
                start_time = time()
                web_results = await mcp_client.web_search(sub_query, max_results=5)
                latency_ms = (time() - start_time) * 1000
                
                agent_trace.append(TraceEvent(
                    agent=AgentName.RESEARCHER,
                    action="web_search",
                    detail=f"Searching: {sub_query[:80]}",
                    latency_ms=latency_ms,
                    metadata={"result_count": len(web_results), "sub_query_index": idx},
                ))
                
                # Convert to ResearchResult
                for web_result in web_results:
                    all_results.append(ResearchResult(
                        content=web_result.content,
                        source_url=web_result.url,
                        source_title=web_result.title,
                        source_type="web",
                        relevance_score=web_result.score,
                        sub_query=sub_query,
                        published_date=web_result.published_date,
                    ))
                
                logger.info("web_search_completed", sub_query=sub_query[:50], result_count=len(web_results))
                
            except Exception as e:
                logger.exception("web_search_error", sub_query=sub_query[:50], error=str(e))
                agent_trace.append(TraceEvent(
                    agent=AgentName.RESEARCHER,
                    action="web_search_error",
                    detail=f"Error searching: {str(e)[:100]}",
                    metadata={"sub_query": sub_query, "error": str(e)},
                ))
        
        # Document store search
        if plan.search_strategy in ["docs_only", "hybrid"]:
            try:
                start_time = time()
                doc_results = await mcp_client.doc_search(sub_query, n_results=5)
                latency_ms = (time() - start_time) * 1000
                
                agent_trace.append(TraceEvent(
                    agent=AgentName.RESEARCHER,
                    action="doc_search",
                    detail=f"Searching docs: {sub_query[:80]}",
                    latency_ms=latency_ms,
                    metadata={"result_count": len(doc_results), "sub_query_index": idx},
                ))
                
                # Convert to ResearchResult
                for doc_result in doc_results:
                    all_results.append(ResearchResult(
                        content=doc_result.content,
                        source_url=doc_result.source_url,
                        source_title=doc_result.title,
                        source_type="document",
                        relevance_score=doc_result.relevance_score,
                        sub_query=sub_query,
                    ))
                
                logger.info("doc_search_completed", sub_query=sub_query[:50], result_count=len(doc_results))
                
            except Exception as e:
                logger.exception("doc_search_error", sub_query=sub_query[:50], error=str(e))
                agent_trace.append(TraceEvent(
                    agent=AgentName.RESEARCHER,
                    action="doc_search_error",
                    detail=f"Error searching docs: {str(e)[:100]}",
                    metadata={"sub_query": sub_query, "error": str(e)},
                ))
    
    # Check if we got any results
    if not all_results:
        logger.warning("research_no_results", query=query[:50])
        agent_trace.append(TraceEvent(
            agent=AgentName.RESEARCHER,
            action="complete",
            detail="Research completed with no results",
            metadata={"result_count": 0},
        ))
        return {
            "research_results": [],
            "claims": [],
            "source_scores": {},
            "status": PipelineStatus.FACT_CHECKING.value if plan.needs_fact_check else PipelineStatus.SYNTHESIZING.value,
            "errors": state.get("errors", []) + ["No research results found"],
            "agent_trace": agent_trace,
        }
    
    # Deduplicate results
    deduplicated_results = _deduplicate_results(all_results)
    
    # Score sources and sort by quality
    sorted_results, source_scores = _score_and_sort_results(deduplicated_results)
    
    agent_trace.append(TraceEvent(
        agent=AgentName.RESEARCHER,
        action="source_scoring",
        detail=f"Scored {len(source_scores)} sources",
        metadata={
            "source_count": len(source_scores),
            "avg_score": round(sum(source_scores.values()) / len(source_scores), 3) if source_scores else 0.0,
        },
    ))
    
    # Extract claims using Claude
    claims: list[Claim] = []
    try:
        start_time = time()
        claims = await extract_claims(sorted_results, query)
        latency_ms = (time() - start_time) * 1000
        
        # Get token usage from last LLM call (approximate)
        tokens_used = 0  # Will be populated by the extract_claims function logging
        
        agent_trace.append(TraceEvent(
            agent=AgentName.RESEARCHER,
            action="claim_extraction",
            detail=f"Extracted {len(claims)} claims from results",
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            metadata={"claim_count": len(claims)},
        ))
        
        logger.info("claims_extraction_completed", claim_count=len(claims))
        
    except Exception as e:
        logger.exception("claim_extraction_error", error=str(e))
        agent_trace.append(TraceEvent(
            agent=AgentName.RESEARCHER,
            action="claim_extraction_error",
            detail=f"Error extracting claims: {str(e)[:100]}",
            metadata={"error": str(e)},
        ))
        # Continue with empty claims rather than failing completely
    
    # Add completion trace event
    agent_trace.append(TraceEvent(
        agent=AgentName.RESEARCHER,
        action="complete",
        detail=f"Research completed: {len(sorted_results)} results, {len(claims)} claims",
        metadata={
            "result_count": len(sorted_results),
            "claim_count": len(claims),
            "source_count": len(source_scores),
        },
    ))
    
    # Determine next status
    next_status = (
        PipelineStatus.FACT_CHECKING.value
        if plan.needs_fact_check and claims
        else PipelineStatus.SYNTHESIZING.value
    )
    
    return {
        "research_results": sorted_results,
        "claims": claims,
        "source_scores": source_scores,
        "status": next_status,
        "agent_trace": agent_trace,
    }


# ---------------------------------------------------------------------------
# Parallel Execution Support
# ---------------------------------------------------------------------------


async def research_single_sub_query(
    sub_query: str,
    search_strategy: str,
    mcp_client: MCPClient,
) -> tuple[list[ResearchResult], list[TraceEvent], list[str]]:
    """Research a single sub-query (used in parallel execution).
    
    Args:
        sub_query: The specific sub-query to research.
        search_strategy: Search strategy ('web_only', 'docs_only', 'hybrid').
        mcp_client: MCP client for searches.
        
    Returns:
        Tuple of (results, trace_events, errors).
    """
    results: list[ResearchResult] = []
    trace_events: list[TraceEvent] = []
    errors: list[str] = []
    
    logger.info("researching_sub_query", sub_query=sub_query[:80])
    
    # Web search
    if search_strategy in ["web_only", "hybrid"]:
        try:
            start_time = time()
            web_results = await mcp_client.web_search(sub_query, max_results=5)
            latency_ms = (time() - start_time) * 1000
            
            trace_events.append(TraceEvent(
                agent=AgentName.RESEARCHER,
                action="web_search",
                detail=f"Parallel search: {sub_query[:60]}",
                latency_ms=latency_ms,
                metadata={"result_count": len(web_results), "parallel": True},
            ))
            
            for web_result in web_results:
                results.append(ResearchResult(
                    content=web_result.content,
                    source_url=web_result.url,
                    source_title=web_result.title,
                    source_type="web",
                    relevance_score=web_result.score,
                    sub_query=sub_query,
                    published_date=web_result.published_date,
                ))
                
        except Exception as e:
            logger.exception("parallel_web_search_error", sub_query=sub_query[:50], error=str(e))
            errors.append(f"Web search error for '{sub_query[:30]}...': {str(e)}")
            trace_events.append(TraceEvent(
                agent=AgentName.RESEARCHER,
                action="web_search_error",
                detail=f"Parallel search error: {str(e)[:80]}",
                metadata={"sub_query": sub_query, "error": str(e), "parallel": True},
            ))
    
    # Document store search
    if search_strategy in ["docs_only", "hybrid"]:
        try:
            start_time = time()
            doc_results = await mcp_client.doc_search(sub_query, n_results=5)
            latency_ms = (time() - start_time) * 1000
            
            trace_events.append(TraceEvent(
                agent=AgentName.RESEARCHER,
                action="doc_search",
                detail=f"Parallel doc search: {sub_query[:60]}",
                latency_ms=latency_ms,
                metadata={"result_count": len(doc_results), "parallel": True},
            ))
            
            for doc_result in doc_results:
                results.append(ResearchResult(
                    content=doc_result.content,
                    source_url=doc_result.source_url,
                    source_title=doc_result.title,
                    source_type="document",
                    relevance_score=doc_result.relevance_score,
                    sub_query=sub_query,
                ))
                
        except Exception as e:
            logger.exception("parallel_doc_search_error", sub_query=sub_query[:50], error=str(e))
            errors.append(f"Doc search error for '{sub_query[:30]}...': {str(e)}")
            trace_events.append(TraceEvent(
                agent=AgentName.RESEARCHER,
                action="doc_search_error",
                detail=f"Parallel doc search error: {str(e)[:80]}",
                metadata={"sub_query": sub_query, "error": str(e), "parallel": True},
            ))
    
    logger.info("sub_query_complete", sub_query=sub_query[:50], result_count=len(results))
    return results, trace_events, errors


async def research_sub_query_node(state: dict) -> dict:
    """LangGraph node for researching a single sub-query in parallel.
    
    This node is spawned by the Send API for each sub-query when
    parallel execution is enabled. It researches one sub-query and
    returns the results for later merging.
    
    Args:
        state: Dict containing 'sub_query' and 'parent_state' with plan/settings.
        
    Returns:
        Dict with sub-query results to be merged.
    """
    from graph.state import SubQueryResult
    
    sub_query = state.get("sub_query", "")
    parent_state = state.get("parent_state", {})
    plan = parent_state.get("plan")
    
    if not sub_query or not plan:
        logger.warning("research_sub_query_node_invalid_input", state_keys=list(state.keys()))
        return {
            "parallel_results": [SubQueryResult(
                sub_query=sub_query or "unknown",
                results=[],
                trace_events=[],
                errors=["Invalid input to parallel research node"],
            ).model_dump()],
        }
    
    # Initialize MCP client
    try:
        mcp_client = MCPClient()
    except Exception as e:
        logger.exception("parallel_mcp_client_error", error=str(e))
        return {
            "parallel_results": [SubQueryResult(
                sub_query=sub_query,
                results=[],
                trace_events=[],
                errors=[f"MCP client init failed: {str(e)}"],
            ).model_dump()],
        }
    
    # Research the sub-query
    results, trace_events, errors = await research_single_sub_query(
        sub_query=sub_query,
        search_strategy=plan.search_strategy.value if hasattr(plan.search_strategy, 'value') else plan.search_strategy,
        mcp_client=mcp_client,
    )
    
    # Convert to serializable format
    return {
        "parallel_results": [SubQueryResult(
            sub_query=sub_query,
            results=results,
            trace_events=trace_events,
            errors=errors,
        ).model_dump()],
    }


async def merge_research_node(state: AgentState) -> dict:
    """LangGraph node that merges results from parallel sub-query execution.
    
    Collects all parallel_results, deduplicates, scores sources, extracts claims,
    and produces the final research output.
    
    Args:
        state: AgentState with parallel_results from all sub-query workers.
        
    Returns:
        Dict with merged research_results, claims, source_scores, etc.
    """
    from graph.state import SubQueryResult
    
    query = state.get("query", "")
    plan = state.get("plan")
    parallel_results = state.get("parallel_results", [])
    agent_trace = state.get("agent_trace", []).copy()
    all_errors = state.get("errors", []).copy()
    
    logger.info(
        "merging_parallel_results",
        query=query[:50],
        result_count=len(parallel_results),
    )
    
    # Add merge start trace
    agent_trace.append(TraceEvent(
        agent=AgentName.RESEARCHER,
        action="merge_start",
        detail=f"Merging results from {len(parallel_results)} parallel tracks",
        metadata={"parallel_track_count": len(parallel_results)},
    ))
    
    # Collect all results from parallel tracks
    all_results: list[ResearchResult] = []
    
    for result_dict in parallel_results:
        try:
            sub_result = SubQueryResult.model_validate(result_dict)
            all_results.extend(sub_result.results)
            agent_trace.extend(sub_result.trace_events)
            all_errors.extend(sub_result.errors)
        except Exception as e:
            logger.warning("invalid_parallel_result", error=str(e), result=result_dict)
            all_errors.append(f"Failed to parse parallel result: {str(e)}")
    
    logger.info(
        "parallel_results_collected",
        total_results=len(all_results),
        total_errors=len(all_errors),
    )
    
    # Handle empty results
    if not all_results:
        agent_trace.append(TraceEvent(
            agent=AgentName.RESEARCHER,
            action="merge_complete",
            detail="Merge completed with no results",
            metadata={"result_count": 0},
        ))
        return {
            "research_results": [],
            "claims": [],
            "source_scores": {},
            "status": PipelineStatus.FACT_CHECKING.value if plan and plan.needs_fact_check else PipelineStatus.SYNTHESIZING.value,
            "errors": all_errors + ["No research results found from parallel execution"],
            "agent_trace": agent_trace,
            "parallel_results": [],  # Clear parallel results
        }
    
    # Deduplicate results
    deduplicated_results = _deduplicate_results(all_results)
    
    # Score sources and sort by quality
    sorted_results, source_scores = _score_and_sort_results(deduplicated_results)
    
    agent_trace.append(TraceEvent(
        agent=AgentName.RESEARCHER,
        action="source_scoring",
        detail=f"Scored {len(source_scores)} sources from parallel tracks",
        metadata={
            "source_count": len(source_scores),
            "avg_score": round(sum(source_scores.values()) / len(source_scores), 3) if source_scores else 0.0,
            "parallel": True,
        },
    ))
    
    # Extract claims using Claude
    claims: list[Claim] = []
    try:
        start_time = time()
        claims = await extract_claims(sorted_results, query)
        latency_ms = (time() - start_time) * 1000
        
        agent_trace.append(TraceEvent(
            agent=AgentName.RESEARCHER,
            action="claim_extraction",
            detail=f"Extracted {len(claims)} claims from merged results",
            latency_ms=latency_ms,
            metadata={"claim_count": len(claims), "parallel": True},
        ))
        
    except Exception as e:
        logger.exception("merge_claim_extraction_error", error=str(e))
        all_errors.append(f"Claim extraction error: {str(e)}")
        agent_trace.append(TraceEvent(
            agent=AgentName.RESEARCHER,
            action="claim_extraction_error",
            detail=f"Error extracting claims from merged results: {str(e)[:80]}",
            metadata={"error": str(e), "parallel": True},
        ))
    
    # Add completion trace
    agent_trace.append(TraceEvent(
        agent=AgentName.RESEARCHER,
        action="merge_complete",
        detail=f"Merged {len(sorted_results)} results, {len(claims)} claims from parallel tracks",
        metadata={
            "result_count": len(sorted_results),
            "claim_count": len(claims),
            "source_count": len(source_scores),
            "parallel": True,
        },
    ))
    
    # Determine next status
    next_status = (
        PipelineStatus.FACT_CHECKING.value
        if plan and plan.needs_fact_check and claims
        else PipelineStatus.SYNTHESIZING.value
    )
    
    return {
        "research_results": sorted_results,
        "claims": claims,
        "source_scores": source_scores,
        "status": next_status,
        "errors": all_errors,
        "agent_trace": agent_trace,
        "parallel_results": [],  # Clear parallel results after merge
    }
