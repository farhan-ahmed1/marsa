"""Fact-Checker Agent for MARSA.

The Fact-Checker verifies claims from the Researcher by running independent
verification searches and determining if claims are supported, contradicted,
or unverifiable based on the evidence found.

Key features:
- Generates alternative verification queries (different from original searches)
- Uses Claude to analyze evidence and determine verdicts
- Integrates source quality scores into confidence calculations
- Implements loop-back logic for re-research when too many claims fail
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
    PipelineStatus,
    TraceEvent,
    VerificationResult,
    VerificationVerdict,
)
from mcp_client import MCPClient

logger = structlog.get_logger(__name__)


# Maximum iterations to prevent infinite loops
MAX_ITERATIONS = 2

# Threshold for bad claims that triggers re-research (40%)
# Raised from 0.3 to reduce false-positive re-research loops.
# Unverifiable claims are common for recent or niche topics and
# should not automatically trigger expensive re-research.
BAD_CLAIM_THRESHOLD = 0.4


VERIFICATION_QUERY_SYSTEM_PROMPT = """\
You are a fact-checking assistant. Your task is to generate a verification \
query that is different from the original search but can help verify a claim.

The verification query should:
1. Approach the claim from a different angle
2. Use different keywords than would have been used to find the original claim
3. Seek authoritative sources that could confirm or contradict the claim
4. Be specific enough to get relevant results

Output ONLY the verification query as plain text, nothing else.

Examples:
- Claim: "Python was created by Guido van Rossum"
  Verification query: "Who created the Python programming language history origin"

- Claim: "Go was released in 2015"
  Verification query: "Go programming language golang release date first version"

- Claim: "Rust guarantees memory safety without garbage collection"
  Verification query: "Rust memory safety mechanism ownership borrow checker"
"""


VERIFY_CLAIM_SYSTEM_PROMPT = """\
You are a fact-checking agent. Your task is to evaluate whether a claim is \
SUPPORTED, CONTRADICTED, or UNVERIFIABLE based on the search results provided.

## Evaluation Guidelines

**SUPPORTED**: The claim is confirmed by the search results. Multiple credible \
sources agree, or a single highly authoritative source confirms it.

**CONTRADICTED**: The claim conflicts with the search results. Sources provide \
information that directly contradicts the claim.

**UNVERIFIABLE**: The search results do not contain enough information to \
confirm or deny the claim. The topic may be covered, but the specific claim \
cannot be verified.

IMPORTANT: Only mark a claim as CONTRADICTED if sources explicitly and \
directly refute it. Absence of evidence is NOT contradiction. If you cannot \
find evidence for or against a claim, mark it UNVERIFIABLE, not CONTRADICTED.

Be especially careful with:
- Recent claims (data may not be indexed yet)
- Niche technical claims (limited sources)
- Statistical claims (minor differences in numbers do not equal contradiction)

## Confidence Scoring

Score your confidence from 0.0 to 1.0 based on:
- Number of sources (more = higher confidence)
- Source quality (authoritative sources increase confidence)
- Consistency of information across sources
- Directness of evidence (explicit statements vs inferences)

Confidence guidelines:
- 0.9-1.0: Multiple authoritative sources explicitly confirm/contradict
- 0.7-0.9: At least one authoritative source with corroboration
- 0.5-0.7: Sources provide indirect evidence
- 0.3-0.5: Limited evidence, mostly inference
- 0.0-0.3: Very weak evidence, high uncertainty

## Output Format

Output ONLY valid JSON matching this schema (no markdown, no extra text):
{
    "verdict": "supported" | "contradicted" | "unverifiable",
    "confidence": 0.0-1.0,
    "supporting_sources": ["url1", "url2"],
    "contradicting_sources": ["url1", "url2"],
    "reasoning": "Brief explanation of your verdict"
}
"""


def _create_llm(model: str = "claude-sonnet-4-20250514") -> ChatAnthropic:
    """Create a ChatAnthropic instance for fact-checking.
    
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


def _parse_verification_response(
    response_text: str,
    claim: Claim,
    verification_query: str,
) -> VerificationResult:
    """Parse LLM response into a VerificationResult.
    
    Args:
        response_text: Raw JSON response from the LLM.
        claim: The original claim being verified.
        verification_query: The query used for verification.
        
    Returns:
        Validated VerificationResult model.
        
    Raises:
        ValueError: If response cannot be parsed into valid VerificationResult.
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
    
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")
    
    # Map verdict string to enum
    verdict_str = data.get("verdict", "unverifiable").lower()
    verdict_map = {
        "supported": VerificationVerdict.SUPPORTED,
        "contradicted": VerificationVerdict.CONTRADICTED,
        "unverifiable": VerificationVerdict.UNVERIFIABLE,
    }
    verdict = verdict_map.get(verdict_str, VerificationVerdict.UNVERIFIABLE)
    
    # Extract confidence, clamping to valid range
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    
    return VerificationResult(
        claim=claim,
        verdict=verdict,
        confidence=confidence,
        supporting_sources=data.get("supporting_sources", []),
        contradicting_sources=data.get("contradicting_sources", []),
        reasoning=data.get("reasoning", ""),
        verification_query=verification_query,
    )


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((ValueError,)),
    reraise=True,
)
async def generate_verify_query(
    claim: Claim,
    llm: Optional[ChatAnthropic] = None,
) -> str:
    """Generate an independent verification query for a claim.
    
    Creates a search query that approaches the claim from a different angle
    than the original research query to get independent verification.
    
    Args:
        claim: The claim to generate a verification query for.
        llm: Optional ChatAnthropic instance (created if not provided).
        
    Returns:
        A verification search query string.
    """
    if llm is None:
        llm = _create_llm()
    
    user_prompt = f"""Generate a verification query for this claim:

Claim: {claim.statement}
Original source: {claim.source_title}
Category: {claim.category}

Generate a verification query that will help confirm or contradict this claim \
using different search terms than would have been used originally."""
    
    messages = [
        SystemMessage(content=VERIFICATION_QUERY_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    
    logger.debug("generating_verify_query", claim=claim.statement[:50])
    
    response = await llm.ainvoke(messages)
    
    response_text = response.content
    if isinstance(response_text, list):
        response_text = "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in response_text
        )
    
    return str(response_text).strip()


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((ValueError,)),
    reraise=True,
)
async def verify_claim(
    claim: Claim,
    search_results: list,
    source_scores: dict[str, float],
    llm: Optional[ChatAnthropic] = None,
    verification_query: str = "",
) -> VerificationResult:
    """Verify a claim against search results using Claude.
    
    Sends the claim and verification search results to Claude to determine
    if the claim is supported, contradicted, or unverifiable.
    
    Args:
        claim: The claim to verify.
        search_results: Search results from the verification query.
        source_scores: Pre-computed source quality scores (URL -> score).
        llm: Optional ChatAnthropic instance (created if not provided).
        verification_query: The query used to get these results.
        
    Returns:
        VerificationResult with verdict, confidence, and reasoning.
    """
    if llm is None:
        llm = _create_llm()
    
    # Format search results for the LLM
    results_text = ""
    for idx, result in enumerate(search_results[:10], 1):  # Limit to 10 results
        url = getattr(result, 'url', getattr(result, 'source_url', ''))
        title = getattr(result, 'title', getattr(result, 'source_title', 'Unknown'))
        content = getattr(result, 'content', '')
        
        # Get source quality score
        quality_score = source_scores.get(url, 0.5)
        
        results_text += f"\n\n--- Source {idx} ---\n"
        results_text += f"Title: {title}\n"
        results_text += f"URL: {url}\n"
        results_text += f"Quality Score: {quality_score:.2f}\n"
        results_text += f"Content: {content[:500]}\n"  # Limit content length
    
    if not results_text.strip():
        # No search results - claim is unverifiable
        return VerificationResult(
            claim=claim,
            verdict=VerificationVerdict.UNVERIFIABLE,
            confidence=0.3,
            supporting_sources=[],
            contradicting_sources=[],
            reasoning="No search results found to verify this claim.",
            verification_query=verification_query,
        )
    
    user_prompt = f"""Verify this claim based on the search results:

## Claim to Verify
Statement: {claim.statement}
Original Source: {claim.source_title} ({claim.source_url})
Original Confidence: {claim.confidence.value}
Category: {claim.category}

## Verification Search Results
{results_text}

Analyze the search results and determine if the claim is SUPPORTED, CONTRADICTED, \
or UNVERIFIABLE. Consider the quality scores of sources when weighing evidence."""
    
    messages = [
        SystemMessage(content=VERIFY_CLAIM_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    
    logger.info("verifying_claim", claim=claim.statement[:50])
    
    start_time = time()
    response = await llm.ainvoke(messages)
    latency_ms = (time() - start_time) * 1000
    
    response_text = response.content
    if isinstance(response_text, list):
        response_text = "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in response_text
        )
    
    result = _parse_verification_response(
        str(response_text),
        claim,
        verification_query,
    )
    
    # Adjust confidence based on source quality
    result = _adjust_confidence_by_source_quality(result, source_scores)
    
    logger.info(
        "claim_verified",
        claim=claim.statement[:50],
        verdict=result.verdict.value,
        confidence=round(result.confidence, 2),
        latency_ms=round(latency_ms, 2),
    )
    
    return result


def _adjust_confidence_by_source_quality(
    result: VerificationResult,
    source_scores: dict[str, float],
) -> VerificationResult:
    """Adjust verification confidence based on source quality scores.
    
    Claims supported by high-quality sources (e.g., .gov, .edu) get a
    confidence boost, while claims relying on lower-quality sources
    have their confidence reduced.
    
    Args:
        result: The verification result to adjust.
        source_scores: Pre-computed source quality scores (URL -> score).
        
    Returns:
        VerificationResult with adjusted confidence.
    """
    relevant_sources = (
        result.supporting_sources 
        if result.verdict == VerificationVerdict.SUPPORTED 
        else result.contradicting_sources
    )
    
    if not relevant_sources:
        return result
    
    # Calculate average quality of relevant sources
    quality_scores = [
        source_scores.get(url, 0.5) 
        for url in relevant_sources
    ]
    avg_quality = sum(quality_scores) / len(quality_scores)
    
    # Adjust confidence: high-quality sources boost, low-quality sources reduce
    # avg_quality of 0.8+ gives up to 10% boost
    # avg_quality of 0.4 or below gives up to 15% reduction
    if avg_quality >= 0.7:
        adjustment = (avg_quality - 0.5) * 0.2  # Up to +0.06
    elif avg_quality <= 0.4:
        adjustment = (avg_quality - 0.5) * 0.3  # Up to -0.03
    else:
        adjustment = 0.0
    
    new_confidence = max(0.0, min(1.0, result.confidence + adjustment))
    
    if adjustment != 0.0:
        logger.debug(
            "confidence_adjusted",
            original=round(result.confidence, 2),
            adjusted=round(new_confidence, 2),
            avg_source_quality=round(avg_quality, 2),
        )
    
    # Create a new VerificationResult with updated confidence
    return VerificationResult(
        claim=result.claim,
        verdict=result.verdict,
        confidence=new_confidence,
        supporting_sources=result.supporting_sources,
        contradicting_sources=result.contradicting_sources,
        reasoning=result.reasoning,
        verification_query=result.verification_query,
    )


def should_loop_back(state: AgentState) -> str:
    """Determine if the pipeline should loop back for re-research.
    
    If more than 30% of claims are contradicted or unverifiable,
    and we haven't exceeded the maximum iterations, route back
    to the researcher for another attempt with refined queries.
    
    Args:
        state: Current agent state with verification results.
        
    Returns:
        "researcher" to loop back, or "synthesizer" to proceed.
    """
    results = state.get("verification_results", [])
    iteration_count = state.get("iteration_count", 0)
    
    if not results:
        logger.warning("should_loop_back_no_results")
        return "synthesizer"
    
    # Count truly bad claims (contradicted only, not unverifiable)
    # Unverifiable claims are expected for niche/recent topics and should
    # not trigger expensive re-research loops.
    bad_count = sum(
        1 for r in results 
        if r.verdict == VerificationVerdict.CONTRADICTED
    )
    bad_ratio = bad_count / len(results)
    
    logger.info(
        "loop_back_evaluation",
        bad_count=bad_count,
        total_count=len(results),
        bad_ratio=round(bad_ratio, 2),
        iteration_count=iteration_count,
        max_iterations=MAX_ITERATIONS,
    )
    
    # Loop back if too many bad claims and under iteration limit
    if bad_ratio > BAD_CLAIM_THRESHOLD and iteration_count < MAX_ITERATIONS:
        logger.info(
            "looping_back_to_researcher",
            reason=f"{bad_count}/{len(results)} claims not supported",
        )
        return "researcher"
    
    logger.info("proceeding_to_synthesizer")
    return "synthesizer"


async def fact_check_node(state: AgentState) -> dict:
    """LangGraph node function for the Fact-Checker agent.
    
    This function verifies claims from the Researcher by:
    1. Generating independent verification queries for each claim
    2. Running verification searches via MCP servers
    3. Sending claims + search results to Claude for verdict
    4. Adjusting confidence based on source quality
    5. Adding trace events for observability
    
    Args:
        state: Current agent state containing claims to verify.
        
    Returns:
        Dict with state updates (verification_results, source_scores, agent_trace).
    """
    claims = state.get("claims", [])
    existing_source_scores = state.get("source_scores", {}).copy()
    agent_trace = state.get("agent_trace", []).copy()
    iteration_count = state.get("iteration_count", 0)
    
    # Add starting trace event
    agent_trace.append(TraceEvent(
        agent=AgentName.FACT_CHECKER,
        action="start",
        detail=f"Starting fact-check of {len(claims)} claims (iteration {iteration_count + 1})",
        metadata={"claim_count": len(claims), "iteration": iteration_count + 1},
    ))
    
    if not claims:
        logger.warning("fact_check_node_no_claims")
        agent_trace.append(TraceEvent(
            agent=AgentName.FACT_CHECKER,
            action="skip",
            detail="No claims to verify",
        ))
        return {
            "verification_results": [],
            "source_scores": existing_source_scores,
            "status": PipelineStatus.FACT_CHECKING.value,
            "iteration_count": iteration_count + 1,
            "agent_trace": agent_trace,
        }
    
    # Initialize MCP client
    try:
        mcp_client = MCPClient()
    except Exception as e:
        logger.exception("mcp_client_init_error", error=str(e))
        agent_trace.append(TraceEvent(
            agent=AgentName.FACT_CHECKER,
            action="error",
            detail=f"Failed to initialize MCP client: {str(e)}",
        ))
        return {
            "verification_results": [],
            "source_scores": existing_source_scores,
            "status": PipelineStatus.FAILED.value,
            "errors": state.get("errors", []) + [f"MCP client error: {str(e)}"],
            "iteration_count": iteration_count + 1,
            "agent_trace": agent_trace,
        }
    
    llm = _create_llm()
    verification_results: list[VerificationResult] = []
    source_scores = existing_source_scores.copy()
    
    for idx, claim in enumerate(claims):
        claim_start_time = time()
        
        try:
            # Step 1: Generate verification query
            query_start = time()
            verify_query = await generate_verify_query(claim, llm)
            query_latency = (time() - query_start) * 1000
            
            agent_trace.append(TraceEvent(
                agent=AgentName.FACT_CHECKER,
                action="generate_query",
                detail=f"Generated verify query: {verify_query[:50]}...",
                latency_ms=round(query_latency, 2),
                metadata={"claim_index": idx, "query": verify_query},
            ))
            
            # Step 2: Run verification search
            search_start = time()
            search_results = await mcp_client.web_search(verify_query, max_results=5)
            search_latency = (time() - search_start) * 1000
            
            agent_trace.append(TraceEvent(
                agent=AgentName.FACT_CHECKER,
                action="web_search",
                detail=f"Verification search returned {len(search_results)} results",
                latency_ms=round(search_latency, 2),
                metadata={"claim_index": idx, "result_count": len(search_results)},
            ))
            
            # Step 3: Score new sources
            for result in search_results:
                url = getattr(result, 'url', getattr(result, 'source_url', ''))
                if url and url not in source_scores:
                    content = getattr(result, 'content', '')
                    published_date = getattr(result, 'published_date', None)
                    score_result = score_source(url, content, published_date)
                    source_scores[url] = score_result.final_score
            
            # Step 4: Verify claim
            verify_start = time()
            result = await verify_claim(
                claim=claim,
                search_results=search_results,
                source_scores=source_scores,
                llm=llm,
                verification_query=verify_query,
            )
            verify_latency_ms = (time() - verify_start) * 1000
            
            verification_results.append(result)
            
            total_latency = (time() - claim_start_time) * 1000
            agent_trace.append(TraceEvent(
                agent=AgentName.FACT_CHECKER,
                action="claim_verified",
                detail=f"Claim {idx + 1}: {result.verdict.value} ({result.confidence:.2f})",
                latency_ms=round(total_latency, 2),
                metadata={
                    "claim_index": idx,
                    "verdict": result.verdict.value,
                    "confidence": round(result.confidence, 2),
                    "supporting_count": len(result.supporting_sources),
                    "contradicting_count": len(result.contradicting_sources),
                    "verify_latency_ms": round(verify_latency_ms, 2),
                },
            ))
            
        except Exception as e:
            logger.exception(
                "claim_verification_error",
                claim_index=idx,
                claim=claim.statement[:50],
                error=str(e),
            )
            
            # Create an unverifiable result for failed claims
            verification_results.append(VerificationResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIABLE,
                confidence=0.1,
                supporting_sources=[],
                contradicting_sources=[],
                reasoning=f"Verification failed due to error: {str(e)}",
                verification_query="",
            ))
            
            agent_trace.append(TraceEvent(
                agent=AgentName.FACT_CHECKER,
                action="error",
                detail=f"Failed to verify claim {idx + 1}: {str(e)}",
                metadata={"claim_index": idx, "error": str(e)},
            ))
    
    # Calculate summary statistics
    supported_count = sum(
        1 for r in verification_results 
        if r.verdict == VerificationVerdict.SUPPORTED
    )
    contradicted_count = sum(
        1 for r in verification_results 
        if r.verdict == VerificationVerdict.CONTRADICTED
    )
    unverifiable_count = sum(
        1 for r in verification_results 
        if r.verdict == VerificationVerdict.UNVERIFIABLE
    )
    
    agent_trace.append(TraceEvent(
        agent=AgentName.FACT_CHECKER,
        action="complete",
        detail=f"Verified {len(verification_results)} claims: "
               f"{supported_count} supported, {contradicted_count} contradicted, "
               f"{unverifiable_count} unverifiable",
        metadata={
            "total_claims": len(verification_results),
            "supported": supported_count,
            "contradicted": contradicted_count,
            "unverifiable": unverifiable_count,
        },
    ))
    
    logger.info(
        "fact_check_complete",
        total_claims=len(verification_results),
        supported=supported_count,
        contradicted=contradicted_count,
        unverifiable=unverifiable_count,
    )
    
    return {
        "verification_results": verification_results,
        "source_scores": source_scores,
        "status": PipelineStatus.FACT_CHECKING.value,
        "iteration_count": iteration_count + 1,
        "agent_trace": agent_trace,
    }
