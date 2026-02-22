"""Synthesizer Agent for MARSA.

The Synthesizer takes verified claims from the Fact-Checker and produces
a structured research report with inline citations, confidence assessment,
and source quality integration.

Key features:
- Organizes findings into logical sections
- Cites sources inline using numbered references [1], [2]
- Flags areas of uncertainty or conflicting information
- Includes confidence summary based on verification results
- Prioritizes claims backed by higher-quality sources
"""

import json
from datetime import datetime, timezone
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

from config import config
from graph.state import (
    AgentName,
    AgentState,
    Citation,
    PipelineStatus,
    Report,
    ReportMetadata,
    ReportSection,
    TraceEvent,
    VerificationResult,
    VerificationVerdict,
)

logger = structlog.get_logger(__name__)


REPORT_GENERATION_SYSTEM_PROMPT = """\
You are a research report synthesizer. Your task is to generate a comprehensive, \
well-structured research report based on verified claims and source information.

## Report Structure

Generate a JSON report with:
- **title**: A clear, descriptive title for the report
- **summary**: A 2-3 sentence executive summary of key findings
- **sections**: List of content sections with headings and content
- **confidence_summary**: Overall assessment of confidence and any caveats

## Writing Guidelines

1. **Organization**: Group related findings into logical sections with clear headings
2. **Citations**: Use numbered inline citations like [1], [2] to reference sources. EVERY factual claim MUST have at least one citation. Aim for 1-3 citations per paragraph.
3. **Objectivity**: Present findings objectively, distinguish facts from opinions
4. **Uncertainty**: Flag areas of uncertainty or conflicting information clearly
5. **Prioritization**: Emphasize claims backed by high-quality sources (score >= 0.7)
6. **Completeness**: Address all aspects of the original query
7. **Depth**: Each section should contain 2-4 substantive paragraphs (100-200 words per section minimum). Do NOT write single-sentence sections.
8. **Report Length**: The full report should be 500-1500 words depending on query complexity. Comparison and exploratory queries need the longer end; simple factual queries can be shorter.

## Citation Format

- Use [N] format for inline citations where N is the citation number
- Each unique source gets a single citation number
- Place citations immediately after the relevant claim
- EVERY factual statement MUST have a citation. Uncited claims reduce report quality.
- Example: "Python was created by Guido van Rossum [1] and first released in 1991 [2]."
- For paragraphs synthesizing multiple sources: "Studies show X [1][3] while others note Y [2][4]."

## Confidence Summary Guidelines

Based on the verification results, assess overall confidence:
- If most claims are supported with high confidence: "High confidence in these findings..."
- If some claims are unverifiable: "Moderate confidence, with some claims requiring further verification..."
- If significant contradictions exist: "Mixed evidence found, with conflicting sources on..."

## Section Guidelines

Create 2-5 sections based on the query complexity:
- For comparison queries: one section per item being compared, plus a summary
- For factual queries: may only need 1-2 sections
- For exploratory queries: organize by subtopic or theme

## Output Format

Output ONLY valid JSON matching this schema (no markdown, no extra text):
{
    "title": "Report Title",
    "summary": "2-3 sentence executive summary",
    "sections": [
        {
            "heading": "Section Heading",
            "content": "Section content with [1] inline citations [2].",
            "order": 1
        }
    ],
    "confidence_summary": "Overall confidence assessment"
}
"""


def _create_llm(model: str = "claude-sonnet-4-20250514") -> ChatAnthropic:
    """Create a ChatAnthropic instance for report generation.
    
    Args:
        model: The Claude model to use.
        
    Returns:
        Configured ChatAnthropic instance.
    """
    return ChatAnthropic(
        model=model,
        api_key=config.anthropic_api_key,
        temperature=0.3,  # Slightly higher for more natural writing
        max_tokens=4096,  # Reports can be longer
    )


def _build_citation_map(
    verification_results: list[VerificationResult],
    source_scores: dict[str, float],
) -> tuple[list[Citation], dict[str, int]]:
    """Build a list of citations and a URL-to-number mapping.
    
    Creates unique citations for each source URL, prioritizing
    high-quality sources with lower citation numbers.
    
    Args:
        verification_results: Verified claims with source information.
        source_scores: Quality scores for sources (URL -> score).
        
    Returns:
        Tuple of (list of Citation objects, URL to citation number mapping).
    """
    # Collect all unique source URLs with their metadata
    url_info: dict[str, dict] = {}
    
    for result in verification_results:
        # Add claim's original source
        claim_url = result.claim.source_url
        if claim_url and claim_url not in url_info:
            url_info[claim_url] = {
                "title": result.claim.source_title or claim_url,
                "url": claim_url,
                "score": source_scores.get(claim_url, 0.5),
            }
        
        # Add supporting sources
        for url in result.supporting_sources:
            if url and url not in url_info:
                url_info[url] = {
                    "title": url,  # We may not have titles for verification sources
                    "url": url,
                    "score": source_scores.get(url, 0.5),
                }
    
    # Sort by quality score (highest first) for citation numbering
    sorted_urls = sorted(
        url_info.items(),
        key=lambda x: x[1]["score"],
        reverse=True,
    )
    
    citations: list[Citation] = []
    url_to_number: dict[str, int] = {}
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    for idx, (url, info) in enumerate(sorted_urls, 1):
        citations.append(Citation(
            number=idx,
            title=info["title"],
            url=url,
            source_quality_score=info["score"],
            accessed_date=today,
            source_type="web",
        ))
        url_to_number[url] = idx
    
    logger.info("citations_built", count=len(citations))
    return citations, url_to_number


def _format_claims_for_prompt(
    verification_results: list[VerificationResult],
    source_scores: dict[str, float],
    url_to_citation: dict[str, int],
) -> str:
    """Format verified claims for the LLM prompt.
    
    Args:
        verification_results: Verified claims to include.
        source_scores: Quality scores for sources.
        url_to_citation: Mapping of URLs to citation numbers.
        
    Returns:
        Formatted string for the LLM prompt.
    """
    lines = []
    
    # Group by verdict
    supported = [r for r in verification_results if r.verdict == VerificationVerdict.SUPPORTED]
    unverifiable = [r for r in verification_results if r.verdict == VerificationVerdict.UNVERIFIABLE]
    contradicted = [r for r in verification_results if r.verdict == VerificationVerdict.CONTRADICTED]
    
    if supported:
        lines.append("## Supported Claims (include these)")
        for idx, result in enumerate(supported, 1):
            citation_num = url_to_citation.get(result.claim.source_url, "?")
            score = source_scores.get(result.claim.source_url, 0.5)
            lines.append(f"\n{idx}. {result.claim.statement}")
            lines.append(f"   Source: {result.claim.source_title} [citation {citation_num}]")
            lines.append(f"   Quality Score: {score:.2f}")
            lines.append(f"   Confidence: {result.confidence:.2f}")
            lines.append(f"   Category: {result.claim.category}")
            if result.claim.context:
                lines.append(f"   Context: {result.claim.context}")
    
    if unverifiable:
        lines.append("\n## Unverified Claims (mention with caveats)")
        for idx, result in enumerate(unverifiable, 1):
            citation_num = url_to_citation.get(result.claim.source_url, "?")
            lines.append(f"\n{idx}. {result.claim.statement}")
            lines.append(f"   Source: {result.claim.source_title} [citation {citation_num}]")
            lines.append(f"   Note: {result.reasoning}")
    
    if contradicted:
        lines.append("\n## Contradicted Claims (mention the contradiction)")
        for idx, result in enumerate(contradicted, 1):
            lines.append(f"\n{idx}. Original claim: {result.claim.statement}")
            lines.append(f"   Contradiction: {result.reasoning}")
            if result.contradicting_sources:
                contra_citations = [
                    str(url_to_citation.get(url, "?")) 
                    for url in result.contradicting_sources[:3]
                ]
                lines.append(f"   Contradicting sources: [{', '.join(contra_citations)}]")
    
    return "\n".join(lines)


def _parse_report_response(
    response_text: str,
    query: str,
    citations: list[Citation],
    stats: dict,
) -> Report:
    """Parse LLM response into a Report object.
    
    Args:
        response_text: Raw JSON response from the LLM.
        query: Original user query.
        citations: Pre-built citation list.
        stats: Statistics about the research process.
        
    Returns:
        Validated Report model.
        
    Raises:
        ValueError: If response cannot be parsed into valid Report.
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
    
    # Parse sections
    sections = []
    sections_data = data.get("sections", [])
    if isinstance(sections_data, list):
        for idx, section_obj in enumerate(sections_data):
            if isinstance(section_obj, dict):
                sections.append(ReportSection(
                    heading=section_obj.get("heading", f"Section {idx + 1}"),
                    content=section_obj.get("content", ""),
                    order=section_obj.get("order", idx + 1),
                ))
    
    # Sort sections by order
    sections.sort(key=lambda s: s.order)
    
    # Build metadata
    metadata = ReportMetadata(
        query=query,
        total_latency_ms=stats.get("total_latency_ms", 0.0),
        llm_calls=stats.get("llm_calls", 0),
        total_tokens=stats.get("total_tokens", 0),
        sources_searched=stats.get("sources_searched", 0),
        claims_verified=stats.get("claims_verified", 0),
        fact_check_pass_rate=stats.get("fact_check_pass_rate", 0.0),
    )
    
    return Report(
        title=data.get("title", f"Research Report: {query[:50]}"),
        summary=data.get("summary", ""),
        sections=sections,
        confidence_summary=data.get("confidence_summary", ""),
        citations=citations,
        metadata=metadata,
    )


def _calculate_stats(
    state: AgentState,
    verification_results: list[VerificationResult],
) -> dict:
    """Calculate statistics for the report metadata.
    
    Args:
        state: Current agent state.
        verification_results: Verification results to analyze.
        
    Returns:
        Dictionary of statistics.
    """
    # Count supported claims
    supported_count = sum(
        1 for r in verification_results
        if r.verdict == VerificationVerdict.SUPPORTED
    )
    
    total_claims = len(verification_results)
    pass_rate = supported_count / total_claims if total_claims > 0 else 0.0
    
    # Calculate timing from trace events
    agent_trace = state.get("agent_trace", [])
    total_latency_ms = sum(
        event.latency_ms or 0.0
        for event in agent_trace
        if hasattr(event, 'latency_ms') and event.latency_ms
    )
    
    # Count LLM calls and tokens
    llm_calls = sum(
        1 for event in agent_trace
        if hasattr(event, 'action') and event.action in ["llm_call", "generate_query", "claim_verified", "extract_claims"]
    )
    
    total_tokens = sum(
        event.tokens_used or 0
        for event in agent_trace
        if hasattr(event, 'tokens_used') and event.tokens_used
    )
    
    # Count sources
    research_results = state.get("research_results", [])
    sources_searched = len(research_results)
    
    return {
        "total_latency_ms": total_latency_ms,
        "llm_calls": llm_calls,
        "total_tokens": total_tokens,
        "sources_searched": sources_searched,
        "claims_verified": total_claims,
        "fact_check_pass_rate": pass_rate,
    }


def _format_report_as_text(report: Report) -> str:
    """Format a Report object as readable text.
    
    Args:
        report: The report to format.
        
    Returns:
        Formatted text representation.
    """
    lines = []
    
    # Title
    lines.append(f"# {report.title}")
    lines.append("")
    
    # Summary
    if report.summary:
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(report.summary)
        lines.append("")
    
    # Sections
    for section in report.sections:
        lines.append(f"## {section.heading}")
        lines.append("")
        lines.append(section.content)
        lines.append("")
    
    # Confidence Summary
    if report.confidence_summary:
        lines.append("## Confidence Assessment")
        lines.append("")
        lines.append(report.confidence_summary)
        lines.append("")
    
    # Citations
    if report.citations:
        lines.append("## References")
        lines.append("")
        for citation in report.citations:
            quality_indicator = ""
            if citation.source_quality_score >= 0.7:
                quality_indicator = " (high quality)"
            elif citation.source_quality_score >= 0.5:
                quality_indicator = " (moderate quality)"
            else:
                quality_indicator = " (low quality)"
            
            lines.append(f"[{citation.number}] {citation.title}")
            lines.append(f"    URL: {citation.url}")
            lines.append(f"    Quality: {citation.source_quality_score:.2f}{quality_indicator}")
            lines.append("")
    
    return "\n".join(lines)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((ValueError,)),
    reraise=True,
)
async def generate_report(
    query: str,
    verification_results: list[VerificationResult],
    source_scores: dict[str, float],
    state: AgentState,
    llm: Optional[ChatAnthropic] = None,
) -> Report:
    """Generate a structured research report from verified claims.
    
    Uses Claude to synthesize verified claims into a well-organized
    report with inline citations and confidence assessment.
    
    Args:
        query: Original user query.
        verification_results: Verified claims with verdicts.
        source_scores: Quality scores for sources (URL -> score).
        state: Current agent state for statistics.
        llm: Optional ChatAnthropic instance (created if not provided).
        
    Returns:
        Structured Report object.
    """
    if llm is None:
        llm = _create_llm()
    
    # Build citations
    citations, url_to_citation = _build_citation_map(
        verification_results, 
        source_scores,
    )
    
    # Format claims for the prompt
    claims_text = _format_claims_for_prompt(
        verification_results,
        source_scores,
        url_to_citation,
    )
    
    # Calculate statistics
    stats = _calculate_stats(state, verification_results)
    
    # Build citation reference
    citation_ref = "\n".join(
        f"[{c.number}] {c.title} ({c.url}) - Quality: {c.source_quality_score:.2f}"
        for c in citations
    )
    
    user_prompt = f"""Generate a research report for this query:

## Original Query
{query}

## Verified Claims and Sources
{claims_text}

## Available Citations
Use these citation numbers in your report:
{citation_ref}

## Statistics
- Claims verified: {stats['claims_verified']}
- Fact-check pass rate: {stats['fact_check_pass_rate']:.1%}
- Sources consulted: {stats['sources_searched']}

Generate a comprehensive report that synthesizes these findings, uses inline \
citations [N], and provides an honest assessment of confidence in the conclusions."""

    messages = [
        SystemMessage(content=REPORT_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    
    logger.info("generating_report", query=query[:50], claim_count=len(verification_results))
    
    start_time = time()
    response = await llm.ainvoke(messages)
    latency_ms = (time() - start_time) * 1000
    
    response_text = response.content
    if isinstance(response_text, list):
        response_text = "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in response_text
        )
    
    # Update stats with this LLM call
    stats["total_latency_ms"] += latency_ms
    stats["llm_calls"] += 1
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        stats["total_tokens"] += response.usage_metadata.get('total_tokens', 0)
    
    report = _parse_report_response(
        str(response_text),
        query,
        citations,
        stats,
    )
    
    logger.info(
        "report_generated",
        title=report.title[:50],
        section_count=len(report.sections),
        citation_count=len(report.citations),
        latency_ms=round(latency_ms, 2),
    )
    
    return report


async def synthesize_node(state: AgentState) -> dict:
    """LangGraph node function for the Synthesizer agent.
    
    This function takes verified claims and produces a structured report by:
    1. Filtering to include only supported claims (with caveats for others)
    2. Building a citation list from source URLs
    3. Using Claude to generate a structured report
    4. Formatting the report with inline citations
    5. Adding trace events for observability
    
    Args:
        state: Current agent state with verification results.
        
    Returns:
        Dict with state updates (report, report_structured, citations, agent_trace).
    """
    verification_results = state.get("verification_results", [])
    source_scores = state.get("source_scores", {})
    query = state.get("query", "")
    agent_trace = state.get("agent_trace", []).copy()
    
    # Add starting trace event
    agent_trace.append(TraceEvent(
        agent=AgentName.SYNTHESIZER,
        action="start",
        detail=f"Starting synthesis of {len(verification_results)} verification results",
        metadata={"result_count": len(verification_results)},
    ))
    
    if not verification_results:
        logger.warning("synthesize_node_no_results")
        
        # Generate a minimal report indicating no findings
        empty_report = Report(
            title=f"Research Report: {query[:50]}",
            summary="No verified claims were available to synthesize into a report.",
            sections=[ReportSection(
                heading="No Results",
                content="The research process did not produce any verified claims. "
                        "This may be due to search limitations or the specificity of the query.",
                order=1,
            )],
            confidence_summary="Unable to assess confidence due to lack of verified claims.",
            citations=[],
            metadata=ReportMetadata(query=query),
        )
        
        agent_trace.append(TraceEvent(
            agent=AgentName.SYNTHESIZER,
            action="skip",
            detail="No verification results to synthesize",
        ))
        
        return {
            "report": _format_report_as_text(empty_report),
            "report_structured": empty_report,
            "citations": [],
            "status": PipelineStatus.COMPLETED.value,
            "agent_trace": agent_trace,
        }
    
    try:
        llm = _create_llm()
        
        # Generate the report
        report_start = time()
        report = await generate_report(
            query=query,
            verification_results=verification_results,
            source_scores=source_scores,
            state=state,
            llm=llm,
        )
        report_latency = (time() - report_start) * 1000
        
        # Format as text
        report_text = _format_report_as_text(report)
        
        agent_trace.append(TraceEvent(
            agent=AgentName.SYNTHESIZER,
            action="report_generated",
            detail=f"Generated report: {report.title[:50]}",
            latency_ms=round(report_latency, 2),
            metadata={
                "title": report.title,
                "section_count": len(report.sections),
                "citation_count": len(report.citations),
            },
        ))
        
        # Summary statistics
        supported_count = sum(
            1 for r in verification_results
            if r.verdict == VerificationVerdict.SUPPORTED
        )
        
        agent_trace.append(TraceEvent(
            agent=AgentName.SYNTHESIZER,
            action="complete",
            detail=f"Synthesis complete: {len(report.sections)} sections, "
                   f"{len(report.citations)} citations, "
                   f"{supported_count}/{len(verification_results)} supported claims",
            metadata={
                "sections": len(report.sections),
                "citations": len(report.citations),
                "supported_claims": supported_count,
                "total_claims": len(verification_results),
            },
        ))
        
        logger.info(
            "synthesis_complete",
            title=report.title[:50],
            sections=len(report.sections),
            citations=len(report.citations),
        )
        
        return {
            "report": report_text,
            "report_structured": report,
            "citations": report.citations,
            "status": PipelineStatus.COMPLETED.value,
            "agent_trace": agent_trace,
        }
        
    except Exception as e:
        logger.exception("synthesis_error", error=str(e))
        
        agent_trace.append(TraceEvent(
            agent=AgentName.SYNTHESIZER,
            action="error",
            detail=f"Synthesis failed: {str(e)}",
            metadata={"error": str(e)},
        ))
        
        # Return error state
        return {
            "report": f"Error generating report: {str(e)}",
            "report_structured": Report(
                title="Error",
                summary=f"Report generation failed: {str(e)}",
                sections=[],
                confidence_summary="",
                citations=[],
                metadata=ReportMetadata(query=query),
            ),
            "citations": [],
            "status": PipelineStatus.FAILED.value,
            "errors": state.get("errors", []) + [f"Synthesis error: {str(e)}"],
            "agent_trace": agent_trace,
        }
