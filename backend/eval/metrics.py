"""Automated quality metrics for MARSA evaluation.

This module provides functions to compute various quality metrics
for evaluating research report quality, source diversity, fact-checking
accuracy, and system performance.

Metrics implemented:
- citation_accuracy: What % of claims have valid source URLs?
- source_diversity: How many unique domains are cited?
- fact_check_recall: For queries with known false claims, what % were caught?
- latency_p50_p95: Median and 95th percentile latency
- token_efficiency: Tokens per quality point (lower is better)
"""

import re
import statistics
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field


class CitationMetrics(BaseModel):
    """Metrics related to citation quality."""
    
    total_citations: int = Field(default=0, description="Total number of citations")
    valid_urls: int = Field(default=0, description="Citations with valid URLs")
    accuracy: float = Field(default=0.0, description="Percentage of valid citations (0.0-1.0)")
    unique_domains: int = Field(default=0, description="Number of unique domains cited")
    domain_list: list[str] = Field(default_factory=list, description="List of unique domains")


class FactCheckMetrics(BaseModel):
    """Metrics related to fact-checking performance."""
    
    total_claims: int = Field(default=0, description="Total claims fact-checked")
    supported_claims: int = Field(default=0, description="Claims marked as supported")
    contradicted_claims: int = Field(default=0, description="Claims marked as contradicted")
    unverifiable_claims: int = Field(default=0, description="Claims marked as unverifiable")
    pass_rate: float = Field(default=0.0, description="Percentage of claims that passed (0.0-1.0)")
    false_premise_caught: Optional[bool] = Field(
        default=None, 
        description="For false premise queries, whether the false claim was caught"
    )


class LatencyMetrics(BaseModel):
    """Metrics related to system latency."""
    
    total_ms: float = Field(default=0.0, description="Total end-to-end latency in ms")
    p50_ms: float = Field(default=0.0, description="Median latency in ms")
    p95_ms: float = Field(default=0.0, description="95th percentile latency in ms")
    planning_ms: float = Field(default=0.0, description="Planning phase latency")
    research_ms: float = Field(default=0.0, description="Research phase latency")
    fact_check_ms: float = Field(default=0.0, description="Fact-checking phase latency")
    synthesis_ms: float = Field(default=0.0, description="Synthesis phase latency")


class TokenMetrics(BaseModel):
    """Metrics related to token usage."""
    
    total_tokens: int = Field(default=0, description="Total tokens consumed")
    llm_calls: int = Field(default=0, description="Number of LLM calls made")
    tokens_per_call: float = Field(default=0.0, description="Average tokens per LLM call")
    efficiency_score: float = Field(
        default=0.0, 
        description="Tokens per quality point (lower is better)"
    )


class QualityScores(BaseModel):
    """LLM-evaluated quality scores (1-5 scale)."""
    
    relevance: int = Field(default=0, ge=0, le=5, description="How well report addresses query")
    accuracy: int = Field(default=0, ge=0, le=5, description="Factual correctness of claims")
    completeness: int = Field(default=0, ge=0, le=5, description="Coverage of important aspects")
    citation_quality: int = Field(default=0, ge=0, le=5, description="Quality of source citations")
    overall: float = Field(default=0.0, description="Weighted average of all scores")


class EvaluationResult(BaseModel):
    """Complete evaluation result for a single query."""
    
    query_id: str = Field(description="Unique identifier for the query")
    query: str = Field(description="The original query text")
    category: str = Field(description="Query category (factual, comparison, etc.)")
    success: bool = Field(default=True, description="Whether the pipeline completed successfully")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Metrics
    citation_metrics: CitationMetrics = Field(default_factory=CitationMetrics)
    fact_check_metrics: FactCheckMetrics = Field(default_factory=FactCheckMetrics)
    latency_metrics: LatencyMetrics = Field(default_factory=LatencyMetrics)
    token_metrics: TokenMetrics = Field(default_factory=TokenMetrics)
    quality_scores: QualityScores = Field(default_factory=QualityScores)
    
    # Expectations
    expected_sub_queries_min: int = Field(default=1)
    actual_sub_queries: int = Field(default=0)
    expected_sources_min: int = Field(default=3)
    actual_sources: int = Field(default=0)
    must_mention_found: list[str] = Field(default_factory=list)
    must_mention_missing: list[str] = Field(default_factory=list)
    must_not_claim_violations: list[str] = Field(default_factory=list)


class AggregateMetrics(BaseModel):
    """Aggregate metrics across all evaluation queries."""
    
    total_queries: int = Field(default=0)
    successful_queries: int = Field(default=0)
    failed_queries: int = Field(default=0)
    
    # Averages
    avg_citation_accuracy: float = Field(default=0.0)
    avg_source_diversity: float = Field(default=0.0)
    avg_fact_check_pass_rate: float = Field(default=0.0)
    
    # Latency
    latency_p50_ms: float = Field(default=0.0)
    latency_p95_ms: float = Field(default=0.0)
    latency_avg_ms: float = Field(default=0.0)
    
    # Tokens
    total_tokens: int = Field(default=0)
    total_llm_calls: int = Field(default=0)
    avg_tokens_per_query: float = Field(default=0.0)
    token_efficiency: float = Field(default=0.0)
    
    # Quality scores
    avg_relevance: float = Field(default=0.0)
    avg_accuracy: float = Field(default=0.0)
    avg_completeness: float = Field(default=0.0)
    avg_citation_quality: float = Field(default=0.0)
    avg_overall_quality: float = Field(default=0.0)
    
    # False premise detection
    false_premise_recall: float = Field(default=0.0)
    
    # By category
    metrics_by_category: dict[str, dict] = Field(default_factory=dict)


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid and accessible.
    
    Args:
        url: URL string to validate.
        
    Returns:
        True if URL is valid, False otherwise.
    """
    if not url:
        return False
    
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def extract_domain(url: str) -> Optional[str]:
    """Extract the domain from a URL.
    
    Args:
        url: URL string.
        
    Returns:
        Domain string or None if invalid.
    """
    try:
        result = urlparse(url)
        domain = result.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain if domain else None
    except Exception:
        return None


def calculate_citation_metrics(
    citations: list[dict],
    report_text: str = "",
) -> CitationMetrics:
    """Calculate citation-related metrics.
    
    Args:
        citations: List of citation dictionaries with 'url' field.
        report_text: The report text to check for inline citations.
        
    Returns:
        CitationMetrics with calculated values.
    """
    if not citations:
        return CitationMetrics()
    
    total = len(citations)
    valid_count = 0
    domains = set()
    
    for citation in citations:
        url = citation.get("url", "") if isinstance(citation, dict) else getattr(citation, "url", "")
        if is_valid_url(url):
            valid_count += 1
            domain = extract_domain(url)
            if domain:
                domains.add(domain)
    
    accuracy = valid_count / total if total > 0 else 0.0
    
    return CitationMetrics(
        total_citations=total,
        valid_urls=valid_count,
        accuracy=accuracy,
        unique_domains=len(domains),
        domain_list=sorted(list(domains)),
    )


def calculate_fact_check_metrics(
    verification_results: list[dict],
    false_premise_claim: Optional[str] = None,
) -> FactCheckMetrics:
    """Calculate fact-checking metrics.
    
    Args:
        verification_results: List of verification result dictionaries.
        false_premise_claim: The false claim to check for (if this is a false premise query).
        
    Returns:
        FactCheckMetrics with calculated values.
    """
    if not verification_results:
        return FactCheckMetrics()
    
    total = len(verification_results)
    supported = 0
    contradicted = 0
    unverifiable = 0
    
    for result in verification_results:
        if isinstance(result, dict):
            verdict = result.get("verdict", "")
        else:
            verdict = getattr(result, "verdict", "")
            if hasattr(verdict, "value"):
                verdict = verdict.value
        
        verdict = str(verdict).lower()
        
        if verdict == "supported":
            supported += 1
        elif verdict == "contradicted":
            contradicted += 1
        elif verdict == "unverifiable":
            unverifiable += 1
    
    pass_rate = supported / total if total > 0 else 0.0
    
    # Check if false premise was caught
    false_premise_caught = None
    if false_premise_claim:
        # Check if any claim was contradicted that relates to the false premise
        for result in verification_results:
            claim_text = ""
            if isinstance(result, dict):
                claim = result.get("claim", {})
                claim_text = claim.get("statement", "") if isinstance(claim, dict) else str(claim)
                verdict = result.get("verdict", "")
            else:
                claim = getattr(result, "claim", None)
                claim_text = getattr(claim, "statement", "") if claim else ""
                verdict = getattr(result, "verdict", "")
            
            if hasattr(verdict, "value"):
                verdict = verdict.value
            
            # Check if this claim relates to the false premise
            if _text_similarity(claim_text.lower(), false_premise_claim.lower()) > 0.3:
                false_premise_caught = str(verdict).lower() == "contradicted"
                break
    
    return FactCheckMetrics(
        total_claims=total,
        supported_claims=supported,
        contradicted_claims=contradicted,
        unverifiable_claims=unverifiable,
        pass_rate=pass_rate,
        false_premise_caught=false_premise_caught,
    )


def _text_similarity(text1: str, text2: str) -> float:
    """Simple word overlap similarity between two texts.
    
    Args:
        text1: First text.
        text2: Second text.
        
    Returns:
        Similarity score between 0 and 1.
    """
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def calculate_latency_metrics(
    trace_events: list[dict],
    total_latency_ms: float = 0.0,
) -> LatencyMetrics:
    """Calculate latency metrics from trace events.
    
    Args:
        trace_events: List of trace event dictionaries.
        total_latency_ms: Total end-to-end latency in ms.
        
    Returns:
        LatencyMetrics with calculated values.
    """
    if not trace_events:
        return LatencyMetrics(total_ms=total_latency_ms)
    
    phase_latencies = {
        "planner": 0.0,
        "researcher": 0.0,
        "fact_checker": 0.0,
        "synthesizer": 0.0,
    }
    
    all_latencies = []
    
    for event in trace_events:
        if isinstance(event, dict):
            agent = event.get("agent", "")
            latency = event.get("latency_ms", 0) or 0
        else:
            agent = getattr(event, "agent", "")
            latency = getattr(event, "latency_ms", 0) or 0
        
        if hasattr(agent, "value"):
            agent = agent.value
        
        agent = str(agent).lower()
        
        if latency > 0:
            all_latencies.append(latency)
            if agent in phase_latencies:
                phase_latencies[agent] += latency
    
    return LatencyMetrics(
        total_ms=total_latency_ms,
        p50_ms=statistics.median(all_latencies) if all_latencies else 0.0,
        p95_ms=statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 20 else max(all_latencies, default=0.0),
        planning_ms=phase_latencies["planner"],
        research_ms=phase_latencies["researcher"],
        fact_check_ms=phase_latencies["fact_checker"],
        synthesis_ms=phase_latencies["synthesizer"],
    )


def calculate_token_metrics(
    trace_events: list[dict],
    quality_score: float = 0.0,
) -> TokenMetrics:
    """Calculate token usage metrics.
    
    Args:
        trace_events: List of trace event dictionaries.
        quality_score: Overall quality score for efficiency calculation.
        
    Returns:
        TokenMetrics with calculated values.
    """
    total_tokens = 0
    llm_calls = 0
    
    for event in trace_events:
        if isinstance(event, dict):
            action = event.get("action", "")
            tokens = event.get("tokens_used", 0) or 0
        else:
            action = getattr(event, "action", "")
            tokens = getattr(event, "tokens_used", 0) or 0
        
        if tokens > 0:
            total_tokens += tokens
            if "llm" in str(action).lower():
                llm_calls += 1
    
    tokens_per_call = total_tokens / llm_calls if llm_calls > 0 else 0.0
    efficiency = total_tokens / quality_score if quality_score > 0 else float('inf')
    
    return TokenMetrics(
        total_tokens=total_tokens,
        llm_calls=llm_calls,
        tokens_per_call=tokens_per_call,
        efficiency_score=efficiency if efficiency != float('inf') else 0.0,
    )


def check_must_mention(
    report_text: str,
    must_mention: list[str],
) -> tuple[list[str], list[str]]:
    """Check which required terms are mentioned in the report.
    
    Args:
        report_text: The report text to search.
        must_mention: List of terms that must be mentioned.
        
    Returns:
        Tuple of (found_terms, missing_terms).
    """
    report_lower = report_text.lower()
    found = []
    missing = []
    
    for term in must_mention:
        if term.lower() in report_lower:
            found.append(term)
        else:
            missing.append(term)
    
    return found, missing


def check_must_not_claim(
    report_text: str,
    must_not_claim: list[str],
) -> list[str]:
    """Check if any forbidden claims appear in the report.
    
    Args:
        report_text: The report text to search.
        must_not_claim: List of claims that must NOT appear.
        
    Returns:
        List of violations found.
    """
    report_lower = report_text.lower()
    violations = []
    
    for claim in must_not_claim:
        # Use fuzzy matching - if >60% of words from claim appear in a sentence
        claim_words = set(re.findall(r'\w+', claim.lower()))
        if not claim_words:
            continue
        
        # Split report into sentences
        sentences = re.split(r'[.!?]', report_lower)
        for sentence in sentences:
            sentence_words = set(re.findall(r'\w+', sentence))
            overlap = len(claim_words & sentence_words) / len(claim_words)
            if overlap > 0.6:
                violations.append(claim)
                break
    
    return violations


def calculate_aggregate_metrics(results: list[EvaluationResult]) -> AggregateMetrics:
    """Calculate aggregate metrics across all evaluation results.
    
    Args:
        results: List of individual evaluation results.
        
    Returns:
        AggregateMetrics with calculated values.
    """
    if not results:
        return AggregateMetrics()
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    # Collect latencies for percentile calculation
    all_latencies = [r.latency_metrics.total_ms for r in successful if r.latency_metrics.total_ms > 0]
    
    # Calculate false premise recall
    false_premise_queries = [
        r for r in successful 
        if r.category == "false_premise" and r.fact_check_metrics.false_premise_caught is not None
    ]
    false_premise_caught = sum(1 for r in false_premise_queries if r.fact_check_metrics.false_premise_caught)
    false_premise_recall = (
        false_premise_caught / len(false_premise_queries) 
        if false_premise_queries else 0.0
    )
    
    # Calculate by category
    categories = set(r.category for r in results)
    metrics_by_category = {}
    
    for category in categories:
        cat_results = [r for r in successful if r.category == category]
        if not cat_results:
            continue
        
        metrics_by_category[category] = {
            "count": len(cat_results),
            "avg_latency_ms": statistics.mean([r.latency_metrics.total_ms for r in cat_results]) if cat_results else 0,
            "avg_quality": statistics.mean([r.quality_scores.overall for r in cat_results]) if cat_results else 0,
            "avg_citation_accuracy": statistics.mean([r.citation_metrics.accuracy for r in cat_results]) if cat_results else 0,
        }
    
    # Calculate totals
    total_tokens = sum(r.token_metrics.total_tokens for r in successful)
    total_llm_calls = sum(r.token_metrics.llm_calls for r in successful)
    avg_quality = statistics.mean([r.quality_scores.overall for r in successful]) if successful else 0
    
    return AggregateMetrics(
        total_queries=len(results),
        successful_queries=len(successful),
        failed_queries=len(failed),
        avg_citation_accuracy=statistics.mean([r.citation_metrics.accuracy for r in successful]) if successful else 0,
        avg_source_diversity=statistics.mean([r.citation_metrics.unique_domains for r in successful]) if successful else 0,
        avg_fact_check_pass_rate=statistics.mean([r.fact_check_metrics.pass_rate for r in successful]) if successful else 0,
        latency_p50_ms=statistics.median(all_latencies) if all_latencies else 0,
        latency_p95_ms=statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 20 else max(all_latencies, default=0),
        latency_avg_ms=statistics.mean(all_latencies) if all_latencies else 0,
        total_tokens=total_tokens,
        total_llm_calls=total_llm_calls,
        avg_tokens_per_query=total_tokens / len(successful) if successful else 0,
        token_efficiency=total_tokens / avg_quality if avg_quality > 0 else 0,
        avg_relevance=statistics.mean([r.quality_scores.relevance for r in successful]) if successful else 0,
        avg_accuracy=statistics.mean([r.quality_scores.accuracy for r in successful]) if successful else 0,
        avg_completeness=statistics.mean([r.quality_scores.completeness for r in successful]) if successful else 0,
        avg_citation_quality=statistics.mean([r.quality_scores.citation_quality for r in successful]) if successful else 0,
        avg_overall_quality=avg_quality,
        false_premise_recall=false_premise_recall,
        metrics_by_category=metrics_by_category,
    )


def latency_p50_p95(latencies: list[float]) -> tuple[float, float]:
    """Calculate the p50 and p95 latency values.
    
    Args:
        latencies: List of latency values in milliseconds.
        
    Returns:
        Tuple of (p50, p95) values.
    """
    if not latencies:
        return 0.0, 0.0
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    # P50 (median)
    if n % 2 == 0:
        p50 = (sorted_latencies[n // 2 - 1] + sorted_latencies[n // 2]) / 2
    else:
        p50 = sorted_latencies[n // 2]
    
    # P95
    p95_idx = int(n * 0.95)
    p95 = sorted_latencies[min(p95_idx, n - 1)]
    
    return p50, p95
