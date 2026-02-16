"""Tavily Search MCP Server.

Provides web search capabilities through the Tavily API via MCP protocol.
Includes production-grade error handling, retry logic, validation, and rate limiting.
"""

import sys
import time
from pathlib import Path
from typing import Optional

# Add backend directory to path for config import
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from fastmcp import FastMCP  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from tavily import TavilyClient  # noqa: E402
from tenacity import (  # noqa: E402
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import Config  # noqa: E402
from utils.rate_limiter import get_rate_limiter  # noqa: E402
from utils.resilience import (  # noqa: E402
    DEFAULT_SEARCH_TIMEOUT,
    ExternalAPIError,
    RateLimitExceededError,
    SearchTimeoutError,
    ValidationError,
    get_logger,
    validate_max_results,
    validate_query,
)

# Initialize configuration
config = Config()

# Initialize structured logging
logger = get_logger("mcp.tavily_search")

# Initialize MCP server
mcp = FastMCP("tavily-search-server")

# Initialize rate limiter for Tavily API (1,000/month free tier)
_rate_limiter = get_rate_limiter("tavily", monthly_limit=1000)

# Lazy-load Tavily client to avoid initialization during test collection
_tavily_client: Optional[TavilyClient] = None


def get_tavily_client() -> TavilyClient:
    """Get or create the Tavily client instance."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=config.tavily_api_key)
    return _tavily_client


class SearchResult(BaseModel):
    """A single search result from Tavily."""
    
    title: str = Field(description="Title of the web page")
    url: str = Field(description="URL of the web page")
    content: str = Field(description="Relevant content snippet from the page")
    score: float = Field(description="Relevance score (0.0 to 1.0)")
    published_date: Optional[str] = Field(
        default=None,
        description="Published date of the content (ISO format), if available"
    )


@mcp.tool()
def search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the web using Tavily API.
    
    Args:
        query: The search query string (max 500 characters)
        max_results: Maximum number of results to return (1-20, default: 5)
        
    Returns:
        List of search results with title, url, content, score, and published_date
        
    Raises:
        ValidationError: If query or max_results are invalid
        RateLimitExceededError: If monthly API limit is reached
        ExternalAPIError: If the Tavily API call fails
        SearchTimeoutError: If the search times out
    """
    return search_impl(query, max_results)


# Retry decorator for transient failures
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    reraise=True,
)
def _call_tavily_api(client: TavilyClient, query: str, max_results: int) -> dict:
    """Make the actual Tavily API call with retry logic.
    
    Args:
        client: The Tavily client instance
        query: The search query
        max_results: Maximum results to return
        
    Returns:
        Raw API response dictionary
    """
    return client.search(
        query=query,
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )


def search_impl(query: str, max_results: int = 5) -> list[SearchResult]:
    """Implementation of web search using Tavily API.
    
    Args:
        query: The search query string (max 500 characters)
        max_results: Maximum number of results to return (1-20, default: 5)
        
    Returns:
        List of search results with title, url, content, score, and published_date
        
    Raises:
        ValidationError: If query or max_results are invalid
        RateLimitExceededError: If monthly API limit is reached
        ExternalAPIError: If the Tavily API call fails
        SearchTimeoutError: If the search times out
    """
    start_time = time.perf_counter()
    
    # Input validation
    try:
        validated_query = validate_query(query)
        validated_max_results = validate_max_results(max_results)
    except ValidationError:
        logger.warning(
            "search_validation_failed",
            query_length=len(query) if query else 0,
            max_results=max_results,
        )
        raise
    
    # Check rate limits before making the call
    try:
        rate_status = _rate_limiter.check_limit()
        logger.debug(
            "rate_limit_status",
            remaining=rate_status["remaining"],
            usage_percent=rate_status["usage_percent"],
        )
    except RateLimitExceededError:
        logger.error(
            "search_rate_limit_exceeded",
            query=validated_query[:100],
            current_usage=_rate_limiter.current_count,
            limit=_rate_limiter.monthly_limit,
        )
        raise
    
    logger.info(
        "search_started",
        query=validated_query[:100],
        max_results=validated_max_results,
    )
    
    # Make the API call with retry logic
    try:
        response = _call_tavily_api(
            get_tavily_client(),
            validated_query,
            validated_max_results
        )
    except TimeoutError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "search_timeout",
            query=validated_query[:100],
            timeout_seconds=DEFAULT_SEARCH_TIMEOUT,
            elapsed_ms=round(elapsed_ms, 2),
        )
        raise SearchTimeoutError(
            f"Search timed out after {DEFAULT_SEARCH_TIMEOUT} seconds",
            details={"query": validated_query, "timeout": DEFAULT_SEARCH_TIMEOUT}
        ) from e
    except (ConnectionError, OSError) as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "search_connection_error",
            query=validated_query[:100],
            error_type=type(e).__name__,
            error_message=str(e),
            elapsed_ms=round(elapsed_ms, 2),
        )
        raise ExternalAPIError(
            f"Failed to connect to Tavily API: {e}",
            details={"query": validated_query, "error": str(e)}
        ) from e
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "search_api_error",
            query=validated_query[:100],
            error_type=type(e).__name__,
            error_message=str(e),
            elapsed_ms=round(elapsed_ms, 2),
        )
        raise ExternalAPIError(
            f"Tavily API error: {e}",
            details={"query": validated_query, "error": str(e)}
        ) from e
    
    # Increment rate limit counter after successful call
    _rate_limiter.increment()
    
    # Transform Tavily response to our SearchResult format
    results = []
    for result in response.get("results", []):
        search_result = SearchResult(
            title=result.get("title", ""),
            url=result.get("url", ""),
            content=result.get("content", ""),
            score=result.get("score", 0.0),
            published_date=result.get("published_date")
        )
        results.append(search_result)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "search_completed",
        query=validated_query[:100],
        result_count=len(results),
        latency_ms=round(elapsed_ms, 2),
    )
    
    return results


@mcp.tool()
def get_rate_limit_status() -> dict:
    """Get current rate limit status for the Tavily API.
    
    Returns:
        Dictionary with current usage, remaining requests, and period info
    """
    return _rate_limiter.get_status()


if __name__ == "__main__":
    mcp.run()