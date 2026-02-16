"""Tavily Search MCP Server.

Provides web search capabilities through the Tavily API via MCP protocol.
"""

import sys
from pathlib import Path
from typing import Optional

# Add backend directory to path for config import
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from tavily import TavilyClient

from config import Config

# Initialize configuration
config = Config()

# Initialize Tavily client
tavily_client = TavilyClient(api_key=config.tavily_api_key)

# Initialize MCP server
mcp = FastMCP("tavily-search-server")


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
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        List of search results with title, url, content, score, and published_date
    """
    return search_impl(query, max_results)


def search_impl(query: str, max_results: int = 5) -> list[SearchResult]:
    """Implementation of web search using Tavily API.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        List of search results with title, url, content, score, and published_date
    """
    # Call Tavily API
    response = tavily_client.search(
        query=query,
        max_results=max_results,
        include_answer=False,  # We don't need the AI-generated answer
        include_raw_content=False,  # Don't need full HTML
        include_images=False  # Don't need images
    )
    
    # Transform Tavily response to our SearchResult format
    results = []
    for result in response.get("results", []):
        search_result = SearchResult(
            title=result.get("title", ""),
            url=result.get("url", ""),
            content=result.get("content", ""),
            score=result.get("score", 0.0),
            published_date=result.get("published_date")  # May be None
        )
        results.append(search_result)
    
    return results


if __name__ == "__main__":
    mcp.run()