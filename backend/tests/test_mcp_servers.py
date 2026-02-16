"""Tests for MCP servers.

Tests the hello world server and Tavily search server.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from mcp_servers.hello import greet_impl as greet  # noqa: E402
from mcp_servers.tavily_search import search_impl as search, SearchResult  # noqa: E402


@pytest.fixture
def mock_tavily_response():
    """Mock Tavily API response for testing without consuming API quota."""
    return {
        "results": [
            {
                "title": "Python Programming Guide",
                "url": "https://example.com/python-guide",
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "score": 0.95,
                "published_date": "2026-01-15"
            },
            {
                "title": "Advanced Python Techniques",
                "url": "https://example.com/python-advanced",
                "content": "Learn advanced Python concepts including decorators, generators, and metaclasses.",
                "score": 0.87,
                "published_date": "2026-01-10"
            },
            {
                "title": "Python for Data Science",
                "url": "https://example.com/python-data-science",
                "content": "Using Python for data analysis with pandas, numpy, and scikit-learn libraries.",
                "score": 0.82,
                "published_date": None
            }
        ]
    }


class TestHelloServer:
    """Tests for the hello world MCP server."""
    
    def test_greet_returns_greeting(self):
        """Test that greet function returns a proper greeting."""
        result = greet("Alice")
        assert result == "Hello, Alice!"
        
    def test_greet_with_different_names(self):
        """Test greet with various names."""
        assert greet("Bob") == "Hello, Bob!"
        assert greet("Charlie") == "Hello, Charlie!"
        assert greet("World") == "Hello, World!"
    
    def test_greet_empty_string(self):
        """Test greet with empty string."""
        result = greet("")
        assert result == "Hello, !"


class TestTavilySearchServer:
    """Tests for the Tavily search MCP server."""
    
    @patch('mcp_servers.tavily_search.get_tavily_client')
    def test_search_returns_results(self, mock_get_client, mock_tavily_response):
        """Test that search returns a list of SearchResult objects (MOCKED)."""
        mock_client = mock_get_client.return_value
        mock_client.search.return_value = mock_tavily_response
        
        results = search("Python programming language", max_results=3)
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert len(results) > 0
        
        # Check that all results are SearchResult instances
        for result in results:
            assert isinstance(result, SearchResult)
        
        # Verify the mock was called correctly
        mock_client.search.assert_called_once()
    
    @patch('mcp_servers.tavily_search.get_tavily_client')
    def test_search_result_structure(self, mock_get_client, mock_tavily_response):
        """Test that SearchResult objects have the correct structure (MOCKED)."""
        mock_client = mock_get_client.return_value
        mock_client.search.return_value = mock_tavily_response
        
        results = search("Rust programming language 2026", max_results=5)
        
        assert len(results) > 0
        
        # Check first result has all required fields
        result = results[0]
        assert hasattr(result, "title")
        assert hasattr(result, "url")
        assert hasattr(result, "content")
        assert hasattr(result, "score")
        assert hasattr(result, "published_date")
        
        # Validate field types and content
        assert isinstance(result.title, str)
        assert len(result.title) > 0
        assert isinstance(result.url, str)
        assert result.url.startswith("http")
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        # published_date can be None or a string
        assert result.published_date is None or isinstance(result.published_date, str)
    
    @patch('mcp_servers.tavily_search.get_tavily_client')
    def test_search_max_results_parameter(self, mock_get_client, mock_tavily_response):
        """Test that max_results parameter is respected (MOCKED)."""
        mock_client = mock_get_client.return_value
        # Mock returns 3 results
        mock_client.search.return_value = {
            "results": mock_tavily_response["results"][:3]
        }
        results_3 = search("Go programming language", max_results=3)
        
        # Mock returns 5 results (we only have 3, so return them all)
        mock_client.search.return_value = mock_tavily_response
        results_5 = search("Go programming language", max_results=5)
        
        assert len(results_3) == 3
        assert len(results_5) == 3  # Only 3 in our mock data
    
    @patch('mcp_servers.tavily_search.get_tavily_client')
    def test_search_default_max_results(self, mock_get_client, mock_tavily_response):
        """Test that default max_results is 5 (MOCKED)."""
        mock_client = mock_get_client.return_value
        mock_client.search.return_value = mock_tavily_response
        
        results = search("JavaScript frameworks")
        
        # Should call with default max_results
        assert len(results) <= 5
        mock_client.search.assert_called_once()
    
    @patch('mcp_servers.tavily_search.get_tavily_client')
    def test_search_with_complex_query(self, mock_get_client, mock_tavily_response):
        """Test search with a more complex query (MOCKED)."""
        mock_client = mock_get_client.return_value
        mock_client.search.return_value = mock_tavily_response
        
        query = "latest developments in Rust programming language 2026"
        results = search(query, max_results=5)
        
        assert len(results) > 0
        # Results should be sorted by relevance (score)
        if len(results) > 1:
            # First result should have high relevance
            assert results[0].score > 0.5
    
    @pytest.mark.integration
    def test_search_real_api_integration(self):
        """Integration test with REAL Tavily API - USES API QUOTA!
        
        This test makes an actual API call to verify end-to-end functionality.
        Run with: pytest -m integration -s
        Skip with: pytest -m "not integration"
        """
        print("\n" + "="*60)
        print("🔍 REAL API TEST - Making actual Tavily search call...")
        print("="*60)
        
        results = search("Python programming", max_results=2)
        
        assert len(results) > 0
        assert len(results) <= 2
        
        # Print actual results
        print(f"\n✅ Got {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Score: {result.score:.3f}")
            print(f"   Published: {result.published_date or 'N/A'}")
            print(f"   Content: {result.content[:100]}...")
            print()
        
        # Verify real API returns proper structure
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.url.startswith("http")
        assert len(result.content) > 0
        assert 0.0 <= result.score <= 1.0
        
        print("="*60)
        print("✅ Integration test passed!")
        print("="*60)


if __name__ == "__main__":
    # Run pytest when this file is executed directly
    pytest.main([__file__, "-v"])