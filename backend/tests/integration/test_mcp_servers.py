"""Integration tests for MCP servers using real APIs.

These tests make actual API calls and should be run separately from unit tests.
"""

import sys
from pathlib import Path

import pytest

# Add backend directory to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from mcp_servers.tavily_search import search_impl as search, SearchResult  # noqa: E402
from mcp_servers.document_store import (  # noqa: E402
    ingest_document_impl,
    search_documents_impl,
    list_documents_impl,
    reset_collection,
)


class TestTavilySearchIntegration:
    """Integration tests for Tavily Search MCP server."""
    
    def test_search_real_api(self):
        """Integration test with REAL Tavily API - USES API QUOTA!
        
        This test makes an actual API call to verify end-to-end functionality.
        """
        print("\n" + "="*60)
        print("REAL API TEST - Making actual Tavily search call...")
        print("="*60)
        
        results = search("Python programming", max_results=2)
        
        assert len(results) > 0
        assert len(results) <= 2
        
        # Print actual results
        print(f"\nGot {len(results)} results:\n")
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
        print("Integration test passed!")
        print("="*60)


class TestDocumentStoreIntegration:
    """Integration tests for Document Store MCP server."""
    
    def test_document_store_real_embeddings(self):
        """Integration test with REAL OpenAI embeddings - USES API QUOTA!
        
        This test makes actual OpenAI API calls for embeddings.
        """
        print("\n" + "="*60)
        print("REAL API TEST - Using actual OpenAI embeddings...")
        print("="*60)
        
        # Reset collection for clean test
        reset_collection()
        
        # Ingest test document
        title = "Go Concurrency Integration Test"
        content = """
        Go provides goroutines and channels as first-class concurrency primitives.
        Goroutines are lightweight threads managed by the Go runtime, not the OS.
        Channels allow goroutines to communicate safely without explicit locking.
        The select statement enables waiting on multiple channel operations.
        This makes concurrent programming in Go simpler and less error-prone
        compared to traditional threading models.
        """
        source = "https://go.dev/concurrency-test"
        
        result = ingest_document_impl(title, content.strip(), source)
        print(f"\nIngested: {result}")
        
        # Search for relevant content
        query = "Go concurrency patterns"
        results = search_documents_impl(query, n_results=3)
        
        print(f"\nSearch results for '{query}':")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. Score: {r.relevance_score:.4f}")
            print(f"   Source: {r.source}")
            print(f"   Content: {r.content[:100]}...")
        
        # Verify results
        assert len(results) >= 1
        # Embedding similarity can vary across models and updates; keep a tolerant floor.
        assert results[0].relevance_score > 0.35
        assert "Go" in results[0].content or "goroutine" in results[0].content.lower()
        
        # List documents
        docs = list_documents_impl()
        print(f"\nTotal documents in store: {len(docs)}")
        
        print("\n" + "="*60)
        print("Integration test passed!")
        print("="*60)


if __name__ == "__main__":
    # Run pytest when this file is executed directly
    pytest.main([__file__, "-v", "-s"])
