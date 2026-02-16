"""Tests for MCP servers.

Tests the hello world server, Tavily search server, and Document Store server.
"""

import sys
import tempfile
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


class TestDocumentStoreServer:
    """Tests for the Document Store MCP server."""
    
    @pytest.fixture(autouse=True)
    def setup_test_collection(self):
        """Set up a test ChromaDB collection for each test."""
        # Import after path setup
        from mcp_servers import document_store
        
        # Create a temp directory for test ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch the CHROMADB_PATH to use temp directory
        self._original_path = document_store.CHROMADB_PATH
        document_store.CHROMADB_PATH = Path(self.temp_dir) / "chromadb"
        document_store.CHROMADB_PATH.mkdir(parents=True, exist_ok=True)
        
        # Reset the cached clients
        document_store._chroma_client = None
        document_store._collection = None
        
        yield
        
        # Cleanup - reset clients
        document_store._chroma_client = None
        document_store._collection = None
        document_store.CHROMADB_PATH = self._original_path
    
    @patch('mcp_servers.document_store.get_embedding')
    def test_ingest_document_creates_chunks(self, mock_embedding):
        """Test that ingesting a document creates proper chunks."""
        from mcp_servers.document_store import (
            ingest_document_impl,
            list_documents_impl,
            get_collection,
        )
        
        # Mock embedding to return consistent vector
        mock_embedding.return_value = [0.1] * 1536
        
        title = "Test Document"
        content = "This is a test document. " * 100  # Create substantial content
        source_url = "https://example.com/test"
        
        result = ingest_document_impl(title, content, source_url)
        
        assert "Successfully ingested" in result
        assert title in result
        
        # Verify document was stored
        docs = list_documents_impl()
        assert len(docs) == 1
        assert docs[0].title == title
        assert docs[0].source_url == source_url
        assert docs[0].chunk_count >= 1
    
    @patch('mcp_servers.document_store.get_embedding')
    def test_ingest_duplicate_document_skipped(self, mock_embedding):
        """Test that duplicate documents are not re-ingested."""
        from mcp_servers.document_store import ingest_document_impl
        
        mock_embedding.return_value = [0.1] * 1536
        
        title = "Duplicate Test"
        content = "Test content for duplicate detection."
        source_url = "https://example.com/duplicate"
        
        # Ingest first time
        result1 = ingest_document_impl(title, content, source_url)
        assert "Successfully" in result1
        
        # Ingest second time - should skip
        result2 = ingest_document_impl(title, content, source_url)
        assert "already exists" in result2.lower()
    
    @patch('mcp_servers.document_store.get_embedding')
    def test_search_documents_returns_results(self, mock_embedding):
        """Test that search returns relevant documents."""
        from mcp_servers.document_store import (
            ingest_document_impl,
            search_documents_impl,
        )
        
        # Mock embedding to return different vectors for different inputs
        call_count = [0]
        def mock_embed(text):
            call_count[0] += 1
            # Return slightly different embeddings for variety
            base = [0.1] * 1536
            base[0] = 0.1 + (call_count[0] * 0.01)
            return base
        
        mock_embedding.side_effect = mock_embed
        
        # Ingest a test document
        ingest_document_impl(
            "Go Concurrency Guide",
            "Go provides goroutines and channels for concurrent programming. "
            "Goroutines are lightweight threads managed by the Go runtime.",
            "https://example.com/go-concurrency"
        )
        
        # Search for it
        results = search_documents_impl("Go concurrency patterns", n_results=5)
        
        assert len(results) >= 1
        assert results[0].source == "https://example.com/go-concurrency"
        assert 0.0 <= results[0].relevance_score <= 1.0
    
    @patch('mcp_servers.document_store.get_embedding')
    def test_search_empty_collection_returns_empty(self, mock_embedding):
        """Test that searching an empty collection returns empty list."""
        from mcp_servers.document_store import search_documents_impl
        
        mock_embedding.return_value = [0.1] * 1536
        
        results = search_documents_impl("any query", n_results=5)
        
        assert results == []
    
    @patch('mcp_servers.document_store.get_embedding')
    def test_list_documents_returns_summaries(self, mock_embedding):
        """Test that list_documents returns proper summaries."""
        from mcp_servers.document_store import (
            ingest_document_impl,
            list_documents_impl,
            DocumentSummary,
        )
        
        mock_embedding.return_value = [0.1] * 1536
        
        # Ingest multiple documents
        docs_to_ingest = [
            ("Doc 1", "Content for document one.", "https://example.com/1"),
            ("Doc 2", "Content for document two.", "https://example.com/2"),
            ("Doc 3", "Content for document three.", "https://example.com/3"),
        ]
        
        for title, content, url in docs_to_ingest:
            ingest_document_impl(title, content, url)
        
        # List all documents
        summaries = list_documents_impl()
        
        assert len(summaries) == 3
        for summary in summaries:
            assert isinstance(summary, DocumentSummary)
            assert summary.title in ["Doc 1", "Doc 2", "Doc 3"]
            assert summary.chunk_count >= 1
            assert summary.ingested_at  # Should have timestamp
    
    @patch('mcp_servers.document_store.get_embedding')
    def test_document_result_has_correct_structure(self, mock_embedding):
        """Test that DocumentResult objects have correct fields."""
        from mcp_servers.document_store import (
            ingest_document_impl,
            search_documents_impl,
            DocumentResult,
        )
        
        mock_embedding.return_value = [0.1] * 1536
        
        # Ingest a document
        ingest_document_impl(
            "Rust Ownership",
            "Rust's ownership system prevents memory leaks and data races at compile time.",
            "https://rust-lang.org/ownership"
        )
        
        # Search and check result structure
        results = search_documents_impl("memory safety", n_results=1)
        
        assert len(results) >= 1
        result = results[0]
        
        assert isinstance(result, DocumentResult)
        assert hasattr(result, "content")
        assert hasattr(result, "source")
        assert hasattr(result, "relevance_score")
        assert hasattr(result, "metadata")
        
        # Check metadata contains expected fields
        assert "title" in result.metadata
        assert "chunk_index" in result.metadata
        assert "document_id" in result.metadata
    
    @patch('mcp_servers.document_store.get_embedding')
    def test_relevance_scores_are_reasonable(self, mock_embedding):
        """Test that relevance scores are in expected range."""
        from mcp_servers.document_store import (
            ingest_document_impl,
            search_documents_impl,
        )
        
        # Create embeddings that should result in high similarity
        embedding_values = [[0.5] * 1536, [0.5] * 1536]
        mock_embedding.side_effect = lambda x: embedding_values.pop(0) if embedding_values else [0.5] * 1536
        
        # Ingest document
        ingest_document_impl(
            "Similar Content",
            "This document is about programming.",
            "https://example.com/similar"
        )
        
        # Search with vector that should be very similar
        results = search_documents_impl("programming content", n_results=1)
        
        assert len(results) >= 1
        # Score should be between 0 and 1
        assert 0.0 <= results[0].relevance_score <= 1.0


if __name__ == "__main__":
    # Run pytest when this file is executed directly
    pytest.main([__file__, "-v"])