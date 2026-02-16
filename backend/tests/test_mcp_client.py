"""Tests for the unified MCP client wrapper.

Tests the MCPClient class and its methods for web search, document search,
and document ingestion.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from mcp_client import (  # noqa: E402
    MCPClient,
    WebSearchResult,
    DocSearchResult,
    IngestResult,
    MCPToolError,
    get_mcp_client,
    reset_mcp_client,
)
from mcp_servers.tavily_search import SearchResult  # noqa: E402
from mcp_servers.document_store import DocumentResult  # noqa: E402


class TestWebSearchResult:
    """Tests for WebSearchResult model."""
    
    def test_from_search_result(self):
        """Test creating WebSearchResult from SearchResult."""
        tavily_result = SearchResult(
            title="Python Tutorial",
            url="https://example.com/python",
            content="Learn Python programming.",
            score=0.95,
            published_date="2026-01-15"
        )
        
        web_result = WebSearchResult.from_search_result(tavily_result)
        
        assert web_result.title == "Python Tutorial"
        assert web_result.url == "https://example.com/python"
        assert web_result.content == "Learn Python programming."
        assert web_result.score == 0.95
        assert web_result.published_date == "2026-01-15"
    
    def test_from_search_result_no_date(self):
        """Test creating WebSearchResult with no published date."""
        tavily_result = SearchResult(
            title="Python Guide",
            url="https://example.com/guide",
            content="Guide content.",
            score=0.8,
            published_date=None
        )
        
        web_result = WebSearchResult.from_search_result(tavily_result)
        
        assert web_result.published_date is None


class TestDocSearchResult:
    """Tests for DocSearchResult model."""
    
    def test_from_document_result(self):
        """Test creating DocSearchResult from DocumentResult."""
        doc_result = DocumentResult(
            content="Document chunk content.",
            source="https://docs.python.org/3/",
            relevance_score=0.89,
            metadata={
                "title": "Python Docs",
                "chunk_index": 2,
                "document_id": "abc123"
            }
        )
        
        result = DocSearchResult.from_document_result(doc_result)
        
        assert result.content == "Document chunk content."
        assert result.source_url == "https://docs.python.org/3/"
        assert result.title == "Python Docs"
        assert result.relevance_score == 0.89
        assert result.chunk_index == 2
        assert result.document_id == "abc123"
    
    def test_from_document_result_empty_metadata(self):
        """Test creating DocSearchResult with empty metadata."""
        doc_result = DocumentResult(
            content="Content",
            source="https://example.com",
            relevance_score=0.7,
            metadata={}
        )
        
        result = DocSearchResult.from_document_result(doc_result)
        
        assert result.title == ""
        assert result.chunk_index == 0
        assert result.document_id == ""


class TestMCPClientInitialization:
    """Tests for MCPClient initialization."""
    
    def test_default_initialization(self):
        """Test default client initialization uses direct mode."""
        client = MCPClient()
        
        assert client.use_mcp_transport is False
        assert client._initialized is False
    
    def test_mcp_transport_initialization(self):
        """Test MCP transport mode initialization."""
        client = MCPClient(use_mcp_transport=True)
        
        assert client.use_mcp_transport is True
    
    def test_custom_server_paths(self):
        """Test custom server paths are stored."""
        client = MCPClient(
            tavily_server_path="/custom/tavily.py",
            doc_store_server_path="/custom/docstore.py"
        )
        
        assert client.tavily_server_path == "/custom/tavily.py"
        assert client.doc_store_server_path == "/custom/docstore.py"


class TestMCPClientWebSearch:
    """Tests for MCPClient.web_search method."""
    
    @pytest.fixture
    def mock_tavily_results(self):
        """Mock Tavily search results."""
        return [
            SearchResult(
                title="Python Programming",
                url="https://example.com/python",
                content="Python is great.",
                score=0.9,
                published_date="2026-01-15"
            ),
            SearchResult(
                title="Python Tutorial",
                url="https://tutorial.com/py",
                content="Learn Python basics.",
                score=0.85,
                published_date=None
            )
        ]
    
    @pytest.mark.asyncio
    @patch('mcp_client.tavily_search_impl')
    async def test_web_search_returns_results(self, mock_search, mock_tavily_results):
        """Test web_search returns WebSearchResult objects."""
        mock_search.return_value = mock_tavily_results
        
        client = MCPClient()
        results = await client.web_search("Python programming", max_results=2)
        
        assert len(results) == 2
        assert all(isinstance(r, WebSearchResult) for r in results)
        assert results[0].title == "Python Programming"
        assert results[1].title == "Python Tutorial"
        
        mock_search.assert_called_once_with("Python programming", 2)
    
    @pytest.mark.asyncio
    @patch('mcp_client.tavily_search_impl')
    async def test_web_search_default_max_results(self, mock_search, mock_tavily_results):
        """Test web_search uses default max_results of 5."""
        mock_search.return_value = mock_tavily_results
        
        client = MCPClient()
        await client.web_search("test query")
        
        mock_search.assert_called_once_with("test query", 5)
    
    @pytest.mark.asyncio
    @patch('mcp_client.tavily_search_impl')
    async def test_web_search_handles_errors(self, mock_search):
        """Test web_search wraps unexpected errors in MCPToolError."""
        mock_search.side_effect = RuntimeError("Unexpected error")
        
        client = MCPClient()
        
        with pytest.raises(MCPToolError) as exc_info:
            await client.web_search("test query")
        
        assert "Web search failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_web_search_mcp_transport_not_implemented(self):
        """Test MCP transport mode raises MCPToolError wrapping NotImplementedError."""
        client = MCPClient(use_mcp_transport=True)
        
        with pytest.raises(MCPToolError) as exc_info:
            await client.web_search("test query")
        
        assert "not yet implemented" in str(exc_info.value).lower()


class TestMCPClientDocSearch:
    """Tests for MCPClient.doc_search method."""
    
    @pytest.fixture
    def mock_doc_results(self):
        """Mock document search results."""
        return [
            DocumentResult(
                content="Document content about Go.",
                source="https://go.dev/doc/",
                relevance_score=0.92,
                metadata={"title": "Go Documentation", "chunk_index": 0}
            )
        ]
    
    @pytest.mark.asyncio
    @patch('mcp_client.search_documents_impl')
    async def test_doc_search_returns_results(self, mock_search, mock_doc_results):
        """Test doc_search returns DocSearchResult objects."""
        mock_search.return_value = mock_doc_results
        
        client = MCPClient()
        results = await client.doc_search("Go concurrency", n_results=3)
        
        assert len(results) == 1
        assert isinstance(results[0], DocSearchResult)
        assert results[0].title == "Go Documentation"
        
        mock_search.assert_called_once_with("Go concurrency", 3)
    
    @pytest.mark.asyncio
    @patch('mcp_client.search_documents_impl')
    async def test_doc_search_empty_results(self, mock_search):
        """Test doc_search handles empty results."""
        mock_search.return_value = []
        
        client = MCPClient()
        results = await client.doc_search("nonexistent topic")
        
        assert results == []
    
    @pytest.mark.asyncio
    @patch('mcp_client.search_documents_impl')
    async def test_doc_search_handles_errors(self, mock_search):
        """Test doc_search wraps unexpected errors in MCPToolError."""
        mock_search.side_effect = RuntimeError("Database error")
        
        client = MCPClient()
        
        with pytest.raises(MCPToolError) as exc_info:
            await client.doc_search("test query")
        
        assert "Document search failed" in str(exc_info.value)


class TestMCPClientIngestDoc:
    """Tests for MCPClient.ingest_doc method."""
    
    @pytest.mark.asyncio
    @patch('mcp_client.ingest_document_impl')
    async def test_ingest_doc_success(self, mock_ingest):
        """Test successful document ingestion."""
        mock_ingest.return_value = "Document 'Test Doc' ingested successfully as abc123 (5 chunks)"
        
        client = MCPClient()
        result = await client.ingest_doc(
            title="Test Doc",
            content="Long document content here...",
            source_url="https://example.com/doc"
        )
        
        assert isinstance(result, IngestResult)
        assert result.document_id == "abc123"
        assert result.chunk_count == 5
        assert "ingested successfully" in result.message
    
    @pytest.mark.asyncio
    @patch('mcp_client.ingest_document_impl')
    async def test_ingest_doc_single_chunk(self, mock_ingest):
        """Test document ingestion with single chunk."""
        mock_ingest.return_value = "Document 'Short' ingested successfully as xyz789 (1 chunk)"
        
        client = MCPClient()
        result = await client.ingest_doc(
            title="Short",
            content="Short content",
            source_url="https://example.com/short"
        )
        
        assert result.chunk_count == 1
    
    @pytest.mark.asyncio
    @patch('mcp_client.ingest_document_impl')
    async def test_ingest_doc_handles_errors(self, mock_ingest):
        """Test ingest_doc wraps errors in MCPToolError."""
        mock_ingest.side_effect = RuntimeError("Storage error")
        
        client = MCPClient()
        
        with pytest.raises(MCPToolError) as exc_info:
            await client.ingest_doc("Title", "Content", "https://example.com")
        
        assert "Document ingestion failed" in str(exc_info.value)


class TestMCPClientListDocuments:
    """Tests for MCPClient.list_documents method."""
    
    @pytest.mark.asyncio
    @patch('mcp_client.list_documents_impl')
    async def test_list_documents(self, mock_list):
        """Test listing documents."""
        from mcp_servers.document_store import DocumentSummary
        
        mock_list.return_value = [
            DocumentSummary(
                document_id="doc1",
                title="Document 1",
                source_url="https://example.com/1",
                chunk_count=3,
                ingested_at="2026-02-15T10:00:00Z"
            )
        ]
        
        client = MCPClient()
        results = await client.list_documents()
        
        assert len(results) == 1
        assert results[0].document_id == "doc1"


class TestMCPClientClose:
    """Tests for MCPClient.close method."""
    
    @pytest.mark.asyncio
    async def test_close_direct_mode(self):
        """Test closing client in direct mode."""
        client = MCPClient()
        await client.close()
        # Should complete without error
    
    @pytest.mark.asyncio
    async def test_close_resets_state(self):
        """Test closing client resets MCP state."""
        client = MCPClient(use_mcp_transport=True)
        client._initialized = True
        
        await client.close()
        
        assert client._initialized is False


class TestGetMCPClient:
    """Tests for get_mcp_client singleton function."""
    
    @pytest.fixture(autouse=True)
    async def reset_singleton(self):
        """Reset the singleton before each test."""
        await reset_mcp_client()
        yield
        await reset_mcp_client()
    
    def test_get_mcp_client_returns_instance(self):
        """Test get_mcp_client returns MCPClient instance."""
        client = get_mcp_client()
        
        assert isinstance(client, MCPClient)
    
    def test_get_mcp_client_returns_same_instance(self):
        """Test get_mcp_client returns singleton."""
        client1 = get_mcp_client()
        client2 = get_mcp_client()
        
        assert client1 is client2
    
    @pytest.mark.asyncio
    async def test_reset_mcp_client(self):
        """Test reset_mcp_client creates new instance."""
        client1 = get_mcp_client()
        await reset_mcp_client()
        client2 = get_mcp_client()
        
        assert client1 is not client2
