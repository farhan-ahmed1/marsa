"""Unified MCP Client Wrapper.

Provides a clean interface for agents to interact with MCP servers (Tavily search
and ChromaDB document store) without needing to know MCP internals.

Usage:
    from mcp_client import MCPClient
    
    client = MCPClient()
    
    # Web search
    results = await client.web_search("Python async patterns")
    
    # Document search
    docs = await client.doc_search("distributed systems CAP theorem")
    
    # Ingest document
    doc_id = await client.ingest_doc(
        title="My Document",
        content="Document content...",
        source_url="https://example.com/doc"
    )
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add backend directory to path for imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from pydantic import BaseModel, Field # noqa: E402

# Import server implementations directly (for direct mode)
from mcp_servers.tavily_search import ( # noqa: E402
    SearchResult,
    search_impl as tavily_search_impl,
)
from mcp_servers.document_store import ( # noqa: E402
    DocumentResult,
    DocumentSummary,
    search_documents_impl,
    ingest_document_impl,
    list_documents_impl,
)
from utils.resilience import ( # noqa: E402
    get_logger,
    ExternalAPIError,
    ValidationError,
)


# Initialize structured logging
logger = get_logger("mcp_client")


class WebSearchResult(BaseModel):
    """Web search result from Tavily.
    
    Wraps the raw SearchResult with a cleaner interface for agents.
    """
    
    title: str = Field(description="Title of the web page")
    url: str = Field(description="URL of the web page")
    content: str = Field(description="Relevant content snippet from the page")
    score: float = Field(description="Relevance score (0.0 to 1.0)")
    published_date: Optional[str] = Field(
        default=None,
        description="Published date of the content (ISO format), if available"
    )

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "WebSearchResult":
        """Create from a Tavily SearchResult."""
        return cls(
            title=result.title,
            url=result.url,
            content=result.content,
            score=result.score,
            published_date=result.published_date,
        )


class DocSearchResult(BaseModel):
    """Document search result from ChromaDB.
    
    Wraps the raw DocumentResult with a cleaner interface for agents.
    """
    
    content: str = Field(description="The document chunk content")
    source_url: str = Field(description="Source URL of the document")
    title: str = Field(description="Document title")
    relevance_score: float = Field(description="Relevance score (0.0 to 1.0)")
    chunk_index: int = Field(default=0, description="Index of this chunk within the document")
    document_id: str = Field(default="", description="Unique document identifier")

    @classmethod
    def from_document_result(cls, result: DocumentResult) -> "DocSearchResult":
        """Create from a ChromaDB DocumentResult."""
        return cls(
            content=result.content,
            source_url=result.source,
            title=result.metadata.get("title", ""),
            relevance_score=result.relevance_score,
            chunk_index=result.metadata.get("chunk_index", 0),
            document_id=result.metadata.get("document_id", ""),
        )


class IngestResult(BaseModel):
    """Result from ingesting a document."""
    
    document_id: str = Field(description="Unique document identifier")
    chunk_count: int = Field(description="Number of chunks created")
    message: str = Field(description="Status message")


class MCPClientError(Exception):
    """Base exception for MCP client errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class MCPConnectionError(MCPClientError):
    """Raised when connection to MCP server fails."""
    pass


class MCPToolError(MCPClientError):
    """Raised when an MCP tool call fails."""
    pass


class MCPClient:
    """Unified client for interacting with MCP servers.
    
    This client provides a clean async interface for agents to search the web,
    search documents, and ingest new content. It abstracts away the MCP protocol
    details and provides consistent error handling.
    
    The client can operate in two modes:
    - Direct mode (default): Calls server implementations directly (faster, simpler)
    - MCP mode: Connects to servers via stdio transport (full MCP protocol)
    
    For most use cases, direct mode is recommended as it avoids the overhead
    of spawning subprocess servers while maintaining the same interface.
    """
    
    def __init__(
        self,
        use_mcp_transport: bool = False,
        tavily_server_path: Optional[str] = None,
        doc_store_server_path: Optional[str] = None,
    ):
        """Initialize the MCP client.
        
        Args:
            use_mcp_transport: If True, connect to servers via MCP stdio transport.
                              If False (default), call server implementations directly.
            tavily_server_path: Path to tavily_search.py (required if use_mcp_transport=True)
            doc_store_server_path: Path to document_store.py (required if use_mcp_transport=True)
        """
        self.use_mcp_transport = use_mcp_transport
        self.tavily_server_path = tavily_server_path or str(
            Path(__file__).parent / "mcp_servers" / "tavily_search.py"
        )
        self.doc_store_server_path = doc_store_server_path or str(
            Path(__file__).parent / "mcp_servers" / "document_store.py"
        )
        
        # MCP session state (for transport mode)
        self._tavily_session = None
        self._doc_store_session = None
        self._initialized = False
        
        logger.info(
            "mcp_client_initialized",
            mode="mcp_transport" if use_mcp_transport else "direct",
        )
    
    async def _ensure_initialized(self) -> None:
        """Ensure MCP connections are established (transport mode only)."""
        if not self.use_mcp_transport:
            return
        
        if self._initialized:
            return
        
        # TODO: Implement MCP transport connection when needed
        # For now, we use direct mode which is simpler and faster
        raise NotImplementedError(
            "MCP transport mode not yet implemented. Use direct mode (default)."
        )
    
    async def web_search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """Search the web using Tavily API.
        
        Args:
            query: Search query string (max 500 characters)
            max_results: Maximum number of results to return (1-20, default: 5)
            
        Returns:
            List of web search results ordered by relevance
            
        Raises:
            MCPToolError: If the search fails
            ValidationError: If query or max_results are invalid
        """
        logger.info(
            "web_search_started",
            query=query[:100] if query else "",
            max_results=max_results,
        )
        
        try:
            if self.use_mcp_transport:
                await self._ensure_initialized()
                # MCP transport call would go here
                raise NotImplementedError("MCP transport not implemented")
            else:
                # Direct call to server implementation
                results = tavily_search_impl(query, max_results)
            
            web_results = [
                WebSearchResult.from_search_result(r)
                for r in results
            ]
            
            logger.info(
                "web_search_completed",
                query=query[:100] if query else "",
                result_count=len(web_results),
            )
            
            return web_results
            
        except (ValidationError, ExternalAPIError) as e:
            # Re-raise known errors
            logger.error(
                "web_search_failed",
                query=query[:100] if query else "",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "web_search_unexpected_error",
                query=query[:100] if query else "",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise MCPToolError(
                f"Web search failed: {e}",
                details={"query": query, "error": str(e)}
            ) from e
    
    async def doc_search(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[DocSearchResult]:
        """Search the document store (ChromaDB).
        
        Args:
            query: Search query string (max 500 characters)
            n_results: Maximum number of results to return (1-20, default: 5)
            
        Returns:
            List of document search results ordered by relevance
            
        Raises:
            MCPToolError: If the search fails
            ValidationError: If query or n_results are invalid
        """
        logger.info(
            "doc_search_started",
            query=query[:100] if query else "",
            n_results=n_results,
        )
        
        try:
            if self.use_mcp_transport:
                await self._ensure_initialized()
                # MCP transport call would go here
                raise NotImplementedError("MCP transport not implemented")
            else:
                # Direct call to server implementation
                results = search_documents_impl(query, n_results)
            
            doc_results = [
                DocSearchResult.from_document_result(r)
                for r in results
            ]
            
            logger.info(
                "doc_search_completed",
                query=query[:100] if query else "",
                result_count=len(doc_results),
            )
            
            return doc_results
            
        except (ValidationError, ExternalAPIError) as e:
            # Re-raise known errors
            logger.error(
                "doc_search_failed",
                query=query[:100] if query else "",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "doc_search_unexpected_error",
                query=query[:100] if query else "",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise MCPToolError(
                f"Document search failed: {e}",
                details={"query": query, "error": str(e)}
            ) from e
    
    async def ingest_doc(
        self,
        title: str,
        content: str,
        source_url: str,
    ) -> IngestResult:
        """Ingest a new document into the knowledge base.
        
        The document will be chunked and embedded for semantic search.
        
        Args:
            title: Document title
            content: Full document content
            source_url: Source URL of the document
            
        Returns:
            IngestResult with document_id and chunk_count
            
        Raises:
            MCPToolError: If ingestion fails
            ValidationError: If inputs are invalid
        """
        logger.info(
            "ingest_doc_started",
            title=title[:100] if title else "",
            content_length=len(content) if content else 0,
            source_url=source_url[:100] if source_url else "",
        )
        
        try:
            if self.use_mcp_transport:
                await self._ensure_initialized()
                # MCP transport call would go here
                raise NotImplementedError("MCP transport not implemented")
            else:
                # Direct call to server implementation
                result_message = ingest_document_impl(title, content, source_url)
            
            # Parse the result message to extract document_id and chunk_count
            # Expected format: "Document 'title' ingested successfully as doc_id (N chunks)"
            import re
            match = re.search(r"as (\w+) \((\d+) chunks?\)", result_message)
            if match:
                doc_id = match.group(1)
                chunk_count = int(match.group(2))
            else:
                doc_id = "unknown"
                chunk_count = 0
            
            result = IngestResult(
                document_id=doc_id,
                chunk_count=chunk_count,
                message=result_message,
            )
            
            logger.info(
                "ingest_doc_completed",
                title=title[:100] if title else "",
                document_id=result.document_id,
                chunk_count=result.chunk_count,
            )
            
            return result
            
        except (ValidationError, ExternalAPIError) as e:
            # Re-raise known errors
            logger.error(
                "ingest_doc_failed",
                title=title[:100] if title else "",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "ingest_doc_unexpected_error",
                title=title[:100] if title else "",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise MCPToolError(
                f"Document ingestion failed: {e}",
                details={"title": title, "error": str(e)}
            ) from e
    
    async def list_documents(self) -> list[DocumentSummary]:
        """List all documents in the knowledge base.
        
        Returns:
            List of document summaries with metadata
        """
        logger.info("list_documents_started")
        
        try:
            if self.use_mcp_transport:
                await self._ensure_initialized()
                raise NotImplementedError("MCP transport not implemented")
            else:
                results = list_documents_impl()
            
            logger.info(
                "list_documents_completed",
                document_count=len(results),
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "list_documents_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise MCPToolError(
                f"List documents failed: {e}",
                details={"error": str(e)}
            ) from e
    
    async def close(self) -> None:
        """Close any open connections.
        
        Should be called when the client is no longer needed.
        """
        if self.use_mcp_transport:
            # Close MCP sessions
            self._tavily_session = None
            self._doc_store_session = None
            self._initialized = False
        
        logger.info("mcp_client_closed")


# Convenient singleton instance for simple usage
_default_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """Get the default MCP client instance.
    
    Returns:
        The default MCPClient instance (creates one if needed)
    """
    global _default_client
    if _default_client is None:
        _default_client = MCPClient()
    return _default_client


async def reset_mcp_client() -> None:
    """Reset the default MCP client.
    
    Useful for testing or when you need to reconfigure the client.
    """
    global _default_client
    if _default_client is not None:
        await _default_client.close()
        _default_client = None


# Synchronous wrappers for convenience
def web_search_sync(query: str, max_results: int = 5) -> list[WebSearchResult]:
    """Synchronous wrapper for web_search.
    
    Args:
        query: Search query string
        max_results: Maximum number of results
        
    Returns:
        List of web search results
    """
    client = get_mcp_client()
    return asyncio.get_event_loop().run_until_complete(
        client.web_search(query, max_results)
    )


def doc_search_sync(query: str, n_results: int = 5) -> list[DocSearchResult]:
    """Synchronous wrapper for doc_search.
    
    Args:
        query: Search query string
        n_results: Maximum number of results
        
    Returns:
        List of document search results
    """
    client = get_mcp_client()
    return asyncio.get_event_loop().run_until_complete(
        client.doc_search(query, n_results)
    )


def ingest_doc_sync(title: str, content: str, source_url: str) -> IngestResult:
    """Synchronous wrapper for ingest_doc.
    
    Args:
        title: Document title
        content: Document content
        source_url: Source URL
        
    Returns:
        Ingestion result
    """
    client = get_mcp_client()
    return asyncio.get_event_loop().run_until_complete(
        client.ingest_doc(title, content, source_url)
    )
