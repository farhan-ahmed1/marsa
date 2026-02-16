"""Document Store MCP Server.

Provides document storage, retrieval, and search capabilities through ChromaDB
via the MCP protocol. Includes production-grade error handling, retry logic,
validation, and structured logging.
"""

import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add backend directory to path for config import
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import chromadb  # noqa: E402
from fastmcp import FastMCP  # noqa: E402
from openai import OpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from tenacity import (  # noqa: E402
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import Config  # noqa: E402
from utils.resilience import (  # noqa: E402
    DEFAULT_EMBEDDING_TIMEOUT,
    EmbeddingError,
    ExternalAPIError,
    ValidationError,
    get_logger,
    validate_max_results,
    validate_query,
)

# Initialize configuration
config = Config()

# Initialize structured logging
logger = get_logger("mcp.document_store")

# Initialize MCP server
mcp = FastMCP("document-store-server")

# Data directory for ChromaDB persistence
DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHROMADB_PATH = DATA_DIR / "chromadb"

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Validation constants
MAX_CONTENT_LENGTH = 100000  # 100KB max content for ingestion
MAX_TITLE_LENGTH = 500
MAX_URL_LENGTH = 2000

# Lazy-loaded clients
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_openai_client: Optional[OpenAI] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create the ChromaDB client instance."""
    global _chroma_client
    if _chroma_client is None:
        # Ensure data directory exists
        CHROMADB_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """Get or create the knowledge_base collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def get_openai_client() -> OpenAI:
    """Get or create the OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.openai_api_key)
    return _openai_client


# Retry decorator for embedding API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    reraise=True,
)
def _call_embedding_api(client: OpenAI, text: str) -> list[float]:
    """Make the actual OpenAI embedding API call with retry logic.
    
    Args:
        client: The OpenAI client instance
        text: The text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def get_embedding(text: str) -> list[float]:
    """Generate embedding for text using OpenAI text-embedding-3-small.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding vector
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    start_time = time.perf_counter()
    
    if not text or not text.strip():
        raise ValidationError(
            "Cannot generate embedding for empty text",
            details={"field": "text", "value": ""}
        )
    
    # Truncate very long text to prevent API errors
    max_tokens_approx = 8000 * 4  # ~8000 tokens, 4 chars per token
    if len(text) > max_tokens_approx:
        logger.warning(
            "embedding_text_truncated",
            original_length=len(text),
            truncated_length=max_tokens_approx,
        )
        text = text[:max_tokens_approx]
    
    try:
        embedding = _call_embedding_api(get_openai_client(), text)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            "embedding_generated",
            text_length=len(text),
            latency_ms=round(elapsed_ms, 2),
        )
        
        return embedding
        
    except TimeoutError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "embedding_timeout",
            text_length=len(text),
            timeout_seconds=DEFAULT_EMBEDDING_TIMEOUT,
            elapsed_ms=round(elapsed_ms, 2),
        )
        raise EmbeddingError(
            f"Embedding generation timed out after {DEFAULT_EMBEDDING_TIMEOUT} seconds",
            details={"text_length": len(text), "timeout": DEFAULT_EMBEDDING_TIMEOUT}
        ) from e
    except (ConnectionError, OSError) as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "embedding_connection_error",
            text_length=len(text),
            error_type=type(e).__name__,
            error_message=str(e),
            elapsed_ms=round(elapsed_ms, 2),
        )
        raise EmbeddingError(
            f"Failed to connect to OpenAI API: {e}",
            details={"text_length": len(text), "error": str(e)}
        ) from e
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "embedding_api_error",
            text_length=len(text),
            error_type=type(e).__name__,
            error_message=str(e),
            elapsed_ms=round(elapsed_ms, 2),
        )
        raise EmbeddingError(
            f"Embedding generation failed: {e}",
            details={"text_length": len(text), "error": str(e)}
        ) from e


def generate_document_id(title: str, source_url: str) -> str:
    """Generate a unique document ID based on title and source URL.
    
    Args:
        title: Document title
        source_url: Source URL of the document
        
    Returns:
        SHA256 hash of the title and source URL
    """
    content = f"{title}:{source_url}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a unique chunk ID.
    
    Args:
        doc_id: Parent document ID
        chunk_index: Index of the chunk within the document
        
    Returns:
        Unique chunk ID
    """
    return f"{doc_id}_chunk_{chunk_index:04d}"


# Pydantic models for MCP tools
class DocumentResult(BaseModel):
    """A single document search result."""
    
    content: str = Field(description="The text content of the document chunk")
    source: str = Field(description="Source URL of the document")
    relevance_score: float = Field(description="Relevance score (0.0 to 1.0, higher is better)")
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (title, chunk_index, ingested_at)"
    )


class DocumentSummary(BaseModel):
    """Summary information about a document in the store."""
    
    document_id: str = Field(description="Unique document identifier")
    title: str = Field(description="Document title")
    source_url: str = Field(description="Source URL of the document")
    chunk_count: int = Field(description="Number of chunks this document was split into")
    ingested_at: str = Field(description="ISO timestamp when document was ingested")


@mcp.tool()
def search_documents(query: str, n_results: int = 5) -> list[DocumentResult]:
    """Search for documents by semantic similarity.
    
    Embeds the query and runs similarity search against the ChromaDB
    knowledge base, returning the most relevant document chunks.
    
    Args:
        query: The search query string
        n_results: Maximum number of results to return (default: 5)
        
    Returns:
        List of document results ordered by relevance
    """
    return search_documents_impl(query, n_results)


def search_documents_impl(query: str, n_results: int = 5) -> list[DocumentResult]:
    """Implementation of document search.
    
    Args:
        query: The search query string (max 500 characters)
        n_results: Maximum number of results to return (1-20, default: 5)
        
    Returns:
        List of document results ordered by relevance
        
    Raises:
        ValidationError: If query or n_results are invalid
        EmbeddingError: If embedding generation fails
        ExternalAPIError: If ChromaDB query fails
    """
    start_time = time.perf_counter()
    
    # Input validation
    try:
        validated_query = validate_query(query)
        validated_n_results = validate_max_results(n_results)
    except ValidationError:
        logger.warning(
            "search_documents_validation_failed",
            query_length=len(query) if query else 0,
            n_results=n_results,
        )
        raise
    
    logger.info(
        "search_documents_started",
        query=validated_query[:100],
        n_results=validated_n_results,
    )
    
    try:
        collection = get_collection()
        
        # Check if collection is empty
        doc_count = collection.count()
        if doc_count == 0:
            logger.info(
                "search_documents_empty_collection",
                query=validated_query[:100],
            )
            return []
        
        # Generate embedding for query
        query_embedding = get_embedding(validated_query)
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(validated_n_results, doc_count),
            include=["documents", "metadatas", "distances"]
        )
    except (ValidationError, EmbeddingError):
        # Re-raise validation and embedding errors as-is
        raise
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "search_documents_error",
            query=validated_query[:100],
            error_type=type(e).__name__,
            error_message=str(e),
            elapsed_ms=round(elapsed_ms, 2),
        )
        raise ExternalAPIError(
            f"Document search failed: {e}",
            details={"query": validated_query, "error": str(e)}
        ) from e
    
    # Transform results to DocumentResult format
    # ChromaDB returns cosine distance, convert to similarity score
    document_results = []
    
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    for doc, meta, distance in zip(documents, metadatas, distances):
        # Convert cosine distance to similarity score (1 - distance for cosine)
        similarity_score = max(0.0, min(1.0, 1.0 - distance))
        
        result = DocumentResult(
            content=doc,
            source=meta.get("source_url", ""),
            relevance_score=round(similarity_score, 4),
            metadata={
                "title": meta.get("title", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "document_id": meta.get("document_id", ""),
                "ingested_at": meta.get("ingested_at", "")
            }
        )
        document_results.append(result)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "search_documents_completed",
        query=validated_query[:100],
        result_count=len(document_results),
        latency_ms=round(elapsed_ms, 2),
    )
    
    return document_results


@mcp.tool()
def ingest_document(title: str, content: str, source_url: str) -> str:
    """Ingest a new document into the knowledge base.
    
    Chunks the document, generates embeddings, and stores them in ChromaDB.
    This is useful for the researcher agent to save findings at runtime.
    
    Args:
        title: Title of the document
        content: Full text content of the document
        source_url: Source URL of the document
        
    Returns:
        Success message with document ID and chunk count
    """
    return ingest_document_impl(title, content, source_url)


def ingest_document_impl(
    title: str, 
    content: str, 
    source_url: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> str:
    """Implementation of document ingestion.
    
    Args:
        title: Title of the document (max 500 characters)
        content: Full text content of the document (max 100KB)
        source_url: Source URL of the document (max 2000 characters)
        chunk_size: Target chunk size in tokens (approximate)
        chunk_overlap: Overlap between chunks in tokens (approximate)
        
    Returns:
        Success message with document ID and chunk count
        
    Raises:
        ValidationError: If inputs are invalid
        EmbeddingError: If embedding generation fails
        ExternalAPIError: If ChromaDB operations fail
    """
    start_time = time.perf_counter()
    
    # Input validation
    if not title or not title.strip():
        raise ValidationError(
            "Document title cannot be empty",
            details={"field": "title", "value": title}
        )
    
    title = title.strip()
    if len(title) > MAX_TITLE_LENGTH:
        raise ValidationError(
            f"Title exceeds maximum length of {MAX_TITLE_LENGTH} characters",
            details={"field": "title", "length": len(title), "max_length": MAX_TITLE_LENGTH}
        )
    
    if not content or not content.strip():
        raise ValidationError(
            "Document content cannot be empty",
            details={"field": "content", "value": ""}
        )
    
    content = content.strip()
    if len(content) > MAX_CONTENT_LENGTH:
        raise ValidationError(
            f"Content exceeds maximum length of {MAX_CONTENT_LENGTH} characters",
            details={"field": "content", "length": len(content), "max_length": MAX_CONTENT_LENGTH}
        )
    
    if not source_url or not source_url.strip():
        raise ValidationError(
            "Source URL cannot be empty",
            details={"field": "source_url", "value": source_url}
        )
    
    source_url = source_url.strip()
    if len(source_url) > MAX_URL_LENGTH:
        raise ValidationError(
            f"Source URL exceeds maximum length of {MAX_URL_LENGTH} characters",
            details={"field": "source_url", "length": len(source_url), "max_length": MAX_URL_LENGTH}
        )
    
    logger.info(
        "ingest_document_started",
        title=title[:100],
        content_length=len(content),
        source_url=source_url[:100],
    )
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    try:
        collection = get_collection()
        
        # Generate document ID
        doc_id = generate_document_id(title, source_url)
        
        # Check if document already exists
        existing = collection.get(
            where={"document_id": doc_id},
            limit=1
        )
        if existing and existing.get("ids"):
            logger.info(
                "ingest_document_already_exists",
                document_id=doc_id,
                title=title[:100],
            )
            return f"Document already exists with ID: {doc_id}"
        
        # Split document into chunks
        # Approximate 4 characters per token for English text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,
            chunk_overlap=chunk_overlap * 4,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(content)
        
        if not chunks:
            logger.warning(
                "ingest_document_no_chunks",
                title=title[:100],
                content_length=len(content),
            )
            return "Error: Document content resulted in no chunks"
        
        logger.debug(
            "ingest_document_chunked",
            document_id=doc_id,
            chunk_count=len(chunks),
        )
        
        # Generate embeddings for all chunks
        embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            embeddings.append(embedding)
            
            # Log progress for large documents
            if len(chunks) > 10 and (i + 1) % 10 == 0:
                logger.debug(
                    "ingest_document_embedding_progress",
                    document_id=doc_id,
                    completed=i + 1,
                    total=len(chunks),
                )
        
        # Prepare metadata and IDs
        timestamp = datetime.now(timezone.utc).isoformat()
        ids = [generate_chunk_id(doc_id, i) for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": doc_id,
                "title": title,
                "source_url": source_url,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingested_at": timestamp
            }
            for i in range(len(chunks))
        ]
        
        # Add to ChromaDB
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "ingest_document_completed",
            document_id=doc_id,
            title=title[:100],
            chunk_count=len(chunks),
            latency_ms=round(elapsed_ms, 2),
        )
        
        return f"Successfully ingested document '{title}' with ID: {doc_id} ({len(chunks)} chunks)"
        
    except (ValidationError, EmbeddingError):
        # Re-raise validation and embedding errors as-is
        raise
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "ingest_document_error",
            title=title[:100],
            error_type=type(e).__name__,
            error_message=str(e),
            elapsed_ms=round(elapsed_ms, 2),
        )
        raise ExternalAPIError(
            f"Document ingestion failed: {e}",
            details={"title": title, "error": str(e)}
        ) from e


@mcp.tool()
def list_documents() -> list[DocumentSummary]:
    """List all documents in the knowledge base.
    
    Returns metadata about all ingested documents, including
    title, source URL, chunk count, and ingestion timestamp.
    
    Returns:
        List of document summaries
    """
    return list_documents_impl()


def list_documents_impl() -> list[DocumentSummary]:
    """Implementation of document listing.
    
    Returns:
        List of document summaries
        
    Raises:
        ExternalAPIError: If ChromaDB query fails
    """
    start_time = time.perf_counter()
    
    logger.debug("list_documents_started")
    
    try:
        collection = get_collection()
        
        # Get all documents with metadata
        all_docs = collection.get(
            include=["metadatas"]
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "list_documents_error",
            error_type=type(e).__name__,
            error_message=str(e),
            elapsed_ms=round(elapsed_ms, 2),
        )
        raise ExternalAPIError(
            f"Failed to list documents: {e}",
            details={"error": str(e)}
        ) from e
    
    if not all_docs or not all_docs.get("metadatas"):
        logger.info(
            "list_documents_empty",
            document_count=0,
        )
        return []
    
    # Group by document_id to get unique documents
    documents_map: dict[str, dict] = {}
    
    for metadata in all_docs["metadatas"]:
        doc_id = metadata.get("document_id", "")
        if doc_id not in documents_map:
            documents_map[doc_id] = {
                "document_id": doc_id,
                "title": metadata.get("title", ""),
                "source_url": metadata.get("source_url", ""),
                "chunk_count": 0,
                "ingested_at": metadata.get("ingested_at", "")
            }
        documents_map[doc_id]["chunk_count"] += 1
    
    # Convert to DocumentSummary list
    summaries = [
        DocumentSummary(**doc_info)
        for doc_info in documents_map.values()
    ]
    
    # Sort by ingestion time (most recent first)
    summaries.sort(key=lambda x: x.ingested_at, reverse=True)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "list_documents_completed",
        document_count=len(summaries),
        total_chunks=sum(s.chunk_count for s in summaries),
        latency_ms=round(elapsed_ms, 2),
    )
    
    return summaries


# Utility function for testing and scripts
def reset_collection() -> str:
    """Reset the knowledge base collection (for testing purposes).
    
    Returns:
        Success message
    """
    global _collection
    client = get_chroma_client()
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("knowledge_base")
    except (ValueError, Exception):
        pass  # Collection doesn't exist or other error
    
    # Reset cached collection
    _collection = None
    
    # Recreate collection
    get_collection()
    
    return "Collection reset successfully"


if __name__ == "__main__":
    mcp.run()
