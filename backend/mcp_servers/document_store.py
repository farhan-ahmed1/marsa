"""Document Store MCP Server.

Provides document storage, retrieval, and search capabilities through ChromaDB
via the MCP protocol.
"""

import hashlib
import sys
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

from config import Config  # noqa: E402

# Initialize configuration
config = Config()

# Initialize MCP server
mcp = FastMCP("document-store-server")

# Data directory for ChromaDB persistence
DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHROMADB_PATH = DATA_DIR / "chromadb"

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

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


def get_embedding(text: str) -> list[float]:
    """Generate embedding for text using OpenAI text-embedding-3-small.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


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
        query: The search query string
        n_results: Maximum number of results to return (default: 5)
        
    Returns:
        List of document results ordered by relevance
    """
    collection = get_collection()
    
    # Check if collection is empty
    if collection.count() == 0:
        return []
    
    # Generate embedding for query
    query_embedding = get_embedding(query)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    
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
        title: Title of the document
        content: Full text content of the document
        source_url: Source URL of the document
        chunk_size: Target chunk size in tokens (approximate)
        chunk_overlap: Overlap between chunks in tokens (approximate)
        
    Returns:
        Success message with document ID and chunk count
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    collection = get_collection()
    
    # Generate document ID
    doc_id = generate_document_id(title, source_url)
    
    # Check if document already exists
    existing = collection.get(
        where={"document_id": doc_id},
        limit=1
    )
    if existing and existing.get("ids"):
        return f"Document already exists with ID: {doc_id}"
    
    # Split document into chunks
    # Approximate 4 characters per token for English text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,  # Convert tokens to chars approximately
        chunk_overlap=chunk_overlap * 4,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(content)
    
    if not chunks:
        return "Error: Document content resulted in no chunks"
    
    # Generate embeddings for all chunks
    embeddings = [get_embedding(chunk) for chunk in chunks]
    
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
    
    return f"Successfully ingested document '{title}' with ID: {doc_id} ({len(chunks)} chunks)"


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
    """
    collection = get_collection()
    
    # Get all documents with metadata
    all_docs = collection.get(
        include=["metadatas"]
    )
    
    if not all_docs or not all_docs.get("metadatas"):
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
