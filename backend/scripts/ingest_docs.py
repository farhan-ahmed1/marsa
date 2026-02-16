#!/usr/bin/env python3
"""Document Ingestion Script for MARSA.

This script ingests documents from the data/sample_docs directory into
the ChromaDB knowledge base for the Document Store MCP server.

Usage:
    python backend/scripts/ingest_docs.py
    
    # Or from backend directory:
    python scripts/ingest_docs.py
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from mcp_servers.document_store import (
    ingest_document_impl,
    list_documents_impl,
    get_collection,
    reset_collection,
)


# Sample documents directory
SAMPLE_DOCS_DIR = Path(__file__).parent.parent.parent / "data" / "sample_docs"


def read_document(file_path: Path) -> tuple[str, str, str]:
    """Read a document file and extract metadata.
    
    Expects the first line to be the title and an optional second line
    with 'Source: <url>'. The rest is the content.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Tuple of (title, content, source_url)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        return file_path.stem, "", f"file://{file_path}"
    
    # First line is title
    title = lines[0].strip().lstrip("# ")
    
    # Check for source URL on second line
    source_url = f"file://{file_path}"
    content_start = 1
    
    if len(lines) > 1 and lines[1].strip().lower().startswith("source:"):
        source_url = lines[1].strip()[7:].strip()
        content_start = 2
    
    # Rest is content
    content = "".join(lines[content_start:]).strip()
    
    return title, content, source_url


def ingest_all_documents(reset: bool = False) -> dict:
    """Ingest all documents from the sample_docs directory.
    
    Args:
        reset: If True, reset the collection before ingesting
        
    Returns:
        Dictionary with ingestion statistics
    """
    if not SAMPLE_DOCS_DIR.exists():
        print(f"Sample docs directory not found: {SAMPLE_DOCS_DIR}")
        return {"error": "Sample docs directory not found"}
    
    # Find all text files
    doc_files = list(SAMPLE_DOCS_DIR.glob("*.txt")) + list(SAMPLE_DOCS_DIR.glob("*.md"))
    
    if not doc_files:
        print(f"No documents found in {SAMPLE_DOCS_DIR}")
        return {"error": "No documents found", "directory": str(SAMPLE_DOCS_DIR)}
    
    print(f"Found {len(doc_files)} documents to ingest")
    
    if reset:
        print("Resetting collection...")
        reset_collection()
    
    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }
    
    for doc_file in sorted(doc_files):
        print(f"\nProcessing: {doc_file.name}")
        try:
            title, content, source_url = read_document(doc_file)
            
            if not content:
                print("  WARNING: Skipped (no content)")
                results["skipped"].append(doc_file.name)
                continue
            
            result = ingest_document_impl(title, content, source_url)
            print(f"  SUCCESS: {result}")
            
            if "already exists" in result.lower():
                results["skipped"].append(doc_file.name)
            else:
                results["success"].append(doc_file.name)
                
        except Exception as e:
            print(f"  FAILED: {e}")
            results["failed"].append({"file": doc_file.name, "error": str(e)})
    
    # Print summary
    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Successfully ingested: {len(results['success'])}")
    print(f"Skipped (already exist or empty): {len(results['skipped'])}")
    print(f"Failed: {len(results['failed'])}")
    
    # Show collection stats
    collection = get_collection()
    print(f"\nCollection now contains {collection.count()} chunks")
    
    # List all documents
    documents = list_documents_impl()
    print(f"From {len(documents)} unique documents:")
    for doc in documents:
        print(f"  - {doc.title} ({doc.chunk_count} chunks)")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into MARSA knowledge base")
    parser.add_argument(
        "--reset", 
        action="store_true",
        help="Reset the collection before ingesting"
    )
    parser.add_argument(
        "--single",
        type=str,
        help="Ingest a single file by path"
    )
    
    args = parser.parse_args()
    
    if args.single:
        file_path = Path(args.single)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            sys.exit(1)
        
        title, content, source_url = read_document(file_path)
        result = ingest_document_impl(title, content, source_url)
        print(result)
    else:
        ingest_all_documents(reset=args.reset)


if __name__ == "__main__":
    main()
