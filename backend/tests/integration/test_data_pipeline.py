"""Integration tests for the full data pipeline.

Tests the end-to-end flow: MCP client -> search -> score sources.
These tests make REAL API calls and should be run separately from unit tests.

Run with: make test:integration
"""

import sys
from pathlib import Path

import pytest

# Add backend directory to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from mcp_client import MCPClient, WebSearchResult, DocSearchResult  # noqa: E402
from agents.source_scorer import (  # noqa: E402
    score_source,
    score_sources,
    filter_sources_by_quality,
    SourceScore,
    is_authoritative,
)
from mcp_servers.document_store import reset_collection, ingest_document_impl  # noqa: E402


class TestWebSearchToScoring:
    """Integration tests: Tavily search -> source scoring."""
    
    @pytest.mark.asyncio
    async def test_web_search_and_score_results(self):
        """Test full flow: web search -> score each result.
        
        USES REAL TAVILY API - consumes quota!
        """
        print("\n" + "="*60)
        print("INTEGRATION TEST: Web Search -> Source Scoring")
        print("="*60)
        
        client = MCPClient()
        
        # Perform real web search
        query = "Python async await tutorial"
        results = await client.web_search(query, max_results=3)
        
        print(f"\nSearch query: '{query}'")
        print(f"Got {len(results)} results\n")
        
        assert len(results) > 0, "Should get at least one result"
        assert all(isinstance(r, WebSearchResult) for r in results)
        
        # Score each result
        scores = []
        for i, result in enumerate(results, 1):
            score = score_source(
                url=result.url,
                content=result.content,
                published_date=result.published_date
            )
            scores.append(score)
            
            print(f"{i}. {result.title[:50]}...")
            print(f"   URL: {result.url[:60]}...")
            print(f"   Score: {score.final_score:.2f} ({score.to_quality_level()})")
            print(f"   Domain: {score.domain} ({score.domain_category})")
            print(f"   Components: domain={score.domain_score:.1f}, "
                  f"recency={score.recency_score:.1f}, depth={score.depth_score:.1f}")
            print()
        
        # Verify scores are valid
        assert all(isinstance(s, SourceScore) for s in scores)
        assert all(0 <= s.final_score <= 1 for s in scores)
        
        print("="*60)
        print("Integration test passed: Web search + scoring works!")
        print("="*60)
    
    @pytest.mark.asyncio
    async def test_filter_web_results_by_quality(self):
        """Test filtering web search results by quality threshold.
        
        USES REAL TAVILY API - consumes quota!
        """
        print("\n" + "="*60)
        print("INTEGRATION TEST: Filter Web Results by Quality")
        print("="*60)
        
        client = MCPClient()
        
        # Search for a topic likely to have mixed quality results
        results = await client.web_search("Python best practices", max_results=5)
        
        # Convert to format expected by filter function
        sources = [
            {
                "url": r.url,
                "content": r.content,
                "published_date": r.published_date
            }
            for r in results
        ]
        
        # Filter by quality
        high_quality = filter_sources_by_quality(sources, min_score=0.6)
        
        print(f"\nTotal results: {len(results)}")
        print(f"High quality (>= 0.6): {len(high_quality)}")
        
        for score in high_quality:
            print(f"  - {score.domain}: {score.final_score:.2f} ({score.domain_category})")
        
        # All filtered results should meet threshold
        assert all(s.final_score >= 0.6 for s in high_quality)
        
        print("\n" + "="*60)
        print("Integration test passed: Quality filtering works!")
        print("="*60)


class TestDocSearchToScoring:
    """Integration tests: ChromaDB search -> source scoring."""
    
    @pytest.fixture(autouse=True)
    def setup_test_documents(self):
        """Set up test documents in ChromaDB."""
        print("\n[Setup] Resetting ChromaDB collection...")
        reset_collection()
        
        # Ingest test documents
        test_docs = [
            {
                "title": "Go Concurrency Patterns",
                "content": """
                Go provides powerful concurrency primitives through goroutines and channels.
                Goroutines are lightweight threads managed by the Go runtime, allowing efficient
                concurrent execution without the overhead of OS threads. Channels enable safe
                communication between goroutines without explicit locking mechanisms.
                
                Key patterns include:
                1. Worker pools for parallel task processing
                2. Fan-out/fan-in for distributing and collecting work
                3. Pipeline patterns for data transformation stages
                4. Context for cancellation and timeouts
                
                The select statement enables waiting on multiple channel operations,
                making it easy to implement timeouts and non-blocking operations.
                This combination of features makes concurrent programming in Go simpler
                and less error-prone compared to traditional threading models.
                """ * 3,  # Make it longer for depth score
                "source_url": "https://go.dev/doc/effective_go"
            },
            {
                "title": "Rust Ownership System",
                "content": """
                Rust's ownership system ensures memory safety without garbage collection.
                Each value has an owner, and when the owner goes out of scope, the value
                is dropped. Borrowing rules prevent data races at compile time.
                """ * 3,
                "source_url": "https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html"
            },
            {
                "title": "Random Blog Post",
                "content": "Short random content about programming.",
                "source_url": "https://random-blog-xyz.com/post"
            }
        ]
        
        for doc in test_docs:
            result = ingest_document_impl(
                doc["title"],
                doc["content"].strip(),
                doc["source_url"]
            )
            print(f"[Setup] Ingested: {result}")
        
        yield
    
    @pytest.mark.asyncio
    async def test_doc_search_and_score_results(self):
        """Test full flow: document search -> score each result.
        
        USES REAL OpenAI embeddings - consumes quota!
        """
        print("\n" + "="*60)
        print("INTEGRATION TEST: Document Search -> Source Scoring")
        print("="*60)
        
        client = MCPClient()
        
        # Search documents
        query = "Go concurrency patterns"
        results = await client.doc_search(query, n_results=3)
        
        print(f"\nSearch query: '{query}'")
        print(f"Got {len(results)} results\n")
        
        assert len(results) > 0, "Should get at least one result"
        assert all(isinstance(r, DocSearchResult) for r in results)
        
        # Score each result
        scores = []
        for i, result in enumerate(results, 1):
            score = score_source(
                url=result.source_url,
                content=result.content,
                published_date=None  # Doc store doesn't track dates
            )
            scores.append(score)
            
            print(f"{i}. {result.title}")
            print(f"   URL: {result.source_url}")
            print(f"   Relevance: {result.relevance_score:.3f}")
            print(f"   Quality Score: {score.final_score:.2f} ({score.to_quality_level()})")
            print(f"   Domain Category: {score.domain_category}")
            print()
        
        # Verify scores are valid
        assert all(isinstance(s, SourceScore) for s in scores)
        assert all(0 <= s.final_score <= 1 for s in scores)
        
        # Go.dev should be recognized as official docs
        go_score = next((s for s in scores if "go.dev" in s.url), None)
        if go_score:
            assert go_score.domain_category == "official_docs"
            print("Verified: go.dev recognized as official_docs")
        
        print("="*60)
        print("Integration test passed: Doc search + scoring works!")
        print("="*60)
    
    @pytest.mark.asyncio
    async def test_official_docs_ranked_higher(self):
        """Test that official documentation sources get higher quality scores.
        
        USES REAL OpenAI embeddings - consumes quota!
        """
        print("\n" + "="*60)
        print("INTEGRATION TEST: Official Docs Quality Ranking")
        print("="*60)
        
        client = MCPClient()
        
        # Search should return both official docs and random blog
        results = await client.doc_search("programming", n_results=3)
        
        # Score all results
        scored_results = []
        for result in results:
            score = score_source(
                url=result.source_url,
                content=result.content,
                published_date=None
            )
            scored_results.append((result, score))
        
        # Sort by quality score
        scored_results.sort(key=lambda x: x[1].final_score, reverse=True)
        
        print("\nResults ranked by quality score:")
        for i, (result, score) in enumerate(scored_results, 1):
            is_auth = "*" if is_authoritative(score) else " "
            print(f"{i}. {is_auth} {score.final_score:.2f} - {result.source_url[:50]}...")
        
        # Check that official docs score higher than unknown blog
        official_scores = [s for _, s in scored_results if s.domain_category == "official_docs"]
        unknown_scores = [s for _, s in scored_results if s.domain_category == "unknown"]
        
        if official_scores and unknown_scores:
            assert max(s.final_score for s in official_scores) > max(s.final_score for s in unknown_scores)
            print("\nVerified: Official docs ranked higher than unknown sources")
        
        print("\n" + "="*60)
        print("Integration test passed: Quality ranking works!")
        print("="*60)


class TestErrorScenarios:
    """Integration tests for error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_search_results_handled(self):
        """Test that empty search results are handled gracefully."""
        print("\n" + "="*60)
        print("INTEGRATION TEST: Empty Search Results Handling")
        print("="*60)
        
        # Reset collection to ensure it's empty
        reset_collection()
        
        client = MCPClient()
        
        # Search in empty collection
        results = await client.doc_search("nonexistent topic xyz123", n_results=5)
        
        assert results == []
        print("Empty results handled correctly")
        
        # Scoring empty list should work
        sources = [
            {"url": r.source_url, "content": r.content, "published_date": None}
            for r in results
        ]
        scores = score_sources(sources)
        
        assert scores == []
        print("Scoring empty list works correctly")
        
        print("\n" + "="*60)
        print("Integration test passed: Empty results handled!")
        print("="*60)
    
    @pytest.mark.asyncio
    async def test_search_query_validation(self):
        """Test that invalid queries are rejected."""
        from utils.resilience import ValidationError
        
        print("\n" + "="*60)
        print("INTEGRATION TEST: Query Validation")
        print("="*60)
        
        client = MCPClient()
        
        # Empty query should raise ValidationError
        with pytest.raises(ValidationError):
            await client.web_search("")
        print("Empty query rejected correctly")
        
        # Very long query should be truncated or rejected
        long_query = "x" * 600  # Over 500 char limit
        with pytest.raises(ValidationError):
            await client.web_search(long_query)
        print("Long query rejected correctly")
        
        print("\n" + "="*60)
        print("Integration test passed: Validation works!")
        print("="*60)
    
    def test_score_malformed_urls(self):
        """Test scoring sources with malformed URLs."""
        print("\n" + "="*60)
        print("INTEGRATION TEST: Malformed URL Handling")
        print("="*60)
        
        test_cases = [
            ("", "empty"),
            ("not-a-url", "no scheme"),
            ("ftp://files.example.com/doc.pdf", "ftp scheme"),
            ("https://", "no domain"),
        ]
        
        for url, description in test_cases:
            score = score_source(
                url=url,
                content="Some content for testing.",
                published_date=None
            )
            
            # Should not raise error, should return valid score
            assert 0 <= score.final_score <= 1
            print(f"  {description}: score={score.final_score:.2f}, domain='{score.domain}'")
        
        print("\n" + "="*60)
        print("Integration test passed: Malformed URLs handled!")
        print("="*60)
    
    def test_score_various_date_formats(self):
        """Test scoring with various date formats."""
        print("\n" + "="*60)
        print("INTEGRATION TEST: Date Format Parsing")
        print("="*60)
        
        test_dates = [
            "2026-02-15",
            "2026-02-15T10:30:00Z",
            "2026-02-15T10:30:00.000Z",
            "February 15, 2026",
            "15 Feb 2026",
            "invalid-date",
            None,
        ]
        
        for date_str in test_dates:
            score = score_source(
                url="https://example.com/article",
                content="Some article content " * 100,
                published_date=date_str
            )
            
            days = score.days_since_published
            days_str = str(days) if days is not None else "N/A"
            print(f"  '{date_str}': recency={score.recency_score:.1f}, days={days_str}")
        
        print("\n" + "="*60)
        print("Integration test passed: Date parsing works!")
        print("="*60)


class TestEndToEndPipeline:
    """Full end-to-end integration test."""
    
    @pytest.fixture(autouse=True)
    def setup_documents(self):
        """Set up test documents."""
        reset_collection()
        
        # Ingest a test document
        ingest_document_impl(
            "Python Async Programming Guide",
            """
            Python's asyncio library provides infrastructure for writing single-threaded
            concurrent code using coroutines. The async/await syntax makes asynchronous
            code look and behave like synchronous code, making it easier to understand
            and maintain.
            
            Key concepts in Python async programming:
            
            1. Coroutines: Functions defined with async def that can be paused and resumed
            2. Event Loop: The core of asyncio that runs coroutines and handles I/O
            3. Tasks: Wrappers around coroutines that schedule execution on the event loop
            4. Futures: Objects representing results of asynchronous operations
            
            Best practices for async Python:
            - Use async context managers for resource cleanup
            - Avoid blocking calls in async code
            - Use asyncio.gather() for concurrent operations
            - Handle cancellation properly with try/finally
            """ * 4,
            "https://docs.python.org/3/library/asyncio.html"
        )
        
        yield
    
    @pytest.mark.asyncio
    async def test_full_research_pipeline(self):
        """Test complete research pipeline: search both sources, score, filter.
        
        USES REAL APIs - consumes quota!
        """
        print("\n" + "="*70)
        print("INTEGRATION TEST: Full Research Pipeline")
        print("="*70)
        
        client = MCPClient()
        query = "Python async programming patterns"
        
        # Step 1: Web search
        print(f"\n[Step 1] Web search for: '{query}'")
        web_results = await client.web_search(query, max_results=3)
        print(f"  Got {len(web_results)} web results")
        
        # Step 2: Document search
        print(f"\n[Step 2] Document search for: '{query}'")
        doc_results = await client.doc_search(query, n_results=3)
        print(f"  Got {len(doc_results)} document results")
        
        # Step 3: Combine and score all results
        print("\n[Step 3] Scoring all sources...")
        all_sources = []
        
        for r in web_results:
            all_sources.append({
                "url": r.url,
                "content": r.content,
                "published_date": r.published_date,
                "type": "web"
            })
        
        for r in doc_results:
            all_sources.append({
                "url": r.source_url,
                "content": r.content,
                "published_date": None,
                "type": "doc"
            })
        
        scores = score_sources(all_sources)
        
        print(f"\n  Total sources: {len(scores)}")
        print("\n  Ranked by quality:")
        for i, score in enumerate(scores, 1):
            source_type = all_sources[scores.index(score)].get("type", "?")
            print(f"    {i}. [{source_type}] {score.final_score:.2f} - {score.domain} ({score.domain_category})")
        
        # Step 4: Filter high-quality sources
        print("\n[Step 4] Filtering high-quality sources (>= 0.55)...")
        high_quality = [s for s in scores if s.final_score >= 0.55]
        print(f"  High-quality sources: {len(high_quality)}/{len(scores)}")
        
        # Verify results
        assert len(scores) > 0, "Should have scored some sources"
        assert all(0 <= s.final_score <= 1 for s in scores), "All scores should be valid"
        
        # Summary stats
        avg_score = sum(s.final_score for s in scores) / len(scores) if scores else 0
        authoritative_count = sum(1 for s in scores if is_authoritative(s))
        
        print("\n[Summary]")
        print(f"  Average quality score: {avg_score:.2f}")
        print(f"  Authoritative sources: {authoritative_count}/{len(scores)}")
        
        print("\n" + "="*70)
        print("INTEGRATION TEST PASSED: Full pipeline works end-to-end!")
        print("="*70)


if __name__ == "__main__":
    # Run pytest when this file is executed directly
    pytest.main([__file__, "-v", "-s"])
