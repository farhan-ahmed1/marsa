"""Manual test script for MCP servers.

Run this to verify both the hello and Tavily search servers work.
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Import after modifying sys.path
from mcp_servers.hello import greet_impl  # noqa: E402
from mcp_servers.tavily_search import search_impl  # noqa: E402


def test_hello_server():
    """Test the hello world MCP server."""
    print("=" * 60)
    print("Testing Hello World MCP Server")
    print("=" * 60)
    
    test_names = ["Alice", "Bob", "World"]
    for name in test_names:
        result = greet_impl(name)
        print(f"greet('{name}') -> {result}")
    
    print("\n✅ Hello server working!\n")


def test_tavily_search_server():
    """Test the Tavily search MCP server."""
    print("=" * 60)
    print("Testing Tavily Search MCP Server")
    print("=" * 60)
    
    query = "latest developments in Rust programming language 2026"
    print(f"\nSearching for: '{query}'")
    print("-" * 60)
    
    results = search_impl(query, max_results=3)
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   Score: {result.score:.3f}")
        print(f"   Published: {result.published_date or 'N/A'}")
        print(f"   Content: {result.content[:100]}...")
        print()
    
    print("✅ Tavily search server working!\n")


if __name__ == "__main__":
    test_hello_server()
    test_tavily_search_server()
    
    print("=" * 60)
    print("All MCP servers tested successfully!")
    print("=" * 60)
