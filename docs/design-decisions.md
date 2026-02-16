# Design Decisions

## MCP (Model Context Protocol)

### What is MCP?

The Model Context Protocol is a protocol developed by Anthropic to standardize how AI applications connect to external data sources and tools. It provides a common interface for LLMs to access:

- **Tools**: Functions that the server exposes (e.g., search, calculate, fetch data)
- **Resources**: Static or dynamic data that the server provides (e.g., files, documents)
- **Prompts**: Templated prompts that can be reused across different contexts

### Server Lifecycle

MCP servers follow a standard lifecycle:

1. **Initialization**: Server starts and registers its capabilities (tools, resources, prompts)
2. **Connection**: Client connects via stdio, HTTP, or other transports
3. **Discovery**: Client queries available tools/resources/prompts
4. **Execution**: Client invokes tools, requests resources, or uses prompts
5. **Shutdown**: Graceful cleanup of resources

### Request/Response Format

MCP uses JSON-RPC 2.0 for communication:

```json
// Request
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {
      "query": "rust programming",
      "max_results": 5
    }
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [...]
  }
}
```

### Why MCP for MARSA?

1. **Standardization**: Rather than building custom integrations for each data source, MCP provides a consistent interface
2. **Modularity**: Each data source (Tavily search, ChromaDB, GitHub) is an independent MCP server that can be tested, deployed, and updated separately
3. **Tool Use**: MCP servers naturally expose tools that LangGraph agents can call - this maps perfectly to our multi-agent architecture
4. **Future-Proofing**: As MCP becomes more widely adopted, we can easily integrate with new MCP-compatible services
5. **Testability**: Each MCP server can be tested independently without running the full agent pipeline

### Tool Definitions

Tools in MCP are defined using Python decorators (with fastmcp):

```python
@mcp.tool()
def search(query: str, max_results: int = 5) -> list[dict]:
    """Search for information on the web.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, content
    """
    # Implementation
```

The decorator automatically:

- Generates JSON schema from type hints
- Handles request/response serialization
- Validates input parameters
- Provides error handling

### Resource vs Tool Decision

For MARSA, we chose to implement everything as **tools** rather than resources because:

- **Tools** are better for dynamic data that requires parameters (search queries, document lookups)
- **Resources** are better for static or slowly-changing data (configuration files, knowledge base metadata)
- Our use case requires parameterized queries, making tools the natural fit

### FastMCP Library

We use `fastmcp` (Anthropic's official Python SDK) because:

- Battle-tested by Anthropic
- Clean decorator-based API
- Built-in development server (`python -m fastmcp dev`)
- Type-safe with Pydantic integration
- Minimal boilerplate

### MCP Server Architecture for MARSA

We implement three MCP servers:

1. **Tavily Search Server** (`tavily_search.py`)
   - Tool: `search(query, max_results)` → web search results
   - Wraps Tavily API in MCP protocol
   - Real-time web data for current information

2. **Document Store Server** (`document_store.py`)
   - Tool: `search_documents(query, n_results)` → vector similarity search
   - Tool: `ingest_document(title, content, source_url)` → store new documents
   - Tool: `list_documents()` → metadata about stored docs
   - ChromaDB backend for persistent knowledge base

3. **GitHub Server** (`github_server.py`) - Optional
   - Tool: `search_code(query, repo)` → search code in repositories
   - Tool: `get_issues(repo, state)` → fetch open/closed issues
   - Useful for technical research queries about open source projects

### Design Trade-offs

**Decision**: Use stdio transport for MCP servers in development

- **Pro**: Simple, no network configuration needed
- **Pro**: Built-in to fastmcp
- **Con**: Not suitable for distributed deployment
- **Future**: Switch to HTTP transport for production

**Decision**: One MCP server per data source (not a unified server)

- **Pro**: Independent deployment and testing
- **Pro**: Easier to debug and optimize each source
- **Pro**: Failure isolation (one server down doesn't kill all data access)
- **Con**: More processes to manage
- **Con**: More complex connection management in agents

**Decision**: Synchronous tool calls initially

- **Pro**: Simpler to implement and reason about
- **Pro**: FastMCP handles sync functions cleanly
- **Con**: Can't run multiple searches in parallel within one MCP call
- **Future**: Consider async implementation for parallel execution

### Integration with LangGraph

LangGraph agents call MCP tools through a unified client wrapper (`backend/mcp_client.py`):

```python
# Agents don't call MCP servers directly
results = await mcp_client.web_search("query")  

# The wrapper handles:
# - MCP server connection management
# - Request/response serialization
# - Error handling and retries
# - Connection pooling
```

This abstraction keeps agent code clean and makes it easy to swap MCP implementations later.
