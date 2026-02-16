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

---

## State Schema Design

### AgentState TypedDict

The `AgentState` is a TypedDict that flows through the LangGraph workflow. We chose TypedDict over Pydantic for the top-level state because:

- **LangGraph compatibility**: LangGraph works natively with TypedDicts for state management
- **Mutability**: State needs to be updated by each agent node
- **Performance**: Avoids serialization overhead on every state transition

### Nested Pydantic Models

While the top-level state is a TypedDict, we use Pydantic models for nested structures:

```python
class AgentState(TypedDict):
    query: str
    plan: QueryPlan           # Pydantic model
    claims: list[Claim]       # List of Pydantic models
    # ...
```

**Benefits of this hybrid approach:**

1. **Type safety**: Pydantic provides runtime validation for complex structures
2. **Serialization**: Easy JSON serialization for persistence and logging
3. **Documentation**: Field descriptions serve as documentation
4. **IDE support**: Type hints provide autocompletion and error checking

### Key State Fields

| Field | Type | Purpose |
| ------- | ------ | --------- |
| `query` | `str` | Original user query |
| `plan` | `QueryPlan` | Planner's decomposition strategy |
| `sub_queries` | `list[str]` | Decomposed research questions |
| `research_results` | `list[ResearchResult]` | Raw findings from searches |
| `claims` | `list[Claim]` | Extracted claims from research |
| `verification_results` | `list[VerificationResult]` | Fact-check verdicts |
| `source_scores` | `dict[str, float]` | URL to quality score mapping |
| `report` | `str` | Final synthesized report |
| `citations` | `list[Citation]` | Numbered source references |
| `agent_trace` | `list[TraceEvent]` | Observability timeline |
| `iteration_count` | `int` | Loop guard counter |
| `status` | `str` | Pipeline status for UI |
| `errors` | `list[str]` | Error accumulator |

### Design Rationale

#### Flat state over nested state**

We use a flat state structure where each agent reads and writes specific fields:

- **Pro**: Clearer data flow, easier debugging
- **Pro**: Agents are decoupled from each other
- **Con**: State object grows as pipeline expands

#### Immutable approach with updates**

Each agent returns a new state dict with updates, rather than mutating in place:

```python
async def research_node(state: AgentState) -> AgentState:
    results = await do_research(state["plan"])
    return {**state, "research_results": results}
```

#### Trace events for observability**

Every significant action appends to `agent_trace`:

```python
TraceEvent(
    agent="researcher",
    action="web_search",
    detail=f"Searching: {sub_query}",
    timestamp=datetime.utcnow(),
    latency_ms=elapsed,
)
```

This enables the real-time UI timeline and post-hoc debugging.

---

## Source Quality Scoring Methodology

### Problem Statement

Not all sources are equally reliable. A claim from a government website or academic paper should carry more weight than one from an unknown blog. We need a systematic way to score source quality.

### Scoring Factors

We use a weighted combination of three factors:

| Factor | Weight | Rationale |
| -------- | -------- | ----------- |
| Domain Authority | 40% | Institutional credibility matters most |
| Recency | 30% | Fresh information is often more relevant |
| Content Depth | 30% | Comprehensive content suggests expertise |

### Domain Authority Scoring

Sources are categorized into tiers based on their domain:

| Category | Score | Examples |
| ---------- | ------- | ---------- |
| Government/Academic | 0.90 | `.gov`, `.edu`, government sites |
| Publications | 0.85 | arXiv, ACM, IEEE, Nature, major news |
| Official Documentation | 0.80-0.85 | docs.python.org, go.dev, MDN |
| Established Tech Blogs | 0.65-0.75 | Martin Fowler, company engineering blogs |
| Community Sites | 0.50-0.65 | StackOverflow, Medium, dev.to |
| Unknown | 0.40 | Unrecognized domains |

**Implementation approach:**

```python
# Domain lookup tables by tier
DOMAIN_SCORES_TLD = {".gov": 0.9, ".edu": 0.9, ...}
DOMAIN_SCORES_PUBLICATIONS = {"arxiv.org": 0.85, "acm.org": 0.85, ...}
DOMAIN_SCORES_OFFICIAL = {"docs.python.org": 0.85, "go.dev": 0.85, ...}

def _get_domain_score(domain: str, url: str) -> tuple[float, str]:
    # Check each tier in priority order
    # Return (score, category)
```

### Recency Scoring

Content freshness affects relevance, especially for fast-moving fields:

| Age | Score | Rationale |
| ----- | ------- | ----------- |
| 0-3 months | 1.0 | Very current |
| 3-6 months | 0.8 | Recent |
| 6-12 months | 0.6 | Somewhat dated |
| >12 months | 0.4 | May be outdated |

**Handling missing dates:**

When no publication date is available (common for documentation and evergreen content), we default to 0.6 (6-12 month equivalent) as a conservative middle ground.

### Content Depth Scoring

Longer, more comprehensive content often indicates deeper analysis:

| Word Count | Score | Rationale |
| ------------ | ------- | ----------- |
| >2000 words | 0.8 | In-depth article or documentation |
| 500-2000 | 0.6 | Standard article |
| <500 | 0.4 | Snippet or summary |

### Final Score Calculation

```python
final_score = (
    WEIGHT_DOMAIN * domain_score +        # 0.4
    WEIGHT_RECENCY * recency_score +      # 0.3
    WEIGHT_DEPTH * depth_score            # 0.3
)
```

### Usage in the Pipeline

The source scorer integrates at multiple points:

1. **During research**: Score sources as they're retrieved
2. **During fact-checking**: Weight evidence by source quality
3. **During synthesis**: Prioritize claims from higher-quality sources
4. **In the final report**: Display quality badges on citations

### Future Improvements

1. **ML-based scoring**: Train a model on human-rated sources
2. **Citation analysis**: Sources cited by other high-quality sources get boost
3. **Content analysis**: NLP to detect opinion vs. fact, bias indicators
4. **Dynamic updates**: Track source quality over time
5. **User preferences**: Allow domain-specific scoring adjustments
