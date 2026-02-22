# Design Decisions

This document explains the rationale behind key technical decisions in MARSA. Each section covers the problem, the alternatives considered, and why we chose a particular approach.

## Table of Contents

- [Orchestration Framework](#orchestration-framework)
- [MCP Architecture](#mcp-architecture)
- [State Schema Design](#state-schema-design)
- [Parallel Execution](#parallel-execution)
- [Human-in-the-Loop Design](#human-in-the-loop-design)
- [Source Quality Scoring](#source-quality-scoring-methodology)
- [Cross-Session Memory](#cross-session-memory)
- [Trade-offs and Future Improvements](#trade-offs-and-future-improvements)

---

## Orchestration Framework

### Decision: LangGraph over CrewAI, AutoGen, and Custom Solutions

**Context**: We needed a framework to orchestrate multiple AI agents through a research pipeline with conditional routing, loops, and state persistence.

**Alternatives Considered**:

| Framework | Pros | Cons |
| ----------- | ------ | ------ |
| **LangGraph** | Explicit state machine, conditional edges, checkpointing, mature | Steeper learning curve |
| **CrewAI** | Simple agent definitions, role-based | Less control over flow, implicit orchestration |
| **AutoGen** | Multi-agent conversations, Microsoft backing | Conversation-centric, harder to enforce pipeline |
| **Custom** | Full control | Significant implementation effort for state, routing, persistence |

**Decision**: LangGraph

**Rationale**:

1. **Explicit control flow**: Research pipelines need deterministic routing. LangGraph's `add_conditional_edges` lets us define exactly when to loop back (failed verification) or proceed (claims verified).

2. **State machine model**: The StateGraph abstraction matches our mental model - each agent is a node that transforms state. This makes the pipeline easy to reason about and debug.

3. **Built-in checkpointing**: SQLite and memory-based checkpointers give us state persistence "for free." This is critical for human-in-the-loop workflows and debugging.

4. **Parallel execution**: The `Send` API enables fan-out to parallel workers, which maps directly to our parallel sub-query research feature.

5. **LangChain ecosystem**: Integration with LangSmith for observability, familiar tooling patterns, strong documentation.

**Trade-off accepted**: LangGraph has a steeper learning curve than CrewAI. However, the explicit control is worth it for a production-grade system where we need to understand exactly what's happening.

---

## MCP Architecture

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

## Parallel Execution

### Decision: LangGraph Send API for Fan-Out/Merge

**Context**: Multi-faceted queries like "Compare Rust vs Go for distributed systems" benefit from researching each aspect in parallel rather than sequentially.

**Alternatives Considered**:

| Approach | Pros | Cons |
| ---------- | ------ | ------ |
| **Sequential** | Simple, predictable | Slow for 3+ sub-queries |
| **asyncio.gather** | Fast, built-in | Complex state merging, no checkpointing |
| **LangGraph Send** | Native to framework, checkpointed, clean merge | Requires understanding Send API |
| **Celery/Ray** | Distributed execution | Over-engineered for this scale |

**Decision**: LangGraph Send API

**Implementation**:

```python
def route_sub_queries(state: AgentState) -> Union[list[Send], Literal["research_sequential"]]:
    """Route sub-queries for parallel or sequential execution."""
    plan = state.get("plan")
    
    if plan.parallel and len(plan.sub_queries) >= 2:
        # Fan out to parallel workers
        return [
            Send("research_sub_query", {"sub_query": sq, "parent_state": state})
            for sq in plan.sub_queries
        ]
    
    return "research_sequential"
```

**Rationale**:

1. **Framework-native**: Send API is built for this exact use case. Each parallel branch gets its own state copy, and LangGraph handles the merge.

2. **Checkpointed**: Each parallel execution is captured in the checkpoint, enabling debugging and replay.

3. **Adaptive**: The Planner decides whether to use parallel execution based on query complexity. Simple queries run sequentially to avoid overhead.

4. **Spool connection**: This pattern mirrors distributed task systems - fan out work to workers, merge results. It demonstrates understanding of distributed systems principles.

**Performance impact**: For a 3-way comparison query, parallel execution reduces latency from ~30s (sequential) to ~12s (parallel, limited by slowest sub-query).

---

## Human-in-the-Loop Design

### Decision: Interrupt After Fact-Checking, Before Synthesis

**Context**: For critical research, users may want to review claims and provide feedback before the final report is generated.

**Design Choices**:

| Checkpoint Location | Pros | Cons |
| --------------------- | ------ | ------ |
| After planning | User can modify sub-queries | Too early, no research to review |
| After research | Review raw findings | Claims not yet verified |
| **After fact-checking** | Review verified claims | Optimal - all verification done |
| After synthesis | Review final report | Too late to influence research |

**Decision**: Interrupt after fact-checking

**Implementation**:

```python
# Compile with interrupt
compile_kwargs = {
    "checkpointer": checkpointer,
    "interrupt_after": ["fact_checker"]  # Pause after fact-checking
}
app = workflow.compile(**compile_kwargs)
```

**Feedback Actions**:

| Action | Behavior |
| -------- | ---------- |
| `approve` | Proceed to synthesis with current claims |
| `dig_deeper` | Return to planner with new focus topic |
| `abort` | End workflow immediately |

**Critical Implementation Detail**: The same `InMemorySaver` instance must be used for both the initial run and the resume. Creating a new checkpointer would lose all state:

```python
# Module-level singleton - CRITICAL for HITL
_shared_memory_checkpointer: Optional[InMemorySaver] = None

def get_shared_checkpointer() -> InMemorySaver:
    global _shared_memory_checkpointer
    if _shared_memory_checkpointer is None:
        _shared_memory_checkpointer = InMemorySaver()
    return _shared_memory_checkpointer
```

**Rationale**:

1. **Informed decision**: Users see verified claims with confidence scores, not raw research dump.

2. **Actionable feedback**: "Dig deeper" loops back to planning with user's focus area, enabling iterative refinement.

3. **Non-blocking for simple queries**: HITL is opt-in via `enable_hitl=True`. Default workflows run to completion.

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

---

## Cross-Session Memory

### Decision: ChromaDB-Based Memory with Similarity Retrieval

**Context**: Follow-up queries often relate to prior research. Rather than starting fresh, we can retrieve relevant verified findings from past sessions.

**Design Choices**:

| Approach | Pros | Cons |
| ---------- | ------ | ------ |
| **No memory** | Simple, stateless | Redundant research on related queries |
| **Full session replay** | Complete context | Too much context, slow, expensive |
| **Keyword matching** | Fast, simple | Misses semantic similarity |
| **Vector similarity** | Semantic understanding | Requires embedding + retrieval |

**Decision**: Vector similarity search over prior verified claims

**Implementation**:

```python
def get_relevant_memories(query: str, threshold: float = 0.7) -> str:
    """Retrieve relevant prior research context."""
    # Embed the query
    query_embedding = embed_text(query)
    
    # Search memory collection
    results = memory_collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"verification_status": "verified"}
    )
    
    # Filter by similarity threshold and format
    return format_memory_context(results, threshold)
```

**What Gets Stored**:

| Field | Description |
| ------- | ------------- |
| `topics` | Main topics from the research |
| `verified_claims` | Claims that passed fact-checking |
| `source_quality` | Average source quality score |
| `timestamp` | When research was conducted |

**Rationale**:

1. **Reduced redundancy**: If user asked about "Rust concurrency" before, a follow-up on "Rust async/await" retrieves relevant prior findings.

2. **Verified only**: We only store claims that passed fact-checking, avoiding propagation of unverified information.

3. **Threshold filtering**: Low-similarity matches are filtered out to avoid irrelevant context pollution.

4. **Fire-and-forget storage**: Memory storage happens after synthesis and doesn't block the main workflow.

---

## Trade-offs and Future Improvements

### Current Trade-offs

| Decision | Trade-off | Mitigation |
| ---------- | ----------- | ------------ |
| **InMemorySaver for dev** | State lost on restart | SQLite for production |
| **Stdio MCP transport** | Single-process only | HTTP transport for distributed deployment |
| **Sequential LLM calls in fact-checking** | Slower than parallel | Acceptable for accuracy; parallel in future |
| **Fixed source scoring weights** | May not fit all domains | User-configurable weights planned |
| **Single loop-back iteration** | May miss deep issues | Capped to prevent infinite loops; user can "dig deeper" |

### What I'd Change in Production

1. **Distributed MCP servers**: Use HTTP transport with load balancing for high availability and horizontal scaling.

2. **Redis-backed checkpointing**: For multi-instance deployments where InMemorySaver won't work.

3. **Streaming LLM responses**: Stream partial results to the frontend for better perceived latency.

4. **Async fact-checking**: Verify claims in parallel batches rather than sequentially.

5. **ML-based source scoring**: Train a classifier on human-rated sources rather than hand-tuned heuristics.

6. **A/B testing framework**: Compare different planner prompts or scoring weights on evaluation metrics.

7. **Cost tracking**: Per-query token usage tracking with budget limits.

8. **Rate limiting at API layer**: Protect against abuse with per-user rate limits.

### Known Limitations

1. **Tavily free tier**: 1,000 searches/month limits sustained usage. Production would need paid tier or search API rotation.

2. **Context window limits**: Very long research sessions may exceed Claude's context window. Would need summarization or chunking.

3. **No image/PDF support**: Currently text-only. Future: add document parsing MCP server.

4. **Single-language**: English only. Future: multilingual research capabilities.

---

## Related Documentation

- [Architecture](architecture.md) - System diagrams and data flow
- [Setup Guide](setup.md) - Installation and configuration
