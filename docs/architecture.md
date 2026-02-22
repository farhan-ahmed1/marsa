# Architecture

This document describes the system architecture of MARSA (Multi-Agent ReSearch Assistant), including the overall system design, LangGraph workflow, data flow, and MCP server interactions.

## Table of Contents

- [System Overview](#system-overview)
- [LangGraph Workflow](#langgraph-workflow)
- [Data Flow](#data-flow)
- [MCP Server Architecture](#mcp-server-architecture)
- [State Management](#state-management)
- [Observability](#observability)

---

## System Overview

MARSA is a three-tier application with a Next.js frontend, FastAPI backend, and MCP-connected data sources.

```mermaid
graph TB
    subgraph Client["Client Layer"]
        Browser[Web Browser]
    end

    subgraph Frontend["Frontend (Next.js)"]
        Pages[App Router Pages]
        Components[React Components]
        Hooks[useAgentStream Hook]
    end

    subgraph Backend["Backend (FastAPI)"]
        API["/api/query<br>/api/stream/{id}"]
        
        subgraph Orchestration["LangGraph Orchestration"]
            Graph[StateGraph]
            Checkpointer[SQLite/Memory<br>Checkpointer]
        end
        
        subgraph Agents["Agent Layer"]
            Planner[Planner Agent]
            Researcher[Researcher Agent]
            FactChecker[Fact-Checker Agent]
            Synthesizer[Synthesizer Agent]
        end
        
        MCPClient[MCP Client Wrapper]
        Memory[Cross-Session Memory]
    end

    subgraph DataLayer["Data Layer (MCP Servers)"]
        TavilyMCP[Tavily Search Server]
        DocStoreMCP[Document Store Server]
    end

    subgraph External["External Services"]
        Claude[Claude API]
        OpenAI[OpenAI Embeddings]
        Tavily[Tavily Search API]
        ChromaDB[(ChromaDB)]
    end

    Browser -->|HTTP/SSE| Frontend
    Pages --> Components
    Components --> Hooks
    Hooks -->|SSE Stream| API
    
    API --> Graph
    Graph --> Checkpointer
    Graph --> Planner
    Graph --> Researcher
    Graph --> FactChecker
    Graph --> Synthesizer
    
    Planner --> Claude
    Researcher --> MCPClient
    FactChecker --> MCPClient
    Synthesizer --> Claude
    
    MCPClient -->|MCP Protocol| TavilyMCP
    MCPClient -->|MCP Protocol| DocStoreMCP
    
    TavilyMCP --> Tavily
    DocStoreMCP --> OpenAI
    DocStoreMCP --> ChromaDB
    
    Planner --> Memory
    Memory --> ChromaDB
```

### Component Responsibilities

| Component | Responsibility |
| ----------- | ---------------- |
| **Next.js Frontend** | Query input, real-time agent trace display, report rendering |
| **FastAPI Backend** | HTTP API, SSE streaming, workflow orchestration |
| **LangGraph** | State machine execution, conditional routing, checkpointing |
| **Agents** | Specialized AI reasoning (planning, research, verification, synthesis) |
| **MCP Client** | Unified interface to MCP servers with retry logic |
| **MCP Servers** | Standardized data access via Model Context Protocol |
| **Cross-Session Memory** | Retrieval of prior research for related queries |

---

## LangGraph Workflow

The research pipeline is implemented as a LangGraph StateGraph with conditional routing and optional human-in-the-loop checkpoints.

### Sequential Mode

Default workflow for simple queries:

```mermaid
stateDiagram-v2
    [*] --> Planner
    Planner --> Researcher: plan created
    Researcher --> FactChecker: claims extracted
    FactChecker --> Researcher: >30% claims failed
    FactChecker --> Synthesizer: verification passed
    Synthesizer --> StoreMemory
    StoreMemory --> [*]
```

### Parallel Mode

For multi-faceted queries (e.g., comparisons), sub-queries fan out to parallel workers:

```mermaid
stateDiagram-v2
    [*] --> Planner
    
    state routing <<choice>>
    Planner --> routing
    routing --> Sequential: plan.parallel = false
    routing --> FanOut: plan.parallel = true
    
    state FanOut {
        SubQuery1: Research Sub-Query 1
        SubQuery2: Research Sub-Query 2
        SubQueryN: Research Sub-Query N
    }
    
    FanOut --> MergeResults
    Sequential --> FactChecker
    MergeResults --> FactChecker
    
    state fact_check <<choice>>
    FactChecker --> fact_check
    fact_check --> Sequential: loop back
    fact_check --> Synthesizer: pass
    
    Synthesizer --> StoreMemory
    StoreMemory --> [*]
```

### Human-in-the-Loop Mode

When `enable_hitl=True`, the workflow pauses after fact-checking for user feedback:

```mermaid
stateDiagram-v2
    [*] --> Planner
    Planner --> Researcher
    Researcher --> FactChecker
    
    FactChecker --> HITL_Checkpoint: interrupt
    
    note right of HITL_Checkpoint
        User reviews claims and
        provides feedback:
        - Approve
        - Dig Deeper
        - Abort
    end note
    
    state hitl_routing <<choice>>
    HITL_Checkpoint --> hitl_routing
    hitl_routing --> Synthesizer: approve
    hitl_routing --> Planner: dig_deeper
    hitl_routing --> [*]: abort
    
    Synthesizer --> StoreMemory
    StoreMemory --> [*]
```

### Workflow Graph Definition

The workflow is defined in [backend/graph/workflow.py](../backend/graph/workflow.py):

```python
# Core nodes
workflow.add_node("planner", planner_with_trace)
workflow.add_node("research_sequential", researcher_with_status)
workflow.add_node("research_sub_query", research_sub_query_node)  # Parallel worker
workflow.add_node("merge_research", merge_research_with_status)
workflow.add_node("fact_checker", fact_checker_with_status)
workflow.add_node("synthesizer", synthesizer_with_status)
workflow.add_node("store_memory", store_memory_node)

# Conditional routing for parallel execution
workflow.add_conditional_edges(
    "planner",
    route_sub_queries,  # Returns Send objects for parallel fan-out
    {"research_sequential": "research_sequential"}
)

# Fact-check loop-back condition
workflow.add_conditional_edges(
    "fact_checker",
    route_after_fact_check,
    {"researcher": "research_sequential", "synthesizer": "synthesizer"}
)
```

---

## Data Flow

### Query Processing Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend API
    participant G as LangGraph
    participant P as Planner
    participant R as Researcher
    participant M as MCP Client
    participant T as Tavily MCP
    participant D as DocStore MCP
    participant FC as Fact-Checker
    participant S as Synthesizer

    U->>F: Submit research query
    F->>B: POST /api/query
    B->>G: invoke(initial_state)
    
    G->>P: planner_node(state)
    P-->>G: QueryPlan (sub_queries, parallel, search_strategy)
    
    alt Parallel Execution
        G->>R: Send("research_sub_query", sq1)
        G->>R: Send("research_sub_query", sq2)
        G->>R: Send("research_sub_query", sqN)
        R->>M: search(sub_query)
        M->>T: MCP tools/call search
        T-->>M: SearchResults
        M->>D: MCP tools/call search_documents
        D-->>M: DocumentResults
        R-->>G: ResearchResults + Claims
        G->>G: merge_research_node()
    else Sequential Execution
        G->>R: research_node(state)
        R->>M: search(sub_query)
        R-->>G: ResearchResults + Claims
    end
    
    G->>FC: fact_check_node(state)
    FC->>M: search(verification_query)
    FC-->>G: VerificationResults + SourceScores
    
    alt Claims Failed > 30%
        G->>R: Loop back for re-research
    else Claims Verified
        G->>S: synthesize_node(state)
        S-->>G: Final Report + Citations
    end
    
    G-->>B: Final AgentState
    B-->>F: SSE events (agent_trace, report)
    F-->>U: Display report with sources
```

### State Transformation

Each agent transforms specific fields in the `AgentState`:

```mermaid
flowchart LR
    subgraph Input
        Q[query: str]
    end
    
    subgraph Planner
        QP[plan: QueryPlan]
        SQ[sub_queries: list]
    end
    
    subgraph Researcher
        RR[research_results: list]
        CL[claims: list]
    end
    
    subgraph FactChecker
        VR[verification_results: list]
        SS[source_scores: dict]
    end
    
    subgraph Synthesizer
        RP[report: str]
        CT[citations: list]
    end
    
    Q --> Planner
    Planner --> Researcher
    Researcher --> FactChecker
    FactChecker --> Synthesizer
```

---

## MCP Server Architecture

MARSA uses the Model Context Protocol (MCP) to standardize data access. Agents interact with MCP servers through a unified client wrapper.

### MCP Integration

```mermaid
graph LR
    subgraph Agents
        R[Researcher]
        FC[Fact-Checker]
    end
    
    subgraph MCPClient["MCP Client (mcp_client.py)"]
        WS[web_search]
        DS[doc_search]
        Retry[Retry Logic]
        Pool[Connection Pool]
    end
    
    subgraph MCPServers["MCP Servers"]
        subgraph TavilyServer["tavily_search.py"]
            ST[search tool]
            RL[Rate Limiter]
        end
        
        subgraph DocStoreServer["document_store.py"]
            SD[search_documents tool]
            ID[ingest_document tool]
            LD[list_documents tool]
        end
    end
    
    subgraph External
        TAV[Tavily API]
        OAI[OpenAI API]
        CDB[(ChromaDB)]
    end
    
    R --> WS
    R --> DS
    FC --> WS
    
    WS --> ST
    DS --> SD
    
    ST --> RL --> TAV
    SD --> OAI
    SD --> CDB
    ID --> OAI
    ID --> CDB
```

### MCP Server Tools

#### Tavily Search Server

| Tool | Parameters | Returns |
| ---------- | ------------ | --------- |
| `search` | `query: str`, `max_results: int = 5` | `list[SearchResult]` with title, url, content, score |

#### Document Store Server

| Tool | Parameters | Returns |
| ------ | ------------ | --------- |
| `search_documents` | `query: str`, `n_results: int = 5` | `list[DocumentResult]` with content, source, relevance_score |
| `ingest_document` | `title: str`, `content: str`, `source_url: str` | `IngestResult` with document_id, chunk_count |
| `list_documents` | - | `list[DocumentSummary]` with title, source, chunk_count |

### MCP Protocol Flow

```mermaid
sequenceDiagram
    participant A as Agent
    participant C as MCP Client
    participant S as MCP Server
    participant E as External API

    A->>C: web_search("query")
    C->>S: JSON-RPC request<br>{"method": "tools/call", "params": {"name": "search", "arguments": {...}}}
    S->>E: API call (Tavily/OpenAI)
    E-->>S: Raw response
    S-->>C: JSON-RPC response<br>{"result": {"content": [...]}}
    C-->>A: Parsed SearchResult objects
```

---

## State Management

### AgentState Schema

The `AgentState` TypedDict flows through the LangGraph workflow:

```mermaid
classDiagram
    class AgentState {
        +str query
        +QueryPlan plan
        +list~str~ sub_queries
        +list~ResearchResult~ research_results
        +list~Claim~ claims
        +list~VerificationResult~ verification_results
        +dict~str,float~ source_scores
        +str report
        +list~Citation~ citations
        +list~TraceEvent~ agent_trace
        +int iteration_count
        +str status
        +list~str~ errors
        +str memory_context
        +HITLFeedback hitl_feedback
    }
    
    class QueryPlan {
        +QueryType query_type
        +list~str~ sub_queries
        +bool parallel
        +SearchStrategy search_strategy
        +ComplexityLevel complexity
        +str reasoning
    }
    
    class Claim {
        +str text
        +str source_url
        +ConfidenceLevel confidence
        +str supporting_text
    }
    
    class VerificationResult {
        +str claim_text
        +VerificationVerdict verdict
        +str evidence
        +float confidence_score
        +list~str~ sources_checked
    }
    
    class TraceEvent {
        +AgentName agent
        +str action
        +str detail
        +datetime timestamp
        +int latency_ms
        +dict metadata
    }
    
    AgentState --> QueryPlan
    AgentState --> Claim
    AgentState --> VerificationResult
    AgentState --> TraceEvent
```

### Checkpointing

LangGraph checkpointing enables:

1. **State inspection**: View what each agent produced
2. **Workflow resumption**: Resume interrupted workflows (critical for HITL)
3. **Debugging**: Replay workflows from any checkpoint

```python
# Memory checkpointer (development)
checkpointer = InMemorySaver()

# SQLite checkpointer (production)
checkpointer = SqliteSaver(db_path="checkpoints.db")

# Compile with checkpointer
app = workflow.compile(checkpointer=checkpointer)
```

---

## Observability

### Trace Event Pipeline

Every agent action generates a `TraceEvent` that flows to the frontend via SSE:

```mermaid
flowchart LR
    subgraph Agents
        A1[Planner]
        A2[Researcher]
        A3[Fact-Checker]
        A4[Synthesizer]
    end
    
    subgraph State
        AT[agent_trace: list]
    end
    
    subgraph Backend
        SSE[SSE Streamer]
    end
    
    subgraph Frontend
        TL[Timeline Component]
        TC[Trace Cards]
    end
    
    A1 -->|TraceEvent| AT
    A2 -->|TraceEvent| AT
    A3 -->|TraceEvent| AT
    A4 -->|TraceEvent| AT
    
    AT -->|State updates| SSE
    SSE -->|agent_trace events| TL
    TL --> TC
```

### Event Types

| Event Type | Description | Example |
| ------------ | ------------- | --------- |
| `agent_started` | Agent begins processing | Planner begins query decomposition |
| `tool_called` | MCP tool invocation | Researcher calls web_search |
| `tool_result` | MCP tool response received | 5 search results returned |
| `claim_extracted` | Claim extracted from research | Claim about Python GIL |
| `claim_verified` | Fact-check verdict | Claim supported with evidence |
| `report_generating` | Synthesis in progress | Generating final report |
| `complete` | Workflow finished | Report ready with 8 citations |
| `error` | Error occurred | Rate limit exceeded |

### LangSmith Integration

For production observability, enable LangSmith tracing:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-key
```

LangSmith captures:

- Full LLM prompt/response pairs
- Token usage and costs
- Latency breakdown by component
- Error traces and retries

---

## Related Documentation

- [Design Decisions](design-decisions.md) - Rationale for architectural choices
- [Setup Guide](setup.md) - Installation and configuration
- [API Reference](../backend/api/routes.py) - Endpoint documentation
