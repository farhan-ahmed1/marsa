# MARSA

**Multi-Agent ReSearch Assistant** - A multi-agent system that orchestrates specialized AI agents via LangGraph with MCP-connected data sources to produce well-sourced research reports.

## Overview

MARSA takes complex questions and produces comprehensive reports with transparent agent activity tracking, source quality scoring, and human-in-the-loop checkpoints.

```bash
Query --> Planner --> Researcher(s) --> Fact-Checker --> Synthesizer --> Report
                          |                  |
                          +------------------+ (loop if issues found)
```

**Agents:**

- **Planner**: Decomposes queries, determines strategy (parallel vs sequential, web vs docs)
- **Researcher**: Executes sub-queries via MCP servers, extracts claims
- **Fact-Checker**: Verifies claims with independent searches, scores sources
- **Synthesizer**: Produces final report with inline citations

## Tech Stack

| Layer | Technology |
| ------- | ------------ |
| LLM | Claude API (Anthropic) |
| Orchestration | LangGraph (Python) |
| MCP Servers | fastmcp |
| Vector DB | ChromaDB |
| Embeddings | OpenAI text-embedding-3-small |
| Search | Tavily |
| Backend | FastAPI |
| Frontend | Next.js + Tailwind + shadcn/ui |

## Setup

### Prerequisites

- Python 3.12+
- Node.js 20+
- API keys for:
  - [Anthropic Claude](https://console.anthropic.com/) - LLM for agents
  - [OpenAI](https://platform.openai.com/api-keys) - Embeddings
  - [Tavily](https://tavily.com/) - Web search (free tier: 1,000/month)

### Quick Start

1. **Clone and install dependencies:**

   ```bash
   git clone https://github.com/yourusername/marsa.git
   cd marsa
   make setup
   ```

2. **Configure API keys:**

   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # ANTHROPIC_API_KEY=sk-ant-...
   # OPENAI_API_KEY=sk-...
   # TAVILY_API_KEY=tvly-...
   ```

3. **Ingest sample documents (optional):**

   ```bash
   python backend/scripts/ingest_docs.py
   ```

4. **Run the application:**

   ```bash
   # Terminal 1: Backend
   make run-backend
   
   # Terminal 2: Frontend
   make run-frontend
   ```

5. **Open the UI:** Navigate to `http://localhost:3000`

### Development Commands

| Command | Description |
| --------- | ------------- |
| `make setup` | Install all dependencies (Python + Node.js) |
| `make test` | Run unit tests |
| `make test:integration` | Run integration tests (uses real APIs) |
| `make lint` | Lint code (ruff + eslint) |
| `make format` | Format code |
| `make run-backend` | Start FastAPI server |
| `make run-frontend` | Start Next.js dev server |
| `make docker` | Run with Docker Compose |

### Running MCP Servers Manually

For debugging or development:

```bash
# Tavily Search Server
python -m fastmcp dev backend/mcp_servers/tavily_search.py

# Document Store Server
python -m fastmcp dev backend/mcp_servers/document_store.py
```

### Testing the Pipeline

```bash
# Run a manual test
python backend/manual_test_mcp.py

# Run the full test suite
make test
```

## Project Structure

```bash
marsa/
+-- backend/
|   +-- agents/           # AI agents (planner, researcher, fact_checker, synthesizer)
|   +-- mcp_servers/      # MCP server implementations
|   +-- graph/            # LangGraph workflow and state
|   +-- api/              # FastAPI endpoints
|   +-- tests/            # Unit and integration tests
|   +-- mcp_client.py     # Unified MCP client wrapper
|   +-- config.py         # Configuration management
+-- frontend/
|   +-- src/app/          # Next.js pages
|   +-- src/components/   # React components
|   +-- src/hooks/        # Custom hooks (useAgentStream)
+-- data/
|   +-- chromadb/         # Vector database storage
|   +-- sample_docs/      # Sample documents for ingestion
+-- docs/
|   +-- architecture.md   # System architecture
|   +-- design-decisions.md # Design rationale
```

## Architecture

### MCP Servers

MARSA uses Model Context Protocol (MCP) servers as the data access layer:

1. **Tavily Search** (`tavily_search.py`) - Web search with rate limiting
2. **Document Store** (`document_store.py`) - ChromaDB vector search with OpenAI embeddings

Agents interact with these servers through a unified client (`mcp_client.py`) that handles connection management, error handling, and retries.

### Source Quality Scoring

Not all sources are equal. MARSA scores source quality based on:

- **Domain authority** (40%): `.gov`/`.edu` = 0.9, arxiv/ACM = 0.85, docs = 0.8, blogs = 0.6
- **Recency** (30%): <3 months = 1.0, <6 months = 0.8, <1 year = 0.6
- **Content depth** (30%): >2000 words = 0.8, >500 words = 0.6

See [docs/design-decisions.md](docs/design-decisions.md) for methodology details.

## Documentation

- [Architecture](docs/architecture.md) - System design and data flow
- [Design Decisions](docs/design-decisions.md) - Rationale for technical choices
- [Setup Guide](docs/setup.md) - Detailed installation instructions

## CI/CD

The project uses GitHub Actions for automated testing and linting. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

## License

MIT

---

**Note:** This project is in active development.
