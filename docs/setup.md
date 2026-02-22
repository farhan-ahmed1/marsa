# Setup Guide

This guide covers everything you need to set up MARSA for local development or deployment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Docker Setup](#docker-setup)
- [Environment Variables](#environment-variables)
- [Verifying the Installation](#verifying-the-installation)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
| ---------- | --------- | --------- |
| **Python** | 3.12+ | Backend runtime |
| **Node.js** | 20+ | Frontend runtime |
| **npm** | 10+ | Frontend package manager |
| **Git** | 2.30+ | Version control |

### Verifying Prerequisites

```bash
python --version   # Should be 3.12.x or higher
node --version     # Should be v20.x or higher
npm --version      # Should be 10.x or higher
git --version      # Should be 2.30.x or higher
```

### Required API Keys

You'll need API keys from three services:

| Service | Purpose | Sign Up | Free Tier |
| --------- | --------- | --------- | ----------- |
| **Anthropic** | Claude LLM for agents | [console.anthropic.com](https://console.anthropic.com/) | $5 credit |
| **OpenAI** | Embeddings for document store | [platform.openai.com](https://platform.openai.com/api-keys) | $5 credit |
| **Tavily** | Web search | [tavily.com](https://tavily.com/) | 1,000 searches/month |

### Optional Services

| Service | Purpose | Required For |
| ---------- | --------- | -------------- |
| **LangSmith** | Observability and tracing | Production debugging |
| **GitHub API** | GitHub MCP server | Technical research queries |

---

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/marsa.git
cd marsa
```

### 2. Create Environment File

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=tvly-your-key-here

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key

# Optional: Logging
LOG_LEVEL=INFO
```

### 3. Install Dependencies

The `make setup` command handles both backend and frontend:

```bash
make setup
```

This runs:

- Creates Python virtual environment at `.venv/`
- Installs Python dependencies from `backend/requirements.txt`
- Installs Node.js dependencies in `frontend/`

**Manual alternative:**

```bash
# Backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 4. Initialize Document Store (Optional)

To pre-populate the ChromaDB vector store with sample documents:

```bash
source .venv/bin/activate
python backend/scripts/ingest_docs.py
```

This ingests sample documents from `data/sample_docs/` for document-based research queries.

### 5. Start the Application

#### Option A: Both services with make

```bash
make dev
```

This starts both backend (port 8000) and frontend (port 3000) concurrently.

#### Option B: Separate terminals (recommended for debugging)

Terminal 1 - Backend:

```bash
source .venv/bin/activate
make run-backend
# or: cd backend && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 - Frontend:

```bash
make run-frontend
# or: cd frontend && npm run dev
```

### 6. Access the Application

Open your browser to [http://localhost:3000](http://localhost:3000)

---

## Docker Setup

Docker provides a containerized deployment that works consistently across environments.

### Docker Prerequisites

- Docker 24+ with Docker Compose v2

### Quick Start

```bash
# Build and start all services
make docker
# or: docker-compose up --build
```

This starts:

- **Backend** at `http://localhost:8000`
- **Frontend** at `http://localhost:3000`

### Environment Variables for Docker

Create a `.env` file in the project root (Docker Compose reads it automatically):

```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-key
OPENAI_API_KEY=sk-your-key
TAVILY_API_KEY=tvly-your-key
```

### Docker Commands

| Command | Description |
| --------- | ------------- |
| `docker-compose up` | Start services (attached) |
| `docker-compose up -d` | Start services (detached) |
| `docker-compose down` | Stop services |
| `docker-compose logs -f` | Follow logs |
| `docker-compose build` | Rebuild images |

### Volumes

The `docker-compose.yml` mounts `data/` for persistent storage:

- `data/chromadb/` - Vector database
- `data/eval_results/` - Evaluation outputs

---

## Environment Variables

### Required Variables

| Variable | Description | Example |
| ---------- | ------------- | --------- |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | `sk-ant-api03-...` |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | `sk-...` |
| `TAVILY_API_KEY` | Tavily API key for web search | `tvly-...` |

### Optional Variables

| Variable | Description | Default |
| ---------- | ------------- | --------- |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `false` |
| `LANGCHAIN_API_KEY` | LangSmith API key | - |
| `LANGCHAIN_PROJECT` | LangSmith project name | `marsa` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `data/chromadb` |

### Setting Environment Variables

**Option 1: `.env` file (recommended)**

```bash
cp .env.example .env
# Edit .env with your values
```

#### Option 2: Export in shell

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-your-key
export OPENAI_API_KEY=sk-your-key
export TAVILY_API_KEY=tvly-your-key
```

#### Option 3: Inline with command

```bash
ANTHROPIC_API_KEY=sk-... make run-backend
```

---

## Verifying the Installation

### 1. Check Backend Health

```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

### 2. Test MCP Servers

```bash
source .venv/bin/activate
python backend/manual_test_mcp.py
```

This tests connectivity to both Tavily and Document Store MCP servers.

### 3. Run Unit Tests

```bash
make test
```

All tests should pass. If any fail, check error messages for missing dependencies or configuration issues.

### 4. Test a Research Query

Open [http://localhost:3000](http://localhost:3000) and submit a simple query:

> "What is the Python GIL and why does it exist?"

You should see:

- Agent trace events streaming in real-time
- Source quality scores for each result
- Final report with citations

---

## Troubleshooting

### Common Issues

#### "ANTHROPIC_API_KEY not set"

**Cause**: Environment variable not loaded.

**Fix**:

```bash
# Verify the variable is set
echo $ANTHROPIC_API_KEY

# If empty, source the .env file or export manually
source .env  # If using a shell that supports sourcing .env
# or
export ANTHROPIC_API_KEY=your-key
```

#### "Connection refused" on port 8000

**Cause**: Backend not running or crashed.

**Fix**:

```bash
# Check if backend is running
curl http://localhost:8000/health

# If not, check logs
make run-backend  # Run in foreground to see errors
```

#### "Module not found" errors

**Cause**: Dependencies not installed or wrong Python environment.

**Fix**:

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r backend/requirements.txt
```

#### "Rate limit exceeded" from Tavily

**Cause**: Hit the 1,000 searches/month free tier limit.

**Fix**:

- Wait until next month for quota reset
- Use document store queries instead (`search_strategy: docs_only`)
- Upgrade to Tavily paid tier

#### ChromaDB "Collection not found"

**Cause**: Document store not initialized.

**Fix**:

```bash
python backend/scripts/ingest_docs.py
```

#### Frontend shows "Failed to connect"

**Cause**: Backend not running or CORS issue.

**Fix**:

1. Verify backend is running: `curl http://localhost:8000/health`
2. Check backend logs for errors
3. Ensure frontend is connecting to correct port (8000)

### Debugging Tips

1. **Enable verbose logging**:

   ```bash
   LOG_LEVEL=DEBUG make run-backend
   ```

2. **Check LangSmith trace** (if enabled):
   Visit [smith.langchain.com](https://smith.langchain.com) to see full agent traces.

3. **Test MCP servers individually**:

   ```bash
   # Test Tavily MCP server
   python -m fastmcp dev backend/mcp_servers/tavily_search.py

   # Test Document Store MCP server
   python -m fastmcp dev backend/mcp_servers/document_store.py
   ```

4. **Run specific tests**:

   ```bash
   pytest backend/tests/test_mcp_servers.py -v
   pytest backend/tests/test_workflow.py -v
   ```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/yourusername/marsa/issues)
2. Search existing issues for similar problems
3. Open a new issue with:
   - OS and Python/Node versions
   - Full error message
   - Steps to reproduce

---

## Next Steps

- Read the [Architecture](architecture.md) to understand system design
- Review [Design Decisions](design-decisions.md) for technical rationale
- Run `make eval` to see evaluation metrics
- Try the CLI: `python backend/run_cli.py "Your research question"`
