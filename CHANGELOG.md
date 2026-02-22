# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-02-22

Initial release of MARSA (Multi-Agent ReSearch Assistant).

### Added

#### Multi-Agent Orchestration
- Four specialized AI agents: Planner, Researcher, Fact-Checker, Synthesizer
- LangGraph state machine with conditional routing and fact-check loops
- Parallel sub-query execution using LangGraph's Send API
- Human-in-the-loop checkpoint with approval workflows after fact-checking
- SQLite-based state persistence via LangGraph checkpointer

#### MCP Servers
- Tavily web search MCP server with rate limiting and retry logic
- ChromaDB document store MCP server with OpenAI embeddings (text-embedding-3-small)
- GitHub MCP server (optional) for repository data access
- Unified MCP client wrapper for agent-to-server communication

#### Backend (FastAPI)
- Async REST API with SSE streaming for real-time agent trace events
- Cross-session memory using ChromaDB for context-aware follow-up queries
- Source quality scoring (domain authority, recency, content depth)
- Structured logging with structlog and request ID tracking
- Input sanitization and API key authentication middleware
- Response caching for repeated queries
- Rate limiting for external API calls (Tavily free tier tracking)
- Resilience utilities with exponential backoff retry logic

#### Frontend (Next.js)
- Real-time agent trace visualization with color-coded timeline
- Report view with inline citations and source quality badges
- Human-in-the-loop feedback interface (approve, dig deeper, correct)
- Observability timeline (Gantt-style) for agent execution breakdown
- Query history sidebar with local storage persistence
- Dark mode support with toggle
- Responsive layout (mobile and desktop)
- shadcn/ui component library (button, card, input, badge, tabs)

#### Evaluation & Testing
- Automated evaluation framework with 20 test queries across 5 categories
- Metrics: citation accuracy, source diversity, fact-check recall, report quality
- 493 backend unit tests (77% code coverage)
- 74 frontend tests (Vitest + Testing Library)
- Integration test suite for real API validation

#### DevOps & Infrastructure
- Docker Compose setup for single-command deployment
- GitHub Actions CI/CD pipeline (lint, type-check, test, build, coverage)
- Makefile with standardized development commands
- LangSmith integration for trace observability

#### Documentation
- Architecture diagrams (Mermaid) covering system overview and LangGraph workflow
- Design decisions document with technical rationale
- Setup guide with prerequisites and troubleshooting
- Evaluation results with per-category breakdowns

### Evaluation Results (v1.0.0)

| Metric                | Value   |
| --------------------- | ------- |
| Citation Accuracy     | 100%    |
| Fact-Check Pass Rate  | 85.3%   |
| False Premise Recall  | 100%    |
| Latency p50           | 10.7s   |
| Latency p95           | 14.5s   |
| Avg Quality Score     | 3.68/5  |
