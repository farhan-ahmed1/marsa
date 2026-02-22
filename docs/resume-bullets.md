# MARSA - Resume Bullet Points

Quantified metrics from evaluation runs (20 test queries across 5 categories).

---

## Top 3 Recommended Bullets

1. **Architected a multi-agent research system orchestrating 4 specialized AI agents via LangGraph, with MCP-connected data sources achieving 100% citation accuracy and 10.7-second median latency across 20 test queries**

2. **Built 3 custom MCP servers (web search, document store, GitHub) integrating Tavily, ChromaDB, and OpenAI embeddings for real-time information retrieval with source quality scoring (domain authority, recency, content depth)**

3. **Built a real-time observability dashboard tracking LLM calls, tool invocations, and agent state transitions with sub-second SSE streaming, backed by 493 unit tests at 77% code coverage**

---

## All Available Bullets

Pick the 3 strongest based on the role you are applying for.

### Architecture & Orchestration

- Architected a multi-agent research system orchestrating 4 specialized AI agents (Planner, Researcher, Fact-Checker, Synthesizer) via LangGraph, with MCP-connected data sources achieving 100% citation accuracy and 10.7-second median latency across 20 test queries
- Designed an adaptive query planning system that decomposes complex research questions into parallel sub-queries, with conditional fact-check loops and a 85.3% verification pass rate across diverse query categories

### MCP & Data Infrastructure

- Built 3 custom MCP servers (web search, document store, GitHub) integrating Tavily, ChromaDB, and OpenAI embeddings for real-time information retrieval with structured Pydantic tool interfaces
- Implemented a document ingestion pipeline with ChromaDB vector store and OpenAI text-embedding-3-small, supporting similarity search and runtime document storage through the Model Context Protocol

### Performance & Parallelism

- Implemented parallel agent execution using LangGraph's Send API, enabling concurrent research across multiple sub-queries with a median end-to-end latency of 10.7 seconds (p95: 14.5 seconds)

### Safety & Oversight

- Designed a human-in-the-loop checkpoint system with approval workflows, enabling safe agent deployment with user oversight and a fact-check loop that achieves 100% false premise recall

### Observability & Quality

- Built a real-time observability dashboard tracking LLM calls, tool invocations, and agent state transitions with sub-second SSE streaming via FastAPI
- Developed an automated evaluation framework measuring citation accuracy, source diversity, fact-check recall, and report quality across 5 query categories, achieving an average quality score of 3.68/5

### Testing & Engineering

- Maintained 77% backend test coverage with 493 unit tests, CI/CD via GitHub Actions (lint, type-check, test, build), and Docker Compose for single-command deployment

---

## Evaluation Metrics Reference

| Metric                  | Value     |
| ----------------------- | --------- |
| Total Test Queries      | 20        |
| Citation Accuracy       | 100%      |
| Fact-Check Pass Rate    | 85.3%     |
| False Premise Recall    | 100%      |
| Latency p50             | 10.7s     |
| Latency p95             | 14.5s     |
| Avg Quality Score       | 3.68/5    |
| Total Unit Tests        | 493       |
| Backend Code Coverage   | 77%       |
| MCP Servers             | 3         |
| AI Agents               | 4         |
| LLM Calls per Query     | ~4        |
| Avg Tokens per Query    | ~4,489    |

### Quality by Category

| Category      | Queries | Avg Quality | Avg Latency |
| ------------- | ------- | ----------- | ----------- |
| Factual       | 5       | 4.11/5      | 11.6s       |
| Comparison    | 5       | 4.01/5      | 9.7s        |
| Exploratory   | 5       | 3.30/5      | 10.3s       |
| Doc Context   | 2       | 3.58/5      | 10.1s       |
| False Premise | 3       | 3.13/5      | 10.3s       |
