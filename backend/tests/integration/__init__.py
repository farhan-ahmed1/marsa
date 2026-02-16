"""Integration tests that use real external APIs.

These tests make actual API calls and consume quota/credits:
- Tavily API (web search)
- OpenAI API (embeddings)
- Anthropic API (LLM calls)

Run with: make test:integration
Or: pytest backend/tests/integration -v -s
"""
