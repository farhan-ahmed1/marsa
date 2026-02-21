"""Cross-session memory for MARSA.

Stores research session summaries in a ChromaDB collection (separate from the
knowledge base) using OpenAI embeddings so future queries can retrieve prior
findings via similarity search.

Lifecycle:
  1. After the Synthesizer completes, call ``store_session(state)`` to persist
     key topics, verified claims, and source quality data for the session.
  2. Before the Planner runs, call ``get_relevant_memories(query)`` to retrieve
     similar past sessions and inject them as context.

Storage schema (each ChromaDB document):
  - document (str): plain-text summary used for embedding
  - id (str): session_id (UUID)
  - metadata (dict): query, fact_check_pass_rate, iteration_count, timestamp,
                     topics_json, key_findings_json, source_quality_json
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
import structlog
from openai import OpenAI

from config import config
from graph.state import AgentState, VerificationVerdict

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHROMADB_PATH = DATA_DIR / "chromadb"
MEMORY_COLLECTION_NAME = "session_memories"
EMBEDDING_MODEL = "text-embedding-3-small"

# Maximum characters for a single memory summary document
MAX_SUMMARY_CHARS = 2000
# Number of past sessions to retrieve
DEFAULT_N_RESULTS = 3


# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_chroma_client: Optional[chromadb.PersistentClient] = None
_memory_collection: Optional[chromadb.Collection] = None
_openai_client: Optional[OpenAI] = None


def _get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        CHROMADB_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    return _chroma_client


def _get_memory_collection() -> chromadb.Collection:
    global _memory_collection
    if _memory_collection is None:
        client = _get_chroma_client()
        _memory_collection = client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _memory_collection


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.openai_api_key)
    return _openai_client


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


def _embed(text: str) -> list[float]:
    """Embed text using OpenAI text-embedding-3-small."""
    client = _get_openai_client()
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:8000],  # respect token limit
    )
    return resp.data[0].embedding


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_topics(state: AgentState) -> list[str]:
    """Extract research topics from the query plan."""
    topics: list[str] = []
    plan = state.get("plan")
    if plan:
        topics.extend(plan.sub_queries)
    # Always include the original query
    query = state.get("query", "")
    if query and query not in topics:
        topics.insert(0, query)
    return topics[:10]  # cap


def _extract_key_findings(state: AgentState) -> list[str]:
    """Extract verified (supported) claims as human-readable statements."""
    findings: list[str] = []
    for r in state.get("verification_results", []):
        try:
            if r.verdict == VerificationVerdict.SUPPORTED:
                findings.append(r.claim.statement)
        except AttributeError:
            pass
    return findings[:20]


def _extract_source_quality(state: AgentState) -> dict[str, float]:
    """Return a trimmed dict of source domain -> quality score."""
    raw: dict[str, float] = state.get("source_scores", {})
    # Keep only the top-10 sources to avoid bloating metadata
    sorted_items = sorted(raw.items(), key=lambda x: x[1], reverse=True)[:10]
    return {url[:120]: score for url, score in sorted_items}


def _compute_fact_check_pass_rate(state: AgentState) -> float:
    """Compute fraction of claims that passed fact-checking."""
    results = state.get("verification_results", [])
    if not results:
        return 0.0
    supported = sum(
        1 for r in results
        if getattr(r, "verdict", None) == VerificationVerdict.SUPPORTED
    )
    return supported / len(results)


def _build_summary(
    query: str,
    topics: list[str],
    key_findings: list[str],
    fact_check_pass_rate: float,
) -> str:
    """Build the plain-text summary stored as the ChromaDB document."""
    parts = [
        f"Research query: {query}",
        f"Topics: {'; '.join(topics)}",
    ]
    if key_findings:
        parts.append(f"Key verified findings: {'; '.join(key_findings[:8])}")
    parts.append(f"Fact-check pass rate: {fact_check_pass_rate:.0%}")
    summary = "\n".join(parts)
    return summary[:MAX_SUMMARY_CHARS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def store_session(state: AgentState) -> None:
    """Persist the completed research session to the memory store.

    This should be called after the Synthesizer finishes. It is synchronous
    (ChromaDB + OpenAI embeddings are synchronous operations) and intentionally
    fire-and-forget — errors are logged but not re-raised.

    Args:
        state: The final AgentState after synthesis completes.
    """
    import uuid

    session_id = str(uuid.uuid4())
    query = state.get("query", "")
    if not query:
        logger.warning("store_session_skipped", reason="empty query")
        return

    try:
        t0 = time.perf_counter()

        topics = _extract_topics(state)
        key_findings = _extract_key_findings(state)
        source_quality = _extract_source_quality(state)
        fact_check_pass_rate = _compute_fact_check_pass_rate(state)
        iteration_count = state.get("iteration_count", 0)

        summary = _build_summary(query, topics, key_findings, fact_check_pass_rate)
        embedding = _embed(summary)

        collection = _get_memory_collection()
        collection.add(
            ids=[session_id],
            documents=[summary],
            embeddings=[embedding],
            metadatas=[
                {
                    "query": query[:500],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fact_check_pass_rate": fact_check_pass_rate,
                    "iteration_count": iteration_count,
                    "topics_json": json.dumps(topics[:10]),
                    "key_findings_json": json.dumps(key_findings[:10]),
                    "source_quality_json": json.dumps(source_quality),
                }
            ],
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "session_stored",
            session_id=session_id,
            query=query[:50],
            topics=len(topics),
            findings=len(key_findings),
            elapsed_ms=round(elapsed),
        )

    except Exception as exc:
        logger.error("store_session_failed", error=str(exc), query=query[:50])


def get_relevant_memories(query: str, n_results: int = DEFAULT_N_RESULTS) -> str:
    """Retrieve past sessions related to the query.

    Returns a formatted string suitable for injection into the Planner's
    system prompt. Returns an empty string if no relevant memories exist or
    on any error.

    Args:
        query: The incoming research query.
        n_results: Number of past sessions to retrieve.

    Returns:
        A multi-line string summarising prior relevant research, or "".
    """
    try:
        collection = _get_memory_collection()
        if collection.count() == 0:
            return ""

        embedding = _embed(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents: list[str] = results.get("documents", [[]])[0]
        metadatas: list[dict] = results.get("metadatas", [[]])[0]
        distances: list[float] = results.get("distances", [[]])[0]

        if not documents:
            return ""

        # Filter out low-similarity results (cosine distance > 0.6 means poor match)
        DISTANCE_THRESHOLD = 0.6
        relevant = [
            (doc, meta, dist)
            for doc, meta, dist in zip(documents, metadatas, distances)
            if dist <= DISTANCE_THRESHOLD
        ]

        if not relevant:
            return ""

        lines = [
            "## Prior Research Context",
            "The following findings come from earlier research sessions on related topics.",
            "Use them as supplementary context, but still perform thorough current research.",
            "",
        ]
        for i, (doc, meta, dist) in enumerate(relevant, 1):
            topics = json.loads(meta.get("topics_json", "[]"))
            findings = json.loads(meta.get("key_findings_json", "[]"))
            ts = meta.get("timestamp", "")[:10]  # YYYY-MM-DD
            pass_rate = float(meta.get("fact_check_pass_rate", 0))

            lines.append(f"### Session {i} (similarity: {1 - dist:.0%}, date: {ts})")
            lines.append(f"Query: {meta.get('query', 'unknown')}")
            if topics:
                lines.append(f"Topics covered: {', '.join(topics[:5])}")
            if findings:
                lines.append(f"Verified findings ({pass_rate:.0%} pass rate):")
                for f in findings[:5]:
                    lines.append(f"  - {f}")
            lines.append("")

        context = "\n".join(lines)
        logger.info(
            "memories_retrieved",
            query=query[:50],
            n_retrieved=len(relevant),
        )
        return context

    except Exception as exc:
        logger.error("get_memories_failed", error=str(exc), query=query[:50])
        return ""


def get_memory_count() -> int:
    """Return the total number of stored memory entries."""
    try:
        return _get_memory_collection().count()
    except Exception:
        return 0
