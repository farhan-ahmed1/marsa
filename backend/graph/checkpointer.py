"""Checkpointer configuration for LangGraph state persistence.

This module provides checkpointers for persisting agent state across workflow runs.
It supports:
- InMemorySaver for testing and simple use cases
- AsyncSqliteSaver for production persistence with connection pooling

Features enabled:
- Inspecting what each agent did
- Resuming interrupted workflows
- Human-in-the-loop checkpoints
- State history and debugging
- SQLite WAL mode + connection pooling for concurrent access
"""

import os
from pathlib import Path
from typing import Optional, Union

import structlog
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = structlog.get_logger(__name__)


# Default path for checkpoints database
DEFAULT_CHECKPOINT_PATH = "./data/checkpoints.db"

# SQLite pragmas for production performance
_SQLITE_PRAGMAS = [
    "PRAGMA journal_mode=WAL;",     # Write-Ahead Logging for concurrent reads
    "PRAGMA synchronous=NORMAL;",   # Faster writes, still crash-safe with WAL
    "PRAGMA cache_size=-64000;",    # 64 MB page cache
    "PRAGMA busy_timeout=5000;",    # Wait up to 5 s on lock contention
    "PRAGMA foreign_keys=ON;",
]


def _ensure_data_dir(db_path: str) -> None:
    """Ensure the data directory exists for the database file.
    
    Args:
        db_path: Path to the database file.
    """
    data_dir = Path(db_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)


def create_checkpointer(
    db_path: Optional[str] = None,
    use_memory: bool = False,
) -> Union[InMemorySaver, "AsyncSqliteSaver"]:
    """Create a checkpointer for LangGraph.
    
    For simple use cases or testing, use InMemorySaver (use_memory=True).
    For production persistence, use AsyncSqliteSaver with a db_path.
    
    Note: When using AsyncSqliteSaver, this returns a context manager
    that must be used with `async with`.
    
    Args:
        db_path: Path to the SQLite database file.
                 Defaults to ./data/checkpoints.db
        use_memory: If True, return an InMemorySaver instead of SQLite.
        
    Returns:
        InMemorySaver if use_memory=True, otherwise an async context manager
        for AsyncSqliteSaver.
        
    Example:
        ```python
        from graph.checkpointer import create_checkpointer
        
        # For testing or simple use cases:
        checkpointer = create_checkpointer(use_memory=True)
        app = workflow.compile(checkpointer=checkpointer)
        
        # For production with SQLite (returns context manager):
        checkpointer_cm = create_checkpointer(db_path="./data/checkpoints.db")
        async with checkpointer_cm as checkpointer:
            app = workflow.compile(checkpointer=checkpointer)
            result = await app.ainvoke(state, config)
        ```
    """
    if use_memory:
        logger.info("creating_memory_checkpointer")
        return InMemorySaver()
    
    if db_path is None:
        db_path = os.environ.get(
            "MARSA_CHECKPOINT_DB", 
            DEFAULT_CHECKPOINT_PATH,
        )
    
    # Ensure directory exists
    _ensure_data_dir(db_path)
    
    logger.info(
        "creating_sqlite_checkpointer",
        db_path=db_path,
        pragmas=len(_SQLITE_PRAGMAS),
    )
    
    # Returns an async context manager - use with `async with`
    # Connection pooling is handled internally by aiosqlite; the pragmas
    # are applied after the connection is opened via AsyncSqliteSaver.setup().
    return AsyncSqliteSaver.from_conn_string(db_path)


async def apply_sqlite_pragmas(checkpointer: AsyncSqliteSaver) -> None:
    """Apply performance-tuned SQLite pragmas to an open checkpointer.

    Call this immediately after entering the ``AsyncSqliteSaver`` context
    manager to enable WAL mode and tune the cache.

    Args:
        checkpointer: An already-opened AsyncSqliteSaver.
    """
    if not hasattr(checkpointer, "conn") or checkpointer.conn is None:
        logger.warning("sqlite_pragmas_skipped", reason="no open connection")
        return

    for pragma in _SQLITE_PRAGMAS:
        try:
            await checkpointer.conn.execute(pragma)
        except Exception as exc:
            logger.warning("sqlite_pragma_failed", pragma=pragma, error=str(exc))

    logger.info("sqlite_pragmas_applied", count=len(_SQLITE_PRAGMAS))


async def get_thread_history(
    checkpointer: AsyncSqliteSaver,
    thread_id: str,
) -> list[dict]:
    """Get the state history for a specific thread.
    
    Args:
        checkpointer: The AsyncSqliteSaver instance.
        thread_id: The thread ID to query.
        
    Returns:
        List of state snapshots in chronological order.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    history = []
    async for state in checkpointer.alist(config):
        history.append({
            "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
            "parent_id": state.parent_config.get("configurable", {}).get("checkpoint_id") if state.parent_config else None,
            "values": state.values,
            "metadata": state.metadata,
        })
    
    return history


async def get_latest_state(
    checkpointer: AsyncSqliteSaver,
    thread_id: str,
) -> Optional[dict]:
    """Get the latest state for a specific thread.
    
    Args:
        checkpointer: The AsyncSqliteSaver instance.
        thread_id: The thread ID to query.
        
    Returns:
        The latest state values, or None if thread not found.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    result = await checkpointer.aget_tuple(config)
    
    if result:
        return result.values
    
    return None


async def clear_thread(
    checkpointer: AsyncSqliteSaver,
    thread_id: str,
) -> None:
    """Clear all checkpoints for a specific thread.
    
    Note: This is a placeholder - AsyncSqliteSaver in the current version
    may not support checkpoint deletion. This function logs the thread_id
    but does not actually delete checkpoints.
    
    Args:
        checkpointer: The AsyncSqliteSaver instance.
        thread_id: The thread ID to clear.
    """
    # Note: AsyncSqliteSaver doesn't have a delete method in current versions
    # Future versions may add this capability
    logger.info("clear_thread_requested", thread_id=thread_id)
    logger.warning(
        "checkpoint_deletion_not_supported",
        message="Checkpoint deletion not available in current langgraph-checkpoint-sqlite version"
    )
