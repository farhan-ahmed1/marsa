"""Checkpointer configuration for LangGraph state persistence.

This module provides checkpointers for persisting agent state across workflow runs.
It supports:
- InMemorySaver for testing and simple use cases
- AsyncSqliteSaver for production persistence

Features enabled:
- Inspecting what each agent did
- Resuming interrupted workflows
- Human-in-the-loop checkpoints
- State history and debugging
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
    
    logger.info("creating_sqlite_checkpointer", db_path=db_path)
    
    # Returns an async context manager - use with `async with`
    return AsyncSqliteSaver.from_conn_string(db_path)


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
