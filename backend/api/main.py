"""FastAPI application entry point for the MARSA API.

This module configures and creates the FastAPI application with
CORS middleware, request-ID tracking, API-key auth, graceful shutdown,
and structured logging.
"""

import asyncio
import os
import signal
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from middleware.auth import APIKeyMiddleware
from middleware.request_id import RequestIDMiddleware
from utils.logging import configure_logging

from .routes import router

# ---------------------------------------------------------------------------
# Bootstrap structured logging before anything else
# ---------------------------------------------------------------------------
configure_logging(
    json_logs=os.getenv("ENVIRONMENT", "development") == "production",
    log_level=os.getenv("LOG_LEVEL", "INFO"),
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_langsmith() -> None:
    """Set LangSmith / LangChain tracing environment variables from config.

    LangChain checks os.environ at import time for these variables, so they
    must be set before any ChatAnthropic / chain objects are created.
    If LANGCHAIN_TRACING_V2 is already set in the environment (e.g. from a
    .env file loaded by dotenv) this is a no-op for that key.
    """
    try:
        from config import config  # local import to avoid circular at module level
        if config.langchain_tracing and config.langchain_api_key:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_API_KEY", config.langchain_api_key)
            os.environ.setdefault("LANGCHAIN_PROJECT", config.langchain_project or "marsa")
            logger.info(
                "langsmith_tracing_enabled",
                project=config.langchain_project,
            )
        else:
            logger.info("langsmith_tracing_disabled")
    except Exception as exc:
        logger.warning("langsmith_configure_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_event = asyncio.Event()


def _handle_shutdown(sig: signal.Signals) -> None:
    """Signal handler for graceful shutdown."""
    logger.info("shutdown_signal_received", signal=sig.name)
    _shutdown_event.set()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown.

    Initializes required resources on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("api_starting", version="1.0.0", environment=os.getenv("ENVIRONMENT", "development"))
    _configure_langsmith()

    # Register graceful shutdown signals
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_shutdown, sig)
        except NotImplementedError:
            # Windows does not support add_signal_handler
            pass

    logger.info("api_ready")

    yield

    # Shutdown
    logger.info("api_shutting_down")

    # Cancel in-flight workflow tasks
    from api.streaming import event_queue_manager

    active = event_queue_manager.get_active_streams()
    if active:
        logger.info("cancelling_active_streams", count=len(active))
        for stream_id in active:
            task = event_queue_manager.get_workflow_task(stream_id)
            if task and not task.done():
                task.cancel()
        # Give tasks a moment to finalize
        await asyncio.sleep(0.5)

    logger.info("api_shutdown_complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MARSA API",
    description="Multi-Agent Research Assistant API - Orchestrates AI agents for research queries",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Middleware (order matters: outermost first) ---

# 1. Request ID tracking (outermost so every response gets the header)
app.add_middleware(RequestIDMiddleware)

# 2. API key auth (after request ID so auth failures include the ID)
app.add_middleware(APIKeyMiddleware)

# 3. CORS (must be near the top to handle preflight correctly)
_allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
# Allow additional origins via env var (comma-separated)
_extra_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
if _extra_origins:
    _allowed_origins.extend(o.strip() for o in _extra_origins.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

# Include API routes
app.include_router(router, prefix="/api")
