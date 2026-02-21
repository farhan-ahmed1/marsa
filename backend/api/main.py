"""FastAPI application entry point for the MARSA API.

This module configures and creates the FastAPI application with
CORS middleware, routes, and startup/shutdown handlers.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

logger = structlog.get_logger(__name__)


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown.
    
    Initializes required resources on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("api_starting", version="1.0.0")
    _configure_langsmith()
    
    yield
    
    # Shutdown
    logger.info("api_shutting_down")


app = FastAPI(
    title="MARSA API",
    description="Multi-Agent Research Assistant API - Orchestrates AI agents for research queries",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")
