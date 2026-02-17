"""FastAPI application entry point for the MARSA API.

This module configures and creates the FastAPI application with
CORS middleware, routes, and startup/shutdown handlers.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown.
    
    Initializes required resources on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("api_starting", version="1.0.0")
    
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
