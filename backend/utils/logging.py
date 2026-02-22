"""Centralized structured logging configuration for MARSA.

Provides a single point of configuration for structlog across the entire
backend. All modules should import ``configure_logging`` and call it once
at application startup, or use ``get_logger`` for per-module loggers.

Features:
- JSON-formatted log output for production
- Console-friendly colored output for development
- Request ID context binding
- Consistent timestamp format (ISO-8601)
"""

import logging
import os
import sys
from typing import Any

import structlog


def configure_logging(
    *,
    json_logs: bool | None = None,
    log_level: str = "INFO",
) -> None:
    """Configure structlog for the entire application.

    Call this once during FastAPI lifespan startup. Subsequent calls are
    idempotent.

    Args:
        json_logs: Force JSON output. Defaults to ``True`` when
            ``ENVIRONMENT`` is ``"production"``.
        log_level: Root log level (DEBUG, INFO, WARNING, ERROR).
    """
    if json_logs is None:
        json_logs = os.getenv("ENVIRONMENT", "development") == "production"

    log_level_value = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Mirror level to stdlib root so structlog's ``filter_by_level`` works
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level_value,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for *name*.

    Args:
        name: Typically ``__name__`` of the calling module.
    """
    return structlog.get_logger(name)
