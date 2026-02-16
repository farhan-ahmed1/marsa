"""Resilience utilities for MCP servers.

Provides retry logic, error handling, validation, and structured logging
for production-grade MCP server operations.
"""

import time
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Type variable for generic function decorators
F = TypeVar("F", bound=Callable[..., Any])

# Configure structlog for consistent JSON logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)


# ---------------------------------------------------------------------------
# Custom Exception Classes
# ---------------------------------------------------------------------------


class MCPServerError(Exception):
    """Base exception for MCP server errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(MCPServerError):
    """Raised when input validation fails."""

    pass


class SearchTimeoutError(MCPServerError):
    """Raised when a search operation times out."""

    pass


class RateLimitExceededError(MCPServerError):
    """Raised when API rate limits are exceeded."""

    pass


class EmbeddingError(MCPServerError):
    """Raised when embedding generation fails."""

    pass


class ExternalAPIError(MCPServerError):
    """Raised when an external API call fails."""

    pass


# ---------------------------------------------------------------------------
# Validation Constants
# ---------------------------------------------------------------------------

MAX_QUERY_LENGTH = 500
MIN_RESULTS = 1
MAX_RESULTS = 20
DEFAULT_SEARCH_TIMEOUT = 10  # seconds
DEFAULT_EMBEDDING_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------


def validate_query(query: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Validate and sanitize a search query.
    
    Args:
        query: The search query string
        max_length: Maximum allowed query length (default: 500)
        
    Returns:
        The validated and stripped query string
        
    Raises:
        ValidationError: If the query is empty, too long, or invalid
    """
    if not query:
        raise ValidationError(
            "Query cannot be empty",
            details={"field": "query", "value": query}
        )
    
    # Strip whitespace
    query = query.strip()
    
    if not query:
        raise ValidationError(
            "Query cannot be only whitespace",
            details={"field": "query", "value": ""}
        )
    
    if len(query) > max_length:
        raise ValidationError(
            f"Query exceeds maximum length of {max_length} characters",
            details={
                "field": "query",
                "length": len(query),
                "max_length": max_length
            }
        )
    
    return query


def validate_max_results(
    max_results: int,
    min_val: int = MIN_RESULTS,
    max_val: int = MAX_RESULTS
) -> int:
    """Validate the max_results parameter.
    
    Args:
        max_results: The requested number of results
        min_val: Minimum allowed value (default: 1)
        max_val: Maximum allowed value (default: 20)
        
    Returns:
        The validated max_results value (clamped to bounds)
    """
    if not isinstance(max_results, int):
        try:
            max_results = int(max_results)
        except (TypeError, ValueError) as e:
            raise ValidationError(
                "max_results must be an integer",
                details={"field": "max_results", "value": max_results}
            ) from e
    
    # Clamp to valid range
    return max(min_val, min(max_val, max_results))


# ---------------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------------


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the given name.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        A configured structlog logger instance
    """
    return structlog.get_logger(name)


def log_tool_invocation(
    logger: structlog.stdlib.BoundLogger,
    tool_name: str,
    query: str,
    **kwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory that logs tool invocations with timing.
    
    Args:
        logger: The structlog logger to use
        tool_name: Name of the tool being invoked
        query: The query parameter (for logging)
        **kwargs: Additional key-value pairs to log
        
    Returns:
        Decorator that wraps the function with logging
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **inner_kwargs: Any) -> Any:
            start_time = time.perf_counter()
            
            logger.info(
                "tool_invocation_started",
                tool=tool_name,
                query=query[:100] if query else None,  # Truncate for logging
                **kwargs
            )
            
            try:
                result = func(*args, **inner_kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                # Determine result count if applicable
                result_count = len(result) if hasattr(result, "__len__") else 1
                
                logger.info(
                    "tool_invocation_completed",
                    tool=tool_name,
                    latency_ms=round(elapsed_ms, 2),
                    result_count=result_count,
                    **kwargs
                )
                
                return result
                
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    "tool_invocation_failed",
                    tool=tool_name,
                    latency_ms=round(elapsed_ms, 2),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    **kwargs
                )
                raise
        
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Retry Decorators
# ---------------------------------------------------------------------------


def retry_with_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Create a retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        min_wait: Minimum wait time between retries in seconds (default: 1.0)
        max_wait: Maximum wait time between retries in seconds (default: 10.0)
        retry_on: Tuple of exception types to retry on
        
    Returns:
        Decorator that wraps the function with retry logic
        
    Example:
        @retry_with_backoff(max_attempts=3, retry_on=(TimeoutError, ConnectionError))
        async def fetch_data():
            ...
    """
    def decorator(func: F) -> F:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(retry_on),
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        return wrapper  # type: ignore[return-value]
    
    return decorator


def retry_search(func: F) -> F:
    """Decorator for retrying search operations with appropriate settings.
    
    Retries on network-related errors with exponential backoff.
    Max 3 attempts, 1-10 second wait between attempts.
    """
    return retry_with_backoff(
        max_attempts=3,
        min_wait=1.0,
        max_wait=10.0,
        retry_on=(TimeoutError, ConnectionError, OSError),
    )(func)


def retry_embedding(func: F) -> F:
    """Decorator for retrying embedding operations with appropriate settings.
    
    Retries on network-related errors with exponential backoff.
    Max 3 attempts, 2-15 second wait between attempts (longer for embeddings).
    """
    return retry_with_backoff(
        max_attempts=3,
        min_wait=2.0,
        max_wait=15.0,
        retry_on=(TimeoutError, ConnectionError, OSError),
    )(func)


# ---------------------------------------------------------------------------
# Timeout Utilities
# ---------------------------------------------------------------------------


class TimeoutContext:
    """Context manager for timing operations.
    
    Example:
        with TimeoutContext(timeout=10.0, name="search") as ctx:
            result = do_search()
        print(f"Took {ctx.elapsed_ms}ms")
    """
    
    def __init__(self, timeout: float, name: str = "operation"):
        self.timeout = timeout
        self.name = name
        self.start_time: float = 0
        self.elapsed_ms: float = 0
    
    def __enter__(self) -> "TimeoutContext":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        return False  # Don't suppress exceptions


def timed_operation(func: F) -> F:
    """Decorator that adds timing information to function calls.
    
    The function must accept **kwargs, and the timing info will be
    available via _timing_ms in the result if it's a dict.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # If result is a dict, add timing info
        if isinstance(result, dict):
            result["_timing_ms"] = round(elapsed_ms, 2)
        
        return result
    
    return wrapper  # type: ignore[return-value]
