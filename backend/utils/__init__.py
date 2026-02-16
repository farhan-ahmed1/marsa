"""Utility modules for the MARSA backend."""

from utils.resilience import (
    EmbeddingError,
    ExternalAPIError,
    MCPServerError,
    RateLimitExceededError,
    SearchTimeoutError,
    ValidationError,
    retry_with_backoff,
    retry_search,
    retry_embedding,
    validate_query,
    validate_max_results,
    get_logger,
    DEFAULT_SEARCH_TIMEOUT,
    DEFAULT_EMBEDDING_TIMEOUT,
)
from utils.rate_limiter import (
    RateLimiter,
    get_rate_limiter,
    DEFAULT_MONTHLY_LIMIT,
)

__all__ = [
    # Exceptions
    "EmbeddingError",
    "ExternalAPIError",
    "MCPServerError",
    "RateLimitExceededError", 
    "SearchTimeoutError",
    "ValidationError",
    # Retry decorators
    "retry_with_backoff",
    "retry_search",
    "retry_embedding",
    # Validation
    "validate_query",
    "validate_max_results",
    # Logging
    "get_logger",
    # Rate limiting
    "RateLimiter",
    "get_rate_limiter",
    # Constants
    "DEFAULT_SEARCH_TIMEOUT",
    "DEFAULT_EMBEDDING_TIMEOUT",
    "DEFAULT_MONTHLY_LIMIT",
]
