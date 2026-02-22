"""Unit tests for the resilience module.

Tests cover validation functions, custom exceptions, retry decorators,
TimeoutContext, timed_operation, and structured logging utilities.
"""

import sys
import time
from pathlib import Path

import pytest

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.resilience import (  # noqa: E402
    EmbeddingError,
    ExternalAPIError,
    MCPServerError,
    RateLimitExceededError,
    SearchTimeoutError,
    TimeoutContext,
    ValidationError,
    get_logger,
    log_tool_invocation,
    retry_embedding,
    retry_search,
    retry_with_backoff,
    timed_operation,
    validate_max_results,
    validate_query,
)


# ---------------------------------------------------------------------------
# Custom Exception Tests
# ---------------------------------------------------------------------------


class TestCustomExceptions:
    """Tests for custom exception hierarchy."""

    def test_mcp_server_error_attributes(self):
        err = MCPServerError("test error", details={"key": "val"})
        assert str(err) == "test error"
        assert err.message == "test error"
        assert err.details == {"key": "val"}

    def test_mcp_server_error_default_details(self):
        err = MCPServerError("no details")
        assert err.details == {}

    def test_validation_error_inherits(self):
        err = ValidationError("bad input")
        assert isinstance(err, MCPServerError)

    def test_search_timeout_error(self):
        err = SearchTimeoutError("timeout", details={"timeout_s": 10})
        assert err.details["timeout_s"] == 10

    def test_rate_limit_exceeded_error(self):
        err = RateLimitExceededError("rate limited")
        assert isinstance(err, MCPServerError)

    def test_embedding_error(self):
        err = EmbeddingError("embedding failed")
        assert isinstance(err, MCPServerError)

    def test_external_api_error(self):
        err = ExternalAPIError("api down")
        assert isinstance(err, MCPServerError)


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------


class TestValidateQuery:
    """Tests for validate_query function."""

    def test_valid_query(self):
        assert validate_query("What is Python?") == "What is Python?"

    def test_strips_whitespace(self):
        assert validate_query("  hello  ") == "hello"

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_query("")

    def test_none_query_raises(self):
        with pytest.raises(ValidationError):
            validate_query(None)

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError, match="whitespace"):
            validate_query("   ")

    def test_exceeds_max_length(self):
        long_query = "a" * 600
        with pytest.raises(ValidationError, match="maximum length"):
            validate_query(long_query)

    def test_custom_max_length(self):
        with pytest.raises(ValidationError):
            validate_query("a" * 20, max_length=10)


class TestValidateMaxResults:
    """Tests for validate_max_results function."""

    def test_valid_value(self):
        assert validate_max_results(5) == 5

    def test_clamps_to_min(self):
        assert validate_max_results(0) == 1

    def test_clamps_to_max(self):
        assert validate_max_results(100) == 20

    def test_string_conversion(self):
        assert validate_max_results("5") == 5

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError, match="integer"):
            validate_max_results("abc")

    def test_custom_bounds(self):
        assert validate_max_results(50, min_val=10, max_val=30) == 30
        assert validate_max_results(5, min_val=10, max_val=30) == 10


# ---------------------------------------------------------------------------
# Logging Tests
# ---------------------------------------------------------------------------


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert logger is not None


class TestLogToolInvocation:
    """Tests for log_tool_invocation decorator factory."""

    def test_logs_and_returns_result(self):
        logger = get_logger("test")

        @log_tool_invocation(logger, "search", "test query")
        def my_func():
            return [1, 2, 3]

        result = my_func()
        assert result == [1, 2, 3]

    def test_logs_exception(self):
        logger = get_logger("test")

        @log_tool_invocation(logger, "search", "fail query")
        def my_func():
            raise ValueError("bad")

        with pytest.raises(ValueError, match="bad"):
            my_func()


# ---------------------------------------------------------------------------
# Retry Decorator Tests
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_succeeds_without_retry(self):
        call_count = 0

        @retry_with_backoff(max_attempts=3, min_wait=0.01, max_wait=0.02)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_retries_on_failure(self):
        call_count = 0

        @retry_with_backoff(
            max_attempts=3,
            min_wait=0.01,
            max_wait=0.02,
            retry_on=(ValueError,),
        )
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("retry me")
            return "ok"

        assert fail_twice() == "ok"
        assert call_count == 3

    def test_max_attempts_exceeded(self):
        @retry_with_backoff(
            max_attempts=2,
            min_wait=0.01,
            max_wait=0.02,
            retry_on=(ValueError,),
        )
        def always_fail():
            raise ValueError("always")

        with pytest.raises(ValueError, match="always"):
            always_fail()


class TestRetrySearch:
    """Tests for retry_search decorator."""

    def test_retry_search_wraps_function(self):
        call_count = 0

        @retry_search
        def my_search():
            nonlocal call_count
            call_count += 1
            return ["result"]

        assert my_search() == ["result"]
        assert call_count == 1


class TestRetryEmbedding:
    """Tests for retry_embedding decorator."""

    def test_retry_embedding_wraps_function(self):
        @retry_embedding
        def embed_text():
            return [0.1, 0.2]

        assert embed_text() == [0.1, 0.2]


# ---------------------------------------------------------------------------
# TimeoutContext Tests
# ---------------------------------------------------------------------------


class TestTimeoutContext:
    """Tests for TimeoutContext context manager."""

    def test_measures_elapsed_time(self):
        with TimeoutContext(timeout=5.0, name="test") as ctx:
            time.sleep(0.01)

        assert ctx.elapsed_ms > 0
        assert ctx.name == "test"

    def test_does_not_suppress_exceptions(self):
        with pytest.raises(ValueError):
            with TimeoutContext(timeout=5.0, name="test"):
                raise ValueError("boom")


# ---------------------------------------------------------------------------
# timed_operation Tests
# ---------------------------------------------------------------------------


class TestTimedOperation:
    """Tests for timed_operation decorator."""

    def test_adds_timing_to_dict_result(self):
        @timed_operation
        def my_op():
            return {"data": "ok"}

        result = my_op()
        assert "_timing_ms" in result
        assert result["_timing_ms"] >= 0
        assert result["data"] == "ok"

    def test_non_dict_result_unchanged(self):
        @timed_operation
        def my_op():
            return [1, 2, 3]

        result = my_op()
        assert result == [1, 2, 3]
