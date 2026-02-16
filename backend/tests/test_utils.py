"""Tests for the resilience utilities.

Tests validation functions, error classes, retry logic, and rate limiting.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.resilience import ( # noqa: E402
    DEFAULT_EMBEDDING_TIMEOUT,
    DEFAULT_SEARCH_TIMEOUT,
    EmbeddingError,
    ExternalAPIError,
    MAX_QUERY_LENGTH,
    MAX_RESULTS,
    MCPServerError,
    MIN_RESULTS,
    RateLimitExceededError,
    SearchTimeoutError,
    ValidationError,
    get_logger,
    retry_with_backoff,
    validate_max_results,
    validate_query,
)
from utils.rate_limiter import ( # noqa: E402
    DEFAULT_MONTHLY_LIMIT,
    RateLimiter,
    get_rate_limiter,
)


class TestValidateQuery:
    """Tests for the validate_query function."""
    
    def test_valid_query(self):
        """Test that valid queries pass validation."""
        result = validate_query("What is Python?")
        assert result == "What is Python?"
    
    def test_query_strips_whitespace(self):
        """Test that queries are stripped of leading/trailing whitespace."""
        result = validate_query("  What is Python?  ")
        assert result == "What is Python?"
    
    def test_empty_query_raises_error(self):
        """Test that empty query raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_query("")
        assert "cannot be empty" in str(exc_info.value)
    
    def test_whitespace_only_query_raises_error(self):
        """Test that whitespace-only query raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_query("   \t\n  ")
        assert "cannot be only whitespace" in str(exc_info.value)
    
    def test_query_exceeds_max_length(self):
        """Test that query exceeding max length raises ValidationError."""
        long_query = "a" * (MAX_QUERY_LENGTH + 1)
        with pytest.raises(ValidationError) as exc_info:
            validate_query(long_query)
        assert "exceeds maximum length" in str(exc_info.value)
        assert exc_info.value.details["max_length"] == MAX_QUERY_LENGTH
    
    def test_query_at_max_length(self):
        """Test that query at exactly max length passes."""
        exact_query = "a" * MAX_QUERY_LENGTH
        result = validate_query(exact_query)
        assert len(result) == MAX_QUERY_LENGTH
    
    def test_custom_max_length(self):
        """Test custom max length parameter."""
        result = validate_query("short", max_length=10)
        assert result == "short"
        
        with pytest.raises(ValidationError):
            validate_query("this is too long", max_length=10)


class TestValidateMaxResults:
    """Tests for the validate_max_results function."""
    
    def test_valid_max_results(self):
        """Test that valid max_results values pass."""
        assert validate_max_results(5) == 5
        assert validate_max_results(1) == 1
        assert validate_max_results(20) == 20
    
    def test_max_results_clamped_to_minimum(self):
        """Test that values below minimum are clamped."""
        assert validate_max_results(0) == MIN_RESULTS
        assert validate_max_results(-5) == MIN_RESULTS
    
    def test_max_results_clamped_to_maximum(self):
        """Test that values above maximum are clamped."""
        assert validate_max_results(100) == MAX_RESULTS
        assert validate_max_results(50) == MAX_RESULTS
    
    def test_max_results_string_conversion(self):
        """Test that string values are converted to int."""
        assert validate_max_results("10") == 10
    
    def test_max_results_invalid_string_raises_error(self):
        """Test that invalid string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_max_results("not a number")
        assert "must be an integer" in str(exc_info.value)
    
    def test_custom_bounds(self):
        """Test custom min/max bounds."""
        result = validate_max_results(5, min_val=2, max_val=10)
        assert result == 5
        
        result = validate_max_results(1, min_val=2, max_val=10)
        assert result == 2
        
        result = validate_max_results(15, min_val=2, max_val=10)
        assert result == 10


class TestCustomExceptions:
    """Tests for custom exception classes."""
    
    def test_mcp_server_error(self):
        """Test MCPServerError base class."""
        error = MCPServerError("Test error", details={"key": "value"})
        assert error.message == "Test error"
        assert error.details == {"key": "value"}
        assert str(error) == "Test error"
    
    def test_mcp_server_error_no_details(self):
        """Test MCPServerError without details."""
        error = MCPServerError("Test error")
        assert error.details == {}
    
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError(
            "Invalid input",
            details={"field": "query", "value": ""}
        )
        assert isinstance(error, MCPServerError)
        assert error.details["field"] == "query"
    
    def test_search_timeout_error(self):
        """Test SearchTimeoutError."""
        error = SearchTimeoutError(
            "Search timed out",
            details={"timeout": 10, "query": "test"}
        )
        assert isinstance(error, MCPServerError)
    
    def test_rate_limit_exceeded_error(self):
        """Test RateLimitExceededError."""
        error = RateLimitExceededError(
            "Rate limit exceeded",
            details={"current": 1000, "limit": 1000}
        )
        assert isinstance(error, MCPServerError)
    
    def test_embedding_error(self):
        """Test EmbeddingError."""
        error = EmbeddingError(
            "Embedding failed",
            details={"text_length": 5000}
        )
        assert isinstance(error, MCPServerError)
    
    def test_external_api_error(self):
        """Test ExternalAPIError."""
        error = ExternalAPIError(
            "API call failed",
            details={"api": "tavily", "status": 500}
        )
        assert isinstance(error, MCPServerError)


class TestGetLogger:
    """Tests for the get_logger function."""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_module")
        assert logger is not None
    
    def test_logger_can_log(self):
        """Test that the logger can log messages."""
        logger = get_logger("test_module")
        # Should not raise
        logger.info("Test message", key="value")


class TestRetryWithBackoff:
    """Tests for the retry_with_backoff decorator."""
    
    def test_successful_function_no_retry(self):
        """Test that successful functions don't retry."""
        call_count = 0
        
        @retry_with_backoff(max_attempts=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_exception(self):
        """Test that function retries on specified exception."""
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, min_wait=0.01, max_wait=0.1, retry_on=(ValueError,))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert call_count == 3
    
    def test_max_retries_exceeded(self):
        """Test that function fails after max retries."""
        @retry_with_backoff(max_attempts=2, min_wait=0.01, max_wait=0.1, retry_on=(ValueError,))
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
    
    def test_no_retry_on_different_exception(self):
        """Test that function doesn't retry on non-specified exception."""
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, retry_on=(ValueError,))
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Different error")
        
        with pytest.raises(TypeError):
            raises_type_error()
        
        assert call_count == 1  # No retries


class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary directory for rate limit storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_create_rate_limiter(self, temp_storage_dir):
        """Test creating a RateLimiter."""
        limiter = RateLimiter(
            "test",
            monthly_limit=100,
            storage_dir=temp_storage_dir
        )
        
        assert limiter.name == "test"
        assert limiter.monthly_limit == 100
        assert limiter.current_count == 0
        assert limiter.remaining == 100
    
    def test_increment_counter(self, temp_storage_dir):
        """Test incrementing the usage counter."""
        limiter = RateLimiter(
            "test",
            monthly_limit=100,
            storage_dir=temp_storage_dir
        )
        
        limiter.increment()
        assert limiter.current_count == 1
        assert limiter.remaining == 99
        
        limiter.increment(5)
        assert limiter.current_count == 6
        assert limiter.remaining == 94
    
    def test_check_limit_passes(self, temp_storage_dir):
        """Test check_limit passes when under limit."""
        limiter = RateLimiter(
            "test",
            monthly_limit=100,
            storage_dir=temp_storage_dir
        )
        
        result = limiter.check_limit()
        assert result["remaining"] == 100
        assert result["is_warning"] is False
    
    def test_check_limit_raises_when_exceeded(self, temp_storage_dir):
        """Test check_limit raises when limit exceeded."""
        limiter = RateLimiter(
            "test",
            monthly_limit=10,
            storage_dir=temp_storage_dir
        )
        
        # Use up all requests
        limiter.increment(10)
        
        with pytest.raises(RateLimitExceededError) as exc_info:
            limiter.check_limit()
        
        assert "exceeded" in str(exc_info.value).lower()
    
    def test_warning_threshold(self, temp_storage_dir):
        """Test warning threshold detection."""
        limiter = RateLimiter(
            "test",
            monthly_limit=100,
            storage_dir=temp_storage_dir
        )
        
        # 89% usage - not warning yet
        limiter.increment(89)
        assert limiter.is_warning is False
        
        # 90% usage - warning threshold
        limiter.increment(1)
        assert limiter.is_warning is True
    
    def test_persistence(self, temp_storage_dir):
        """Test that usage persists across instances."""
        # First instance
        limiter1 = RateLimiter(
            "test",
            monthly_limit=100,
            storage_dir=temp_storage_dir
        )
        limiter1.increment(50)
        
        # Second instance should load persisted data
        limiter2 = RateLimiter(
            "test",
            monthly_limit=100,
            storage_dir=temp_storage_dir
        )
        assert limiter2.current_count == 50
    
    def test_reset(self, temp_storage_dir):
        """Test resetting the counter."""
        limiter = RateLimiter(
            "test",
            monthly_limit=100,
            storage_dir=temp_storage_dir
        )
        
        limiter.increment(50)
        assert limiter.current_count == 50
        
        limiter.reset()
        assert limiter.current_count == 0
    
    def test_get_status(self, temp_storage_dir):
        """Test getting full status."""
        limiter = RateLimiter(
            "test",
            monthly_limit=100,
            storage_dir=temp_storage_dir
        )
        limiter.increment(25)
        
        status = limiter.get_status()
        
        assert status["name"] == "test"
        assert status["current_count"] == 25
        assert status["monthly_limit"] == 100
        assert status["remaining"] == 75
        assert status["usage_percent"] == 25.0
        assert "period" in status


class TestGetRateLimiter:
    """Tests for the get_rate_limiter singleton function."""
    
    def test_get_rate_limiter_returns_same_instance(self):
        """Test that get_rate_limiter returns singleton."""
        limiter1 = get_rate_limiter("singleton_test")
        limiter2 = get_rate_limiter("singleton_test")
        
        assert limiter1 is limiter2
    
    def test_different_names_different_instances(self):
        """Test that different names get different instances."""
        limiter1 = get_rate_limiter("limiter_a")
        limiter2 = get_rate_limiter("limiter_b")
        
        assert limiter1 is not limiter2


class TestConstants:
    """Tests for module constants."""
    
    def test_default_timeouts(self):
        """Test default timeout values."""
        assert DEFAULT_SEARCH_TIMEOUT == 10
        assert DEFAULT_EMBEDDING_TIMEOUT == 30
    
    def test_validation_constants(self):
        """Test validation constants."""
        assert MAX_QUERY_LENGTH == 500
        assert MIN_RESULTS == 1
        assert MAX_RESULTS == 20
    
    def test_rate_limit_constant(self):
        """Test rate limit constant."""
        assert DEFAULT_MONTHLY_LIMIT == 1000
