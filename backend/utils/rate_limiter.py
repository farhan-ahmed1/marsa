"""Rate limiting utilities for MCP servers.

Provides a simple rate limiter that tracks API usage with monthly resets,
persisted to a JSON file.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.resilience import RateLimitExceededError, get_logger

logger = get_logger(__name__)

# Default rate limit for Tavily free tier
DEFAULT_MONTHLY_LIMIT = 1000
WARNING_THRESHOLD = 0.9  # Warn when 90% of limit is reached


class RateLimiter:
    """Simple rate limiter with monthly reset and JSON persistence.
    
    Tracks API usage against a monthly limit, persisting the counter
    to a JSON file to survive restarts.
    
    Attributes:
        name: Identifier for this rate limiter (e.g., "tavily")
        monthly_limit: Maximum requests allowed per month
        storage_path: Path to the JSON file storing usage data
    
    Example:
        limiter = RateLimiter("tavily", monthly_limit=1000)
        
        # Before each API call:
        limiter.check_limit()  # Raises if limit exceeded
        
        # After successful API call:
        limiter.increment()
    """
    
    def __init__(
        self,
        name: str,
        monthly_limit: int = DEFAULT_MONTHLY_LIMIT,
        storage_dir: Optional[Path] = None
    ):
        """Initialize the rate limiter.
        
        Args:
            name: Identifier for this rate limiter
            monthly_limit: Maximum requests allowed per month
            storage_dir: Directory to store usage data (default: data/)
        """
        self.name = name
        self.monthly_limit = monthly_limit
        
        # Set up storage path
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent.parent / "data"
        
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = storage_dir / f"{name}_rate_limit.json"
        
        # Load or initialize usage data
        self._usage_data = self._load_usage_data()
    
    def _load_usage_data(self) -> dict:
        """Load usage data from JSON file or initialize new data.
        
        Returns:
            Dictionary with count and period (YYYY-MM format)
        """
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    
                # Check if we need to reset for new month
                current_period = self._get_current_period()
                if data.get("period") != current_period:
                    logger.info(
                        "rate_limit_reset",
                        name=self.name,
                        old_period=data.get("period"),
                        new_period=current_period,
                        final_count=data.get("count", 0)
                    )
                    return self._new_usage_data()
                    
                return data
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "rate_limit_data_corrupted",
                    name=self.name,
                    error=str(e)
                )
                return self._new_usage_data()
        
        return self._new_usage_data()
    
    def _new_usage_data(self) -> dict:
        """Create new usage data for current period.
        
        Returns:
            Dictionary with count=0 and current period
        """
        return {
            "count": 0,
            "period": self._get_current_period(),
            "limit": self.monthly_limit
        }
    
    def _get_current_period(self) -> str:
        """Get current period identifier (YYYY-MM format).
        
        Returns:
            String in YYYY-MM format
        """
        return datetime.now(timezone.utc).strftime("%Y-%m")
    
    def _save_usage_data(self) -> None:
        """Persist usage data to JSON file."""
        # Ensure we have the current period
        current_period = self._get_current_period()
        if self._usage_data.get("period") != current_period:
            self._usage_data = self._new_usage_data()
        
        with open(self.storage_path, "w") as f:
            json.dump(self._usage_data, f, indent=2)
    
    @property
    def current_count(self) -> int:
        """Get current usage count for this period.
        
        Returns:
            Number of requests made this month
        """
        # Check for period rollover
        if self._usage_data.get("period") != self._get_current_period():
            self._usage_data = self._new_usage_data()
        
        return self._usage_data.get("count", 0)
    
    @property
    def remaining(self) -> int:
        """Get remaining requests for this period.
        
        Returns:
            Number of requests remaining this month
        """
        return max(0, self.monthly_limit - self.current_count)
    
    @property
    def is_warning(self) -> bool:
        """Check if usage has reached warning threshold.
        
        Returns:
            True if usage is at or above warning threshold
        """
        return self.current_count >= (self.monthly_limit * WARNING_THRESHOLD)
    
    def check_limit(self, raise_on_warning: bool = False) -> dict:
        """Check if we're within rate limits.
        
        Args:
            raise_on_warning: If True, raise error when warning threshold reached
            
        Returns:
            Dictionary with current usage info
            
        Raises:
            RateLimitExceededError: If limit is exceeded or warning threshold
                reached (when raise_on_warning=True)
        """
        usage_info = {
            "name": self.name,
            "current_count": self.current_count,
            "monthly_limit": self.monthly_limit,
            "remaining": self.remaining,
            "period": self._get_current_period(),
            "is_warning": self.is_warning,
            "usage_percent": round((self.current_count / self.monthly_limit) * 100, 1)
        }
        
        if self.current_count >= self.monthly_limit:
            logger.error(
                "rate_limit_exceeded",
                **usage_info
            )
            raise RateLimitExceededError(
                f"Rate limit exceeded for {self.name}: "
                f"{self.current_count}/{self.monthly_limit} requests used this month",
                details=usage_info
            )
        
        if self.is_warning:
            logger.warning(
                "rate_limit_warning",
                **usage_info
            )
            if raise_on_warning:
                raise RateLimitExceededError(
                    f"Rate limit warning for {self.name}: "
                    f"{self.current_count}/{self.monthly_limit} requests "
                    f"({usage_info['usage_percent']}% used)",
                    details=usage_info
                )
        
        return usage_info
    
    def increment(self, count: int = 1) -> dict:
        """Increment the usage counter.
        
        Args:
            count: Number of requests to add (default: 1)
            
        Returns:
            Updated usage info dictionary
        """
        # Handle period rollover
        if self._usage_data.get("period") != self._get_current_period():
            self._usage_data = self._new_usage_data()
        
        self._usage_data["count"] = self._usage_data.get("count", 0) + count
        self._save_usage_data()
        
        usage_info = {
            "name": self.name,
            "current_count": self.current_count,
            "remaining": self.remaining,
            "period": self._get_current_period()
        }
        
        logger.debug(
            "rate_limit_incremented",
            **usage_info
        )
        
        return usage_info
    
    def reset(self) -> dict:
        """Reset the usage counter (for testing purposes).
        
        Returns:
            New usage info after reset
        """
        self._usage_data = self._new_usage_data()
        self._save_usage_data()
        
        logger.info(
            "rate_limit_manually_reset",
            name=self.name,
            period=self._get_current_period()
        )
        
        return {
            "name": self.name,
            "current_count": 0,
            "remaining": self.monthly_limit,
            "period": self._get_current_period()
        }
    
    def get_status(self) -> dict:
        """Get current rate limit status.
        
        Returns:
            Dictionary with full usage status
        """
        return {
            "name": self.name,
            "current_count": self.current_count,
            "monthly_limit": self.monthly_limit,
            "remaining": self.remaining,
            "period": self._get_current_period(),
            "is_warning": self.is_warning,
            "usage_percent": round((self.current_count / self.monthly_limit) * 100, 1),
            "warning_threshold_percent": WARNING_THRESHOLD * 100
        }


# Singleton rate limiters for MCP servers
_rate_limiters: dict[str, RateLimiter] = {}


def get_rate_limiter(name: str, monthly_limit: int = DEFAULT_MONTHLY_LIMIT) -> RateLimiter:
    """Get or create a rate limiter instance.
    
    Args:
        name: Identifier for the rate limiter
        monthly_limit: Maximum requests allowed per month
        
    Returns:
        RateLimiter instance
    """
    if name not in _rate_limiters:
        _rate_limiters[name] = RateLimiter(name, monthly_limit)
    return _rate_limiters[name]
