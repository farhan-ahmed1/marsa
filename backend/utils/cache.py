"""In-memory response cache with TTL for repeated queries.

Caches final reports keyed by a normalized query hash. If the same (or
semantically identical after normalization) query is submitted within
the TTL window, the cached result is returned immediately.

Design:
- Thread-safe via ``asyncio.Lock``.
- Automatic eviction of expired entries on access.
- Configurable TTL (default 1 hour).
- Maximum cache size to bound memory usage.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_TTL_SECONDS: int = 3600  # 1 hour
MAX_CACHE_SIZE: int = 100


def _normalize_query(query: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(query.lower().strip().split())


def _cache_key(query: str) -> str:
    """Deterministic hash of the normalized query."""
    return hashlib.sha256(_normalize_query(query).encode()).hexdigest()


@dataclass
class CacheEntry:
    """A single cached response."""

    key: str
    query: str
    response: dict[str, Any]
    created_at: float = field(default_factory=time.monotonic)
    ttl: float = DEFAULT_TTL_SECONDS

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl


class ResponseCache:
    """Simple async-safe LRU-like cache for API responses.

    Attributes:
        ttl: Time-to-live in seconds for cache entries.
        max_size: Maximum number of entries before eviction.
    """

    def __init__(
        self,
        ttl: float = DEFAULT_TTL_SECONDS,
        max_size: int = MAX_CACHE_SIZE,
    ) -> None:
        self.ttl = ttl
        self.max_size = max_size
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, query: str) -> Optional[dict[str, Any]]:
        """Look up a cached response.

        Args:
            query: The raw user query.

        Returns:
            Cached response dict or ``None`` on miss / expiry.
        """
        key = _cache_key(query)
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired:
                del self._store[key]
                self._misses += 1
                logger.debug("cache_expired", key=key[:12])
                return None
            self._hits += 1
            logger.info(
                "cache_hit",
                key=key[:12],
                query_preview=query[:60],
                age_s=round(time.monotonic() - entry.created_at, 1),
            )
            return entry.response

    async def put(self, query: str, response: dict[str, Any]) -> None:
        """Store a response in the cache.

        If the cache exceeds ``max_size``, the oldest entry is evicted.

        Args:
            query: Raw user query.
            response: Serializable response dict.
        """
        key = _cache_key(query)
        async with self._lock:
            # Evict oldest if full
            if len(self._store) >= self.max_size and key not in self._store:
                oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
                del self._store[oldest_key]
                logger.debug("cache_evicted", evicted_key=oldest_key[:12])

            self._store[key] = CacheEntry(
                key=key,
                query=query,
                response=response,
                ttl=self.ttl,
            )
            logger.info("cache_stored", key=key[:12], query_preview=query[:60])

    async def invalidate(self, query: str) -> bool:
        """Remove a specific cache entry.

        Returns:
            ``True`` if an entry was removed.
        """
        key = _cache_key(query)
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def clear(self) -> int:
        """Drop all entries. Returns removed count."""
        async with self._lock:
            count = len(self._store)
            self._store.clear()
            logger.info("cache_cleared", removed=count)
            return count

    @property
    def stats(self) -> dict[str, int]:
        """Cache hit/miss statistics."""
        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(
                self._hits / max(self._hits + self._misses, 1) * 100, 1
            ),
        }


# Module-level singleton
response_cache = ResponseCache()
