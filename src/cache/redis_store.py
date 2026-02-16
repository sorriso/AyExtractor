# src/cache/redis_store.py — v1
"""Redis-based cache store (CACHE_BACKEND=redis).

Requires 'redis' package: pip install redis.
Suitable for distributed/multi-instance deployments.
See spec §8.3 for cache lookup logic.
"""

from __future__ import annotations

import json
import logging

from ayextractor.cache.base_cache_store import BaseCacheStore
from ayextractor.cache.fingerprint import hamming_distance
from ayextractor.cache.models import CacheEntry, CacheLookupResult, DocumentFingerprint

logger = logging.getLogger(__name__)

_KEY_PREFIX = "ayextractor:cache:"
_INDEX_KEY = "ayextractor:cache:__index__"


class RedisCacheStore(BaseCacheStore):
    """Redis-backed cache store for distributed deployments."""

    def __init__(
        self, redis_url: str, simhash_threshold: int = 3
    ) -> None:
        try:
            import redis
        except ImportError as e:
            raise ImportError(
                "redis package required: pip install redis"
            ) from e

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._simhash_threshold = simhash_threshold

    async def get(self, key: str) -> CacheEntry | None:
        """Retrieve cache entry by key."""
        data = self._client.get(f"{_KEY_PREFIX}{key}")
        if data is None:
            return None
        try:
            return CacheEntry(**json.loads(data))
        except Exception as e:
            logger.warning("Failed to deserialize cache entry %s: %s", key, e)
            return None

    async def put(self, key: str, entry: CacheEntry) -> None:
        """Store a cache entry."""
        redis_key = f"{_KEY_PREFIX}{key}"
        self._client.set(redis_key, entry.model_dump_json())
        # Maintain a set of all cache keys for list_entries / fingerprint scan
        self._client.sadd(_INDEX_KEY, key)

    async def delete(self, key: str) -> None:
        """Remove a cache entry."""
        self._client.delete(f"{_KEY_PREFIX}{key}")
        self._client.srem(_INDEX_KEY, key)

    async def lookup_fingerprint(
        self, fingerprint: DocumentFingerprint
    ) -> CacheLookupResult:
        """Multi-level fingerprint lookup (spec §8.3).

        Scans all entries since Redis has no secondary index on hash fields.
        For large deployments, consider adding Redis Search module for indexing.
        """
        entries = await self.list_entries()

        for entry in entries:
            fp = entry.fingerprint

            # Level 1: exact hash
            if fp.exact_hash == fingerprint.exact_hash:
                return CacheLookupResult(
                    hit_level="exact",
                    matched_entry=entry,
                    similarity_score=1.0,
                    is_reusable=True,
                )

            # Level 2: content hash
            if fp.content_hash == fingerprint.content_hash:
                return CacheLookupResult(
                    hit_level="content",
                    matched_entry=entry,
                    similarity_score=1.0,
                    is_reusable=True,
                )

            # Level 3: structural hash
            if fp.structural_hash and fingerprint.structural_hash:
                distance = hamming_distance(
                    fp.structural_hash, fingerprint.structural_hash
                )
                if distance <= self._simhash_threshold:
                    return CacheLookupResult(
                        hit_level="structural",
                        matched_entry=entry,
                        similarity_score=1.0 - (distance / 64.0),
                        is_reusable=False,
                    )

        return CacheLookupResult()

    async def list_entries(self) -> list[CacheEntry]:
        """List all cached entries."""
        keys = self._client.smembers(_INDEX_KEY)
        entries: list[CacheEntry] = []
        for key in keys:
            entry = await self.get(key)
            if entry is not None:
                entries.append(entry)
        return entries

    def close(self) -> None:
        """Close the Redis connection."""
        self._client.close()
