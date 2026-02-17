# tests/unit/cache/test_redis_store.py — v1
"""Tests for cache/redis_store.py — mocked Redis client."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from ayextractor.cache.models import CacheEntry, CacheLookupResult, DocumentFingerprint


def _make_fingerprint(**overrides):
    defaults = dict(
        exact_hash="abc123",
        content_hash="def456",
        structural_hash="00000000000000ff",
        semantic_hash="sem_001",
        constellation=["a", "b"],
        timestamp=datetime(2026, 2, 16),
        source_format="pdf",
    )
    defaults.update(overrides)
    return DocumentFingerprint(**defaults)


def _make_entry(fp=None):
    if fp is None:
        fp = _make_fingerprint()
    return CacheEntry(
        document_id="doc_001",
        fingerprint=fp,
        result_path="/output/run_001",
        created_at=datetime(2026, 2, 16),
        pipeline_version="0.1.0",
    )


class TestRedisCacheStore:
    def test_import_error_without_redis(self):
        """Clear ImportError when redis is not available."""
        import sys
        redis_mod = sys.modules.get("redis")
        sys.modules["redis"] = None  # type: ignore[assignment]
        try:
            from ayextractor.cache.redis_store import RedisCacheStore
            with pytest.raises(ImportError, match="redis"):
                RedisCacheStore(redis_url="redis://localhost")
        finally:
            if redis_mod is not None:
                sys.modules["redis"] = redis_mod
            else:
                sys.modules.pop("redis", None)

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        """Test put/get with mocked Redis."""
        storage: dict[str, str] = {}
        index: set[str] = set()

        mock_redis = MagicMock()
        mock_redis.get = lambda k: storage.get(k)
        mock_redis.set = lambda k, v: storage.__setitem__(k, v)
        mock_redis.delete = lambda k: storage.pop(k, None)
        mock_redis.sadd = lambda k, v: index.add(v)
        mock_redis.srem = lambda k, v: index.discard(v)
        mock_redis.smembers = lambda k: index.copy()

        with patch("ayextractor.cache.redis_store.RedisCacheStore.__init__", return_value=None):
            from ayextractor.cache.redis_store import RedisCacheStore
            store = RedisCacheStore.__new__(RedisCacheStore)
            store._client = mock_redis
            store._simhash_threshold = 3

        entry = _make_entry()
        await store.put("key1", entry)
        result = await store.get("key1")
        assert result is not None
        assert result.document_id == "doc_001"

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test delete with mocked Redis."""
        storage: dict[str, str] = {}
        index: set[str] = set()

        mock_redis = MagicMock()
        mock_redis.get = lambda k: storage.get(k)
        mock_redis.set = lambda k, v: storage.__setitem__(k, v)
        mock_redis.delete = lambda k: storage.pop(k, None)
        mock_redis.sadd = lambda k, v: index.add(v)
        mock_redis.srem = lambda k, v: index.discard(v)
        mock_redis.smembers = lambda k: index.copy()

        with patch("ayextractor.cache.redis_store.RedisCacheStore.__init__", return_value=None):
            from ayextractor.cache.redis_store import RedisCacheStore
            store = RedisCacheStore.__new__(RedisCacheStore)
            store._client = mock_redis
            store._simhash_threshold = 3

        entry = _make_entry()
        await store.put("key1", entry)
        await store.delete("key1")
        result = await store.get("key1")
        assert result is None
