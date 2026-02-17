# tests/unit/cache/test_cache_factory.py — v3
"""Tests for cache/cache_factory.py."""

from __future__ import annotations

import pytest

from ayextractor.cache.cache_factory import create_cache_store
from ayextractor.cache.json_store import JsonCacheStore
from ayextractor.cache.sqlite_store import SqliteCacheStore


class TestCreateCacheStore:
    def test_default_json(self):
        store = create_cache_store()
        assert isinstance(store, JsonCacheStore)

    def test_sqlite_backend(self, tmp_path):
        from ayextractor.config.settings import Settings
        s = Settings(_env_file=None, cache_backend="sqlite", cache_root=tmp_path)
        store = create_cache_store(s)
        assert isinstance(store, SqliteCacheStore)

    def test_redis_missing_url(self):
        from ayextractor.config.settings import Settings
        s = Settings(_env_file=None, cache_backend="redis", cache_redis_url="")
        with pytest.raises(ValueError, match="CACHE_REDIS_URL"):
            create_cache_store(s)

    def test_unsupported_backend(self):
        """Settings validation rejects invalid backends before factory is reached."""
        from ayextractor.config.settings import Settings
        with pytest.raises((ValueError, Exception)):
            s = Settings(_env_file=None, cache_backend="nonexistent")
            create_cache_store(s)
