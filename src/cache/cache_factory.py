# src/cache/cache_factory.py — v2
"""Factory for cache store instantiation.

See spec §30.8 — CacheStore row.
"""

from __future__ import annotations

from ayextractor.cache.base_cache_store import BaseCacheStore
from ayextractor.config.settings import Settings


def create_cache_store(settings: Settings | None = None) -> BaseCacheStore:
    """Instantiate the configured cache backend.

    Args:
        settings: Application settings. Defaults to JSON backend.

    Returns:
        Configured BaseCacheStore implementation.
    """
    backend = "json" if settings is None else settings.cache_backend
    simhash_threshold = 3 if settings is None else settings.simhash_threshold

    if backend == "json":
        from ayextractor.cache.json_store import JsonCacheStore
        cache_root = "output/.cache" if settings is None else str(settings.cache_root)
        return JsonCacheStore(
            cache_root=cache_root, simhash_threshold=simhash_threshold
        )

    if backend == "sqlite":
        from ayextractor.cache.sqlite_store import SqliteCacheStore
        cache_root = "output/.cache" if settings is None else str(settings.cache_root)
        db_path = f"{cache_root}/ayextractor_cache.db"
        return SqliteCacheStore(
            db_path=db_path, simhash_threshold=simhash_threshold
        )

    if backend == "redis":
        from ayextractor.cache.redis_store import RedisCacheStore
        if settings is None or not settings.cache_redis_url:
            raise ValueError(
                "CACHE_REDIS_URL must be set when CACHE_BACKEND=redis"
            )
        return RedisCacheStore(
            redis_url=settings.cache_redis_url,
            simhash_threshold=simhash_threshold,
        )

    raise ValueError(f"Unsupported cache backend: {backend!r}")
