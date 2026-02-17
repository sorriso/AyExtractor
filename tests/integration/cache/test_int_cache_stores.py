# tests/integration/cache/test_int_cache_stores.py — v2
"""Integration tests for cache backends: JSON + SQLite.

No external services required.
Coverage targets: json_store.py, sqlite_store.py, cache_factory.py, fingerprint.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from ayextractor.cache.models import CacheEntry, DocumentFingerprint


def _make_fingerprint(
    exact: str = "abc123",
    content: str = "def456",
    structural: str = "0" * 64,
    semantic: str = "ghi789",
) -> DocumentFingerprint:
    return DocumentFingerprint(
        exact_hash=exact, content_hash=content,
        structural_hash=structural, semantic_hash=semantic,
        constellation=["chunk1", "chunk2"],
        timestamp=datetime.now(timezone.utc), source_format="pdf",
    )


def _make_entry(
    doc_id: str = "doc_001",
    exact: str = "abc123",
    content: str = "def456",
) -> CacheEntry:
    return CacheEntry(
        document_id=doc_id,
        fingerprint=_make_fingerprint(exact=exact, content=content),
        result_path="/output/doc_001/runs/run_001",
        created_at=datetime.now(timezone.utc),
        pipeline_version="0.3.1",
    )


class TestJsonCacheStore:

    @pytest.mark.asyncio
    async def test_put_and_get(self, tmp_path: Path):
        from ayextractor.cache.json_store import JsonCacheStore
        store = JsonCacheStore(cache_root=tmp_path)
        entry = _make_entry()
        await store.put("key1", entry)
        result = await store.get("key1")
        assert result is not None
        assert result.document_id == "doc_001"
        assert result.pipeline_version == "0.3.1"

    @pytest.mark.asyncio
    async def test_get_missing(self, tmp_path: Path):
        from ayextractor.cache.json_store import JsonCacheStore
        store = JsonCacheStore(cache_root=tmp_path)
        assert await store.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_put_overwrite(self, tmp_path: Path):
        from ayextractor.cache.json_store import JsonCacheStore
        store = JsonCacheStore(cache_root=tmp_path)
        entry1 = _make_entry(doc_id="old")
        entry2 = _make_entry(doc_id="new")
        await store.put("k", entry1)
        await store.put("k", entry2)
        result = await store.get("k")
        assert result.document_id == "new"

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path: Path):
        from ayextractor.cache.json_store import JsonCacheStore
        store = JsonCacheStore(cache_root=tmp_path)
        await store.put("k", _make_entry())
        await store.delete("k")
        assert await store.get("k") is None

    @pytest.mark.asyncio
    async def test_delete_missing_no_error(self, tmp_path: Path):
        from ayextractor.cache.json_store import JsonCacheStore
        store = JsonCacheStore(cache_root=tmp_path)
        await store.delete("nope")  # should not raise

    @pytest.mark.asyncio
    async def test_list_entries(self, tmp_path: Path):
        from ayextractor.cache.json_store import JsonCacheStore
        store = JsonCacheStore(cache_root=tmp_path)
        await store.put("k1", _make_entry(doc_id="d1", exact="a1"))
        await store.put("k2", _make_entry(doc_id="d2", exact="a2"))
        entries = await store.list_entries()
        assert len(entries) == 2
        ids = {e.document_id for e in entries}
        assert ids == {"d1", "d2"}

    @pytest.mark.asyncio
    async def test_lookup_exact_hit(self, tmp_path: Path):
        from ayextractor.cache.json_store import JsonCacheStore
        store = JsonCacheStore(cache_root=tmp_path)
        entry = _make_entry(exact="match_me")
        await store.put("k", entry)
        result = await store.lookup_fingerprint(
            _make_fingerprint(exact="match_me")
        )
        assert result.hit_level == "exact"
        assert result.is_reusable is True
        assert result.matched_entry.document_id == "doc_001"

    @pytest.mark.asyncio
    async def test_lookup_content_hit(self, tmp_path: Path):
        from ayextractor.cache.json_store import JsonCacheStore
        store = JsonCacheStore(cache_root=tmp_path)
        entry = _make_entry(exact="no_match", content="content_match")
        await store.put("k", entry)
        result = await store.lookup_fingerprint(
            _make_fingerprint(exact="different", content="content_match")
        )
        assert result.hit_level == "content"
        assert result.is_reusable is True

    @pytest.mark.asyncio
    async def test_lookup_miss(self, tmp_path: Path):
        from ayextractor.cache.json_store import JsonCacheStore
        store = JsonCacheStore(cache_root=tmp_path)
        await store.put("k", _make_entry(exact="aaa", content="bbb"))
        result = await store.lookup_fingerprint(
            _make_fingerprint(exact="xxx", content="yyy", structural="1" * 64)
        )
        assert result.hit_level is None
        assert result.matched_entry is None

    @pytest.mark.asyncio
    async def test_corrupted_file_returns_none(self, tmp_path: Path):
        from ayextractor.cache.json_store import JsonCacheStore
        store = JsonCacheStore(cache_root=tmp_path)
        # Write corrupt JSON
        (tmp_path / "bad.json").write_text("{invalid json", encoding="utf-8")
        assert await store.get("bad") is None


class TestSqliteCacheStore:

    @pytest.mark.asyncio
    async def test_put_and_get(self, tmp_path: Path):
        from ayextractor.cache.sqlite_store import SqliteCacheStore
        store = SqliteCacheStore(db_path=tmp_path / "cache.db")
        entry = _make_entry()
        await store.put("key1", entry)
        result = await store.get("key1")
        assert result is not None
        assert result.document_id == "doc_001"

    @pytest.mark.asyncio
    async def test_get_missing(self, tmp_path: Path):
        from ayextractor.cache.sqlite_store import SqliteCacheStore
        store = SqliteCacheStore(db_path=tmp_path / "cache.db")
        assert await store.get("missing") is None

    @pytest.mark.asyncio
    async def test_upsert(self, tmp_path: Path):
        from ayextractor.cache.sqlite_store import SqliteCacheStore
        store = SqliteCacheStore(db_path=tmp_path / "cache.db")
        await store.put("k", _make_entry(doc_id="old"))
        await store.put("k", _make_entry(doc_id="new"))
        result = await store.get("k")
        assert result.document_id == "new"

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path: Path):
        from ayextractor.cache.sqlite_store import SqliteCacheStore
        store = SqliteCacheStore(db_path=tmp_path / "cache.db")
        await store.put("k", _make_entry())
        await store.delete("k")
        assert await store.get("k") is None

    @pytest.mark.asyncio
    async def test_list_entries(self, tmp_path: Path):
        from ayextractor.cache.sqlite_store import SqliteCacheStore
        store = SqliteCacheStore(db_path=tmp_path / "cache.db")
        await store.put("k1", _make_entry(doc_id="d1", exact="a1"))
        await store.put("k2", _make_entry(doc_id="d2", exact="a2"))
        entries = await store.list_entries()
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_lookup_exact_hit(self, tmp_path: Path):
        from ayextractor.cache.sqlite_store import SqliteCacheStore
        store = SqliteCacheStore(db_path=tmp_path / "cache.db")
        await store.put("k", _make_entry(exact="match_me"))
        result = await store.lookup_fingerprint(
            _make_fingerprint(exact="match_me")
        )
        assert result.hit_level == "exact"
        assert result.is_reusable is True

    @pytest.mark.asyncio
    async def test_lookup_content_hit(self, tmp_path: Path):
        from ayextractor.cache.sqlite_store import SqliteCacheStore
        store = SqliteCacheStore(db_path=tmp_path / "cache.db")
        await store.put("k", _make_entry(exact="no", content="yes"))
        result = await store.lookup_fingerprint(
            _make_fingerprint(exact="different", content="yes")
        )
        assert result.hit_level == "content"

    @pytest.mark.asyncio
    async def test_lookup_miss(self, tmp_path: Path):
        from ayextractor.cache.sqlite_store import SqliteCacheStore
        store = SqliteCacheStore(db_path=tmp_path / "cache.db")
        result = await store.lookup_fingerprint(
            _make_fingerprint(exact="xxx", content="yyy")
        )
        assert result.hit_level is None

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path: Path):
        from ayextractor.cache.sqlite_store import SqliteCacheStore
        db = tmp_path / "persist.db"
        store1 = SqliteCacheStore(db_path=db)
        await store1.put("k", _make_entry())
        store1.close()
        store2 = SqliteCacheStore(db_path=db)
        result = await store2.get("k")
        assert result is not None
        store2.close()


class TestCacheFactory:

    def test_create_json(self, tmp_path: Path):
        from ayextractor.cache.cache_factory import create_cache_store
        from ayextractor.config.settings import Settings
        settings = Settings(
            _env_file=None, cache_backend="json",
            cache_root=tmp_path / "cache",
        )
        store = create_cache_store(settings)
        assert store is not None

    def test_create_sqlite(self, tmp_path: Path):
        from ayextractor.cache.cache_factory import create_cache_store
        from ayextractor.config.settings import Settings
        settings = Settings(
            _env_file=None, cache_backend="sqlite",
            cache_root=tmp_path / "cache",
        )
        store = create_cache_store(settings)
        assert store is not None
