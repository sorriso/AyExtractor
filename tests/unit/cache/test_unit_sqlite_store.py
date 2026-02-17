# tests/unit/cache/test_sqlite_store.py — v1
"""Tests for cache/sqlite_store.py — full functional tests (stdlib sqlite3)."""

from __future__ import annotations

from datetime import datetime

import pytest

from ayextractor.cache.models import CacheEntry, DocumentFingerprint
from ayextractor.cache.sqlite_store import SqliteCacheStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_cache.db"
    return SqliteCacheStore(db_path=db_path)


@pytest.fixture
def sample_fingerprint():
    return DocumentFingerprint(
        exact_hash="abc123",
        content_hash="def456",
        structural_hash="00000000000000ff",
        semantic_hash="sem_hash_001",
        constellation=["chunk_a", "chunk_b"],
        timestamp=datetime(2026, 2, 16),
        source_format="pdf",
    )


@pytest.fixture
def sample_entry(sample_fingerprint):
    return CacheEntry(
        document_id="doc_001",
        fingerprint=sample_fingerprint,
        result_path="/output/run_20260216",
        created_at=datetime(2026, 2, 16),
        pipeline_version="0.1.0",
    )


class TestSqliteCacheStore:
    @pytest.mark.asyncio
    async def test_put_and_get(self, store, sample_entry):
        await store.put("key1", sample_entry)
        result = await store.get("key1")
        assert result is not None
        assert result.document_id == "doc_001"

    @pytest.mark.asyncio
    async def test_get_missing(self, store):
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store, sample_entry):
        await store.put("key1", sample_entry)
        await store.delete("key1")
        result = await store.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_entries(self, store, sample_entry):
        await store.put("key1", sample_entry)
        await store.put("key2", sample_entry)
        entries = await store.list_entries()
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_lookup_exact(self, store, sample_entry, sample_fingerprint):
        await store.put("key1", sample_entry)
        result = await store.lookup_fingerprint(sample_fingerprint)
        assert result.hit_level == "exact"
        assert result.is_reusable is True
        assert result.similarity_score == 1.0

    @pytest.mark.asyncio
    async def test_lookup_content(self, store, sample_entry):
        await store.put("key1", sample_entry)
        # Different exact hash, same content hash
        fp = DocumentFingerprint(
            exact_hash="different_exact",
            content_hash="def456",
            structural_hash="0000000000000000",
            semantic_hash="sem_other",
            constellation=[],
            timestamp=datetime(2026, 2, 16),
            source_format="pdf",
        )
        result = await store.lookup_fingerprint(fp)
        assert result.hit_level == "content"
        assert result.is_reusable is True

    @pytest.mark.asyncio
    async def test_lookup_structural(self, store, sample_entry):
        await store.put("key1", sample_entry)
        # Different hashes, close structural hash (1 bit different)
        fp = DocumentFingerprint(
            exact_hash="zzz",
            content_hash="yyy",
            structural_hash="00000000000000fe",  # 1 bit diff from ff
            semantic_hash="sem_zzz",
            constellation=[],
            timestamp=datetime(2026, 2, 16),
            source_format="pdf",
        )
        result = await store.lookup_fingerprint(fp)
        assert result.hit_level == "structural"
        assert result.is_reusable is False
        assert result.similarity_score > 0.9

    @pytest.mark.asyncio
    async def test_lookup_miss(self, store, sample_entry):
        await store.put("key1", sample_entry)
        fp = DocumentFingerprint(
            exact_hash="completely_different",
            content_hash="also_different",
            structural_hash="ffffffffffffffff",
            semantic_hash="sem_miss",
            constellation=[],
            timestamp=datetime(2026, 2, 16),
            source_format="pdf",
        )
        result = await store.lookup_fingerprint(fp)
        assert result.hit_level is None

    @pytest.mark.asyncio
    async def test_upsert(self, store, sample_fingerprint):
        """Putting same key twice updates the entry."""
        entry1 = CacheEntry(
            document_id="doc_v1",
            fingerprint=sample_fingerprint,
            result_path="/output/v1",
            created_at=datetime(2026, 1, 1),
            pipeline_version="0.1.0",
        )
        entry2 = CacheEntry(
            document_id="doc_v2",
            fingerprint=sample_fingerprint,
            result_path="/output/v2",
            created_at=datetime(2026, 2, 1),
            pipeline_version="0.1.0",
        )
        await store.put("key1", entry1)
        await store.put("key1", entry2)
        result = await store.get("key1")
        assert result is not None
        assert result.document_id == "doc_v2"
        entries = await store.list_entries()
        assert len(entries) == 1
