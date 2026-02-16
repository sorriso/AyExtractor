# tests/unit/batch/test_dedup.py — v1
"""Tests for batch.dedup — fingerprint comparison against cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ayextractor.batch.dedup import BatchDeduplicator, check_single
from ayextractor.batch.models import ScanEntry
from ayextractor.cache.models import CacheEntry, DocumentFingerprint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_entry(
    filename: str = "test.pdf",
    file_path: str = "/tmp/test.pdf",
    fmt: str = "pdf",
    size: int = 1024,
) -> ScanEntry:
    return ScanEntry(
        file_path=file_path,
        filename=filename,
        format=fmt,
        size_bytes=size,
        fingerprint_exact="",
        fingerprint_content="",
        cache_status="no_match",
    )


def _make_fingerprint(
    exact: str = "abc123",
    content: str = "def456",
    structural: str = "ghi789",
) -> DocumentFingerprint:
    from datetime import datetime, timezone
    return DocumentFingerprint(
        exact_hash=exact,
        content_hash=content,
        structural_hash=structural,
        semantic_hash="sem_placeholder",
        constellation=[],
        timestamp=datetime.now(timezone.utc),
        source_format="pdf",
    )


def _make_cache_entry(
    doc_id: str = "doc_1",
    exact: str = "abc123",
    content: str = "def456",
) -> CacheEntry:
    from datetime import datetime, timezone
    fp = _make_fingerprint(exact=exact, content=content)
    return CacheEntry(
        document_id=doc_id,
        fingerprint=fp,
        result_path="/tmp/results",
        created_at=datetime.now(timezone.utc),
        pipeline_version="0.3.0",
    )


def _make_mock_cache_store(entries: list[CacheEntry] | None = None) -> AsyncMock:
    store = AsyncMock()
    store.list_entries.return_value = entries or []
    return store


# ---------------------------------------------------------------------------
# Tests — BatchDeduplicator
# ---------------------------------------------------------------------------

class TestBatchDeduplicator:
    @pytest.mark.asyncio
    async def test_no_cache_entries_all_no_match(self, tmp_path: Path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"content")

        store = _make_mock_cache_store([])
        dedup = BatchDeduplicator(cache_store=store)

        entry = _make_entry(file_path=str(f))
        results = await dedup.check_entries([entry])

        assert len(results) == 1
        assert results[0].cache_status == "no_match"
        assert results[0].fingerprint_exact != ""

    @pytest.mark.asyncio
    async def test_exact_match_detected(self, tmp_path: Path):
        content = b"exact content"
        f = tmp_path / "test.pdf"
        f.write_bytes(content)

        # Pre-compute the expected hash
        import hashlib
        exact_hash = hashlib.sha256(content).hexdigest()

        cached = _make_cache_entry(doc_id="existing_doc", exact=exact_hash)
        store = _make_mock_cache_store([cached])
        dedup = BatchDeduplicator(cache_store=store)

        entry = _make_entry(file_path=str(f))
        results = await dedup.check_entries([entry])

        assert results[0].cache_status == "exact_match"
        assert results[0].matched_document_id == "existing_doc"

    @pytest.mark.asyncio
    async def test_no_match_when_different(self, tmp_path: Path):
        f = tmp_path / "new.pdf"
        f.write_bytes(b"completely new content")

        cached = _make_cache_entry(
            doc_id="old_doc", exact="oldhash", content="oldcontent",
        )
        store = _make_mock_cache_store([cached])
        dedup = BatchDeduplicator(cache_store=store)

        entry = _make_entry(file_path=str(f))
        results = await dedup.check_entries([entry])

        assert results[0].cache_status == "no_match"
        assert results[0].matched_document_id is None

    @pytest.mark.asyncio
    async def test_handles_read_error_gracefully(self, tmp_path: Path):
        """Entry with non-existent file should be marked no_match."""
        store = _make_mock_cache_store([])
        dedup = BatchDeduplicator(cache_store=store)

        entry = _make_entry(file_path="/nonexistent/file.pdf")
        results = await dedup.check_entries([entry])

        assert results[0].cache_status == "no_match"

    @pytest.mark.asyncio
    async def test_multiple_entries_mixed(self, tmp_path: Path):
        import hashlib

        # File 1: will match cache
        f1 = tmp_path / "existing.pdf"
        f1.write_bytes(b"known content")
        h1 = hashlib.sha256(b"known content").hexdigest()

        # File 2: new file
        f2 = tmp_path / "new.md"
        f2.write_bytes(b"fresh content")

        cached = _make_cache_entry(doc_id="cached_doc", exact=h1)
        store = _make_mock_cache_store([cached])
        dedup = BatchDeduplicator(cache_store=store)

        entries = [
            _make_entry(filename="existing.pdf", file_path=str(f1)),
            _make_entry(filename="new.md", file_path=str(f2), fmt="md"),
        ]
        results = await dedup.check_entries(entries)

        assert results[0].cache_status == "exact_match"
        assert results[1].cache_status == "no_match"


# ---------------------------------------------------------------------------
# Tests — check_single()
# ---------------------------------------------------------------------------

class TestCheckSingle:
    @pytest.mark.asyncio
    async def test_check_single_returns_entry(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_bytes(b"markdown content")

        store = _make_mock_cache_store([])
        result = await check_single(f, store)

        assert isinstance(result, ScanEntry)
        assert result.filename == "doc.md"
        assert result.format == "md"
        assert result.cache_status == "no_match"
