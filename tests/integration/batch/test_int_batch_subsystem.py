# tests/integration/batch/test_int_batch_subsystem.py — v2
"""Integration tests for batch processing subsystem.

Covers: batch/scanner.py, batch/dedup.py, batch/models.py
No Docker required — uses filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ayextractor.batch.models import BatchResult, ScanEntry


# =====================================================================
#  BATCH MODELS
# =====================================================================

class TestBatchModels:

    def test_scan_entry_creation(self):
        entry = ScanEntry(
            file_path="/tmp/doc.pdf",
            filename="doc.pdf",
            format="pdf",
            size_bytes=1024,
            fingerprint_exact="abc123",
            fingerprint_content="def456",
            cache_status="no_match",
        )
        assert entry.format == "pdf"
        assert entry.cache_status == "no_match"
        assert entry.matched_document_id is None

    def test_batch_result_creation(self):
        result = BatchResult(
            scan_root="/tmp/docs",
            total_files_found=10,
            processed=8,
            skipped=2,
            errors=0,
            duration_seconds=5.5,
        )
        assert result.total_files_found == 10
        assert result.entries == []


# =====================================================================
#  BATCH SCANNER
# =====================================================================

class TestBatchScanner:

    def test_scan_empty_directory(self, tmp_path):
        from ayextractor.batch.scanner import BatchScanner
        scanner = BatchScanner()
        entries = scanner.scan(tmp_path)
        assert entries == []

    def test_scan_finds_supported_files(self, tmp_path):
        from ayextractor.batch.scanner import BatchScanner
        # Create test files
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 test")
        (tmp_path / "note.txt").write_text("hello world")
        (tmp_path / "readme.md").write_text("# Title")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "data.csv").write_text("a,b")  # Not supported

        scanner = BatchScanner()
        entries = scanner.scan(tmp_path)
        formats = {e.format for e in entries}
        assert "pdf" in formats
        assert "txt" in formats
        assert "md" in formats
        assert "image" in formats
        assert len(entries) == 4  # csv not included

    def test_scan_recursive(self, tmp_path):
        from ayextractor.batch.scanner import BatchScanner
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "deep.pdf").write_bytes(b"%PDF")
        (tmp_path / "top.txt").write_text("top")

        scanner = BatchScanner()
        entries = scanner.scan(tmp_path, recursive=True)
        assert len(entries) == 2

    def test_scan_non_recursive(self, tmp_path):
        from ayextractor.batch.scanner import BatchScanner
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "deep.pdf").write_bytes(b"%PDF")
        (tmp_path / "top.txt").write_text("top")

        scanner = BatchScanner()
        entries = scanner.scan(tmp_path, recursive=False)
        assert len(entries) == 1  # only top.txt

    def test_scan_with_format_filter(self, tmp_path):
        from ayextractor.batch.scanner import BatchScanner
        (tmp_path / "doc.pdf").write_bytes(b"%PDF")
        (tmp_path / "note.txt").write_text("hello")
        (tmp_path / "img.png").write_bytes(b"\x89PNG")

        scanner = BatchScanner()
        entries = scanner.scan(tmp_path, formats_filter=["pdf"])
        assert len(entries) == 1
        assert entries[0].format == "pdf"

    def test_scan_invalid_directory_raises(self):
        from ayextractor.batch.scanner import BatchScanner
        scanner = BatchScanner()
        with pytest.raises(ValueError):
            scanner.scan(Path("/nonexistent/path"))

    def test_scan_entry_fields_populated(self, tmp_path):
        from ayextractor.batch.scanner import BatchScanner
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 content here")

        scanner = BatchScanner()
        entries = scanner.scan(tmp_path)
        assert len(entries) == 1
        e = entries[0]
        assert e.filename == "doc.pdf"
        assert e.format == "pdf"
        assert e.size_bytes > 0
        assert e.cache_status == "no_match"

    def test_supported_formats_constant(self):
        from ayextractor.batch.scanner import SUPPORTED_FORMATS
        assert ".pdf" in SUPPORTED_FORMATS
        assert ".epub" in SUPPORTED_FORMATS
        assert ".docx" in SUPPORTED_FORMATS
        assert ".md" in SUPPORTED_FORMATS
        assert ".txt" in SUPPORTED_FORMATS


# =====================================================================
#  BATCH DEDUP
# =====================================================================

class TestBatchDedup:

    @pytest.mark.asyncio
    async def test_dedup_no_match(self, tmp_path):
        """New files with no cache → all 'no_match'."""
        from ayextractor.batch.dedup import BatchDeduplicator
        from ayextractor.cache.sqlite_store import SqliteCacheStore

        store = SqliteCacheStore(db_path=str(tmp_path / "cache.db"))
        dedup = BatchDeduplicator(store)

        entries = [
            ScanEntry(
                file_path=str(tmp_path / "a.pdf"),
                filename="a.pdf",
                format="pdf",
                size_bytes=100,
                fingerprint_exact="",
                fingerprint_content="",
                cache_status="no_match",
            ),
        ]
        # Create the actual file for fingerprinting
        (tmp_path / "a.pdf").write_bytes(b"%PDF test content")

        result = await dedup.check_entries(entries)
        assert all(e.cache_status == "no_match" for e in result)

    @pytest.mark.asyncio
    async def test_dedup_populates_fingerprints(self, tmp_path):
        from ayextractor.batch.dedup import BatchDeduplicator
        from ayextractor.cache.sqlite_store import SqliteCacheStore

        store = SqliteCacheStore(db_path=str(tmp_path / "cache.db"))
        dedup = BatchDeduplicator(store)

        (tmp_path / "b.txt").write_text("Hello World")
        entries = [
            ScanEntry(
                file_path=str(tmp_path / "b.txt"),
                filename="b.txt",
                format="txt",
                size_bytes=11,
                fingerprint_exact="",
                fingerprint_content="",
                cache_status="no_match",
            ),
        ]
        result = await dedup.check_entries(entries)
        assert len(result) == 1
        # Fingerprints should now be populated
        assert result[0].fingerprint_exact != ""