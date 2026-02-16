# tests/unit/batch/test_scanner.py — v1
"""Tests for batch.scanner — directory scanning and file discovery."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ayextractor.batch.models import BatchResult, ScanEntry
from ayextractor.batch.scanner import SUPPORTED_FORMATS, BatchScanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_files(tmp_path: Path) -> dict[str, Path]:
    """Create a set of test files in tmp_path."""
    files = {}
    for name, content in [
        ("report.pdf", b"%PDF-1.4 fake"),
        ("notes.md", b"# Notes"),
        ("data.txt", b"plain text"),
        ("photo.png", b"\x89PNG fake"),
        ("ignore.xlsx", b"spreadsheet"),
        ("sub/nested.docx", b"docx content"),
    ]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        files[name] = p
    return files


# ---------------------------------------------------------------------------
# Tests — scan()
# ---------------------------------------------------------------------------

class TestBatchScannerScan:
    def test_scan_finds_supported_formats(self, tmp_path: Path):
        _create_test_files(tmp_path)
        scanner = BatchScanner()
        entries = scanner.scan(tmp_path)

        filenames = {e.filename for e in entries}
        assert "report.pdf" in filenames
        assert "notes.md" in filenames
        assert "data.txt" in filenames
        assert "photo.png" in filenames
        assert "ignore.xlsx" not in filenames  # Not supported

    def test_scan_recursive(self, tmp_path: Path):
        _create_test_files(tmp_path)
        scanner = BatchScanner()
        entries = scanner.scan(tmp_path, recursive=True)
        filenames = {e.filename for e in entries}
        assert "nested.docx" in filenames

    def test_scan_non_recursive(self, tmp_path: Path):
        _create_test_files(tmp_path)
        scanner = BatchScanner()
        entries = scanner.scan(tmp_path, recursive=False)
        filenames = {e.filename for e in entries}
        assert "nested.docx" not in filenames

    def test_scan_formats_filter(self, tmp_path: Path):
        _create_test_files(tmp_path)
        scanner = BatchScanner()
        entries = scanner.scan(tmp_path, formats_filter=["pdf"])
        assert all(e.format == "pdf" for e in entries)
        assert len(entries) == 1

    def test_scan_empty_directory(self, tmp_path: Path):
        scanner = BatchScanner()
        entries = scanner.scan(tmp_path)
        assert entries == []

    def test_scan_invalid_root_raises(self, tmp_path: Path):
        scanner = BatchScanner()
        with pytest.raises(ValueError, match="not a directory"):
            scanner.scan(tmp_path / "nonexistent")

    def test_scan_entry_fields(self, tmp_path: Path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf content")
        scanner = BatchScanner()
        entries = scanner.scan(tmp_path)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.filename == "test.pdf"
        assert entry.format == "pdf"
        assert entry.size_bytes == 16
        assert entry.cache_status == "no_match"
        assert entry.matched_document_id is None


# ---------------------------------------------------------------------------
# Tests — scan_and_process()
# ---------------------------------------------------------------------------

class TestBatchScannerProcess:
    @pytest.mark.asyncio
    async def test_scan_and_process_calls_facade(self, tmp_path: Path):
        # Create one file to process
        f = tmp_path / "doc.md"
        f.write_bytes(b"# Test")

        scanner = BatchScanner()

        with patch("ayextractor.api.facade.analyze", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock()
            result = await scanner.scan_and_process(
                scan_root=tmp_path,
                output_path=tmp_path / "output",
            )

        assert isinstance(result, BatchResult)
        assert result.total_files_found == 1
        assert result.processed == 1
        assert result.skipped == 0
        assert result.errors == 0
        assert result.duration_seconds >= 0
        assert mock_analyze.call_count == 1

    @pytest.mark.asyncio
    async def test_scan_and_process_handles_errors(self, tmp_path: Path):
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"bad")

        scanner = BatchScanner()

        with patch(
            "ayextractor.api.facade.analyze",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            result = await scanner.scan_and_process(
                scan_root=tmp_path,
                output_path=tmp_path / "output",
            )

        assert result.processed == 0
        assert result.errors == 1


# ---------------------------------------------------------------------------
# Tests — supported formats
# ---------------------------------------------------------------------------

class TestSupportedFormats:
    def test_all_expected_formats(self):
        expected = {"pdf", "epub", "docx", "md", "txt", "image"}
        actual = set(SUPPORTED_FORMATS.values())
        assert actual == expected

    def test_image_extensions(self):
        image_exts = [k for k, v in SUPPORTED_FORMATS.items() if v == "image"]
        assert ".png" in image_exts
        assert ".jpg" in image_exts
        assert ".jpeg" in image_exts
        assert ".webp" in image_exts
