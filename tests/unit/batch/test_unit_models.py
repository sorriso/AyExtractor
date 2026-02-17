# tests/unit/batch/test_models.py — v1
"""Tests for batch/models.py — scan and batch result models."""

from __future__ import annotations

from ayextractor.batch.models import BatchResult, ScanEntry


class TestScanEntry:
    def test_create(self):
        e = ScanEntry(
            file_path="/data/report.pdf", filename="report.pdf", format="pdf",
            size_bytes=2_000_000, fingerprint_exact="aabb", fingerprint_content="ccdd",
            cache_status="no_match",
        )
        assert e.matched_document_id is None

    def test_cache_hit(self):
        e = ScanEntry(
            file_path="/data/report.pdf", filename="report.pdf", format="pdf",
            size_bytes=100, fingerprint_exact="x", fingerprint_content="y",
            cache_status="exact_match", matched_document_id="doc_001",
        )
        assert e.cache_status == "exact_match"


class TestBatchResult:
    def test_create(self):
        br = BatchResult(
            scan_root="/data", total_files_found=10,
            processed=7, skipped=2, errors=1, duration_seconds=120.5,
        )
        assert br.processed + br.skipped + br.errors == 10
