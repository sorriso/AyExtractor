# src/batch/models.py — v1
"""Batch processing models: ScanEntry, BatchResult.

See spec §24 for full documentation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScanEntry(BaseModel):
    """A single file discovered during batch scan."""

    file_path: str
    filename: str
    format: str
    size_bytes: int
    fingerprint_exact: str
    fingerprint_content: str
    cache_status: Literal["exact_match", "content_match", "near_match", "no_match"]
    matched_document_id: str | None = None


class BatchResult(BaseModel):
    """Summary result of a batch scan + processing run."""

    scan_root: str
    total_files_found: int
    processed: int
    skipped: int
    errors: int
    entries: list[ScanEntry] = Field(default_factory=list)
    duration_seconds: float
