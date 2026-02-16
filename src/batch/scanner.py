# src/batch/scanner.py — v1
"""Batch scanner — directory scanning, file discovery, and processing.

Scans a directory for supported document formats, deduplicates against
the cache, and queues new documents for analysis via the facade.

See spec §24 for full documentation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ayextractor.batch.models import BatchResult, ScanEntry

if TYPE_CHECKING:
    from ayextractor.cache.base_cache_store import BaseCacheStore
    from ayextractor.config.settings import Settings

logger = logging.getLogger(__name__)

# Supported file extensions mapped to format names
SUPPORTED_FORMATS: dict[str, str] = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".docx": "docx",
    ".md": "md",
    ".txt": "txt",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
}


class BatchScanner:
    """Scan directories for documents and process them.

    Workflow:
        1. List all files matching supported formats (recursive if enabled)
        2. For each file: compute fingerprint, lookup in cache
        3. Queue new documents for processing
        4. Return BatchResult with stats
    """

    def __init__(
        self,
        settings: Settings | None = None,
        cache_store: BaseCacheStore | None = None,
    ) -> None:
        self._settings = settings
        self._cache_store = cache_store

    def scan(
        self,
        scan_root: Path,
        recursive: bool = True,
        formats_filter: list[str] | None = None,
    ) -> list[ScanEntry]:
        """Discover all supported files in directory.

        Args:
            scan_root: Root directory to scan.
            recursive: If True, scan subdirectories recursively.
            formats_filter: If provided, only include these formats.

        Returns:
            List of ScanEntry objects (without cache status yet).
        """
        if not scan_root.is_dir():
            msg = f"Scan root is not a directory: {scan_root}"
            raise ValueError(msg)

        entries: list[ScanEntry] = []
        allowed_formats = set(formats_filter) if formats_filter else None

        pattern_fn = scan_root.rglob if recursive else scan_root.glob
        for path in sorted(pattern_fn("*")):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            fmt = SUPPORTED_FORMATS.get(ext)
            if fmt is None:
                continue
            if allowed_formats and fmt not in allowed_formats:
                continue

            entry = ScanEntry(
                file_path=str(path.resolve()),
                filename=path.name,
                format=fmt,
                size_bytes=path.stat().st_size,
                fingerprint_exact="",
                fingerprint_content="",
                cache_status="no_match",
                matched_document_id=None,
            )
            entries.append(entry)

        logger.info(
            "Scanned %s: found %d supported files (recursive=%s)",
            scan_root, len(entries), recursive,
        )
        return entries

    async def scan_and_dedup(
        self,
        scan_root: Path,
        recursive: bool = True,
        formats_filter: list[str] | None = None,
    ) -> list[ScanEntry]:
        """Scan directory and deduplicate against cache.

        Returns ScanEntry list with cache_status populated.
        """
        entries = self.scan(scan_root, recursive, formats_filter)

        if self._cache_store is None:
            return entries

        from ayextractor.batch.dedup import BatchDeduplicator

        dedup = BatchDeduplicator(cache_store=self._cache_store)
        return await dedup.check_entries(entries)

    async def scan_and_process(
        self,
        scan_root: Path,
        output_path: Path,
        recursive: bool = True,
        formats_filter: list[str] | None = None,
    ) -> BatchResult:
        """Full batch pipeline: scan → dedup → process new documents.

        Args:
            scan_root: Directory to scan.
            output_path: Root output directory for all results.
            recursive: Scan subdirectories.
            formats_filter: Limit to specific formats.

        Returns:
            BatchResult with processing statistics.
        """
        t0 = time.perf_counter()

        entries = await self.scan_and_dedup(
            scan_root, recursive, formats_filter,
        )

        processed = 0
        skipped = 0
        errors = 0

        for entry in entries:
            if entry.cache_status in ("exact_match", "content_match"):
                skipped += 1
                logger.debug("Skipping %s (cache: %s)", entry.filename, entry.cache_status)
                continue

            if entry.cache_status == "near_match":
                # Near-match: skip by default (configurable in future)
                skipped += 1
                logger.warning(
                    "Near-match found for %s (matched: %s), skipping",
                    entry.filename, entry.matched_document_id,
                )
                continue

            # Process new document
            try:
                await self._process_entry(entry, output_path)
                processed += 1
            except Exception:
                errors += 1
                logger.exception("Failed to process %s", entry.filename)

        duration = time.perf_counter() - t0

        return BatchResult(
            scan_root=str(scan_root),
            total_files_found=len(entries),
            processed=processed,
            skipped=skipped,
            errors=errors,
            entries=entries,
            duration_seconds=round(duration, 2),
        )

    async def _process_entry(
        self, entry: ScanEntry, output_path: Path,
    ) -> None:
        """Process a single scan entry through the analysis facade."""
        from ayextractor.api.facade import analyze
        from ayextractor.api.models import DocumentInput, Metadata

        file_path = Path(entry.file_path)
        document = DocumentInput(
            content=file_path,
            format=entry.format,
            filename=entry.filename,
        )
        metadata = Metadata(
            document_type="report",
            output_path=output_path,
        )

        logger.info("Processing %s (%s, %d bytes)",
                     entry.filename, entry.format, entry.size_bytes)
        await analyze(document, metadata, settings=self._settings)
