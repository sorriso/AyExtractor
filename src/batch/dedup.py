# src/batch/dedup.py — v1
"""Batch deduplicator — fingerprint comparison against cache.

Checks scan entries against existing cache to identify duplicates
at multiple levels: exact match, content match, near match.

See spec §24.3 for the dedup decision flow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ayextractor.batch.models import ScanEntry
from ayextractor.cache.fingerprint import compute_fingerprint

if TYPE_CHECKING:
    from ayextractor.cache.base_cache_store import BaseCacheStore

logger = logging.getLogger(__name__)


class BatchDeduplicator:
    """Compare scan entries against the cache for deduplication.

    Decision flow per file:
      1. Compute Level 1 (exact_hash) and Level 2 (content_hash)
      2. Lookup in cache index:
         - EXACT MATCH → skip
         - CONTENT MATCH → skip (different file, same content)
         - NEAR MATCH (simhash) → warn + skip (configurable)
         - NO MATCH → queue for processing
    """

    def __init__(self, cache_store: BaseCacheStore) -> None:
        self._cache_store = cache_store

    async def check_entries(
        self, entries: list[ScanEntry],
    ) -> list[ScanEntry]:
        """Compute fingerprints and check cache for all entries.

        Modifies entries in-place (updates fingerprint_exact,
        fingerprint_content, cache_status, matched_document_id).

        Returns:
            The same list with cache_status populated.
        """
        # Pre-load all cache entries for batch comparison
        cached = await self._cache_store.list_entries()
        exact_index: dict[str, str] = {}
        content_index: dict[str, str] = {}

        for ce in cached:
            if ce.fingerprint and ce.fingerprint.exact_hash:
                exact_index[ce.fingerprint.exact_hash] = ce.document_id
            if ce.fingerprint and ce.fingerprint.content_hash:
                content_index[ce.fingerprint.content_hash] = ce.document_id

        for entry in entries:
            try:
                self._compute_and_check(entry, exact_index, content_index)
            except Exception:
                logger.warning(
                    "Fingerprint computation failed for %s, marking as no_match",
                    entry.filename, exc_info=True,
                )
                entry.cache_status = "no_match"

        matched = sum(
            1 for e in entries if e.cache_status != "no_match"
        )
        logger.info(
            "Dedup complete: %d/%d entries matched cache",
            matched, len(entries),
        )
        return entries

    def _compute_and_check(
        self,
        entry: ScanEntry,
        exact_index: dict[str, str],
        content_index: dict[str, str],
    ) -> None:
        """Compute fingerprint for one entry and check indexes."""
        file_path = Path(entry.file_path)
        raw_bytes = file_path.read_bytes()

        # Compute fingerprint (Level 1 + Level 2 at minimum)
        fp = compute_fingerprint(
            raw_bytes=raw_bytes,
            extracted_text="",  # Text extraction not yet done; use byte-level
            source_format=entry.format,
        )

        entry.fingerprint_exact = fp.exact_hash
        entry.fingerprint_content = fp.content_hash

        # Check Level 1: exact byte match
        if fp.exact_hash in exact_index:
            entry.cache_status = "exact_match"
            entry.matched_document_id = exact_index[fp.exact_hash]
            logger.debug(
                "Exact match: %s → %s", entry.filename, entry.matched_document_id,
            )
            return

        # Check Level 2: content hash match
        if fp.content_hash and fp.content_hash in content_index:
            entry.cache_status = "content_match"
            entry.matched_document_id = content_index[fp.content_hash]
            logger.debug(
                "Content match: %s → %s", entry.filename, entry.matched_document_id,
            )
            return

        # Check Level 3: structural similarity (simhash near-match)
        if fp.structural_hash:
            near_match = self._check_near_match(
                fp.structural_hash, exact_index, content_index,
            )
            if near_match:
                entry.cache_status = "near_match"
                entry.matched_document_id = near_match
                logger.debug(
                    "Near match: %s → %s", entry.filename, near_match,
                )
                return

        entry.cache_status = "no_match"

    def _check_near_match(
        self,
        structural_hash: str,
        exact_index: dict[str, str],
        content_index: dict[str, str],
    ) -> str | None:
        """Check simhash-based near match (Hamming distance).

        Returns matched document_id or None.
        Note: full simhash comparison requires the cached structural
        hashes, which are not indexed yet. Placeholder for future
        implementation.
        """
        # TODO: Implement Hamming distance comparison against cached
        #       structural hashes when Level 3 dedup is enabled.
        return None


async def check_single(
    file_path: Path,
    cache_store: BaseCacheStore,
) -> ScanEntry:
    """Convenience: check a single file against cache.

    Returns a ScanEntry with cache_status populated.
    """
    fmt_map = {
        ".pdf": "pdf", ".epub": "epub", ".docx": "docx",
        ".md": "md", ".txt": "txt",
        ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    }
    fmt = fmt_map.get(file_path.suffix.lower(), "unknown")

    entry = ScanEntry(
        file_path=str(file_path.resolve()),
        filename=file_path.name,
        format=fmt,
        size_bytes=file_path.stat().st_size,
        fingerprint_exact="",
        fingerprint_content="",
        cache_status="no_match",
    )

    dedup = BatchDeduplicator(cache_store=cache_store)
    results = await dedup.check_entries([entry])
    return results[0]
