# src/cache/sqlite_store.py — v1
"""SQLite-based cache store (CACHE_BACKEND=sqlite).

Uses stdlib sqlite3 — no external dependency.
Better performance than JSON for large numbers of documents.
See spec §8.3 for cache lookup logic.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from ayextractor.cache.base_cache_store import BaseCacheStore
from ayextractor.cache.fingerprint import hamming_distance
from ayextractor.cache.models import CacheEntry, CacheLookupResult, DocumentFingerprint

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    exact_hash TEXT,
    content_hash TEXT,
    structural_hash TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_exact_hash ON cache_entries(exact_hash);
CREATE INDEX IF NOT EXISTS idx_content_hash ON cache_entries(content_hash);
"""


class SqliteCacheStore(BaseCacheStore):
    """SQLite-backed cache store for better performance at scale."""

    def __init__(
        self, db_path: Path | str, simhash_threshold: int = 3
    ) -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._simhash_threshold = simhash_threshold
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    async def get(self, key: str) -> CacheEntry | None:
        """Retrieve cache entry by key."""
        cursor = self._conn.execute(
            "SELECT data FROM cache_entries WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        try:
            return CacheEntry(**json.loads(row[0]))
        except Exception as e:
            logger.warning("Failed to deserialize cache entry %s: %s", key, e)
            return None

    async def put(self, key: str, entry: CacheEntry) -> None:
        """Store a cache entry (upsert)."""
        fp = entry.fingerprint
        self._conn.execute(
            """INSERT OR REPLACE INTO cache_entries
               (key, data, exact_hash, content_hash, structural_hash)
               VALUES (?, ?, ?, ?, ?)""",
            (
                key,
                entry.model_dump_json(),
                fp.exact_hash,
                fp.content_hash,
                fp.structural_hash,
            ),
        )
        self._conn.commit()

    async def delete(self, key: str) -> None:
        """Remove a cache entry."""
        self._conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        self._conn.commit()

    async def lookup_fingerprint(
        self, fingerprint: DocumentFingerprint
    ) -> CacheLookupResult:
        """Multi-level fingerprint lookup (spec §8.3).

        Level 1: exact hash match (indexed).
        Level 2: content hash match (indexed).
        Level 3: structural hash via SimHash Hamming distance (scan).
        """
        # Level 1: exact hash
        cursor = self._conn.execute(
            "SELECT data FROM cache_entries WHERE exact_hash = ?",
            (fingerprint.exact_hash,),
        )
        row = cursor.fetchone()
        if row:
            entry = CacheEntry(**json.loads(row[0]))
            return CacheLookupResult(
                hit_level="exact",
                matched_entry=entry,
                similarity_score=1.0,
                is_reusable=True,
            )

        # Level 2: content hash
        cursor = self._conn.execute(
            "SELECT data FROM cache_entries WHERE content_hash = ?",
            (fingerprint.content_hash,),
        )
        row = cursor.fetchone()
        if row:
            entry = CacheEntry(**json.loads(row[0]))
            return CacheLookupResult(
                hit_level="content",
                matched_entry=entry,
                similarity_score=1.0,
                is_reusable=True,
            )

        # Level 3: structural hash (requires scan)
        if fingerprint.structural_hash:
            cursor = self._conn.execute(
                "SELECT data, structural_hash FROM cache_entries WHERE structural_hash IS NOT NULL"
            )
            for row in cursor.fetchall():
                distance = hamming_distance(row[1], fingerprint.structural_hash)
                if distance <= self._simhash_threshold:
                    entry = CacheEntry(**json.loads(row[0]))
                    return CacheLookupResult(
                        hit_level="structural",
                        matched_entry=entry,
                        similarity_score=1.0 - (distance / 64.0),
                        is_reusable=False,
                    )

        return CacheLookupResult()

    async def list_entries(self) -> list[CacheEntry]:
        """List all cached entries."""
        cursor = self._conn.execute("SELECT data FROM cache_entries")
        entries: list[CacheEntry] = []
        for row in cursor.fetchall():
            try:
                entries.append(CacheEntry(**json.loads(row[0])))
            except Exception:
                continue
        return entries

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
