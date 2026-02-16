# src/chunking/structural_chunker.py — v1
"""Structural chunking strategy based on document structure.

Uses section boundaries, headings, and paragraph breaks to produce chunks.
Respects IMAGE_CONTENT and TABLE_CONTENT block atomicity (spec §7.3).
See spec §4.1 step 2a.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

from ayextractor.chunking.base_chunker import BaseChunker
from ayextractor.config.settings import Settings
from ayextractor.core.models import Chunk, ChunkSourceSection, DocumentStructure, Section

# Pattern to match atomic blocks
_ATOMIC_BLOCK = re.compile(
    r"<<<(?:IMAGE_CONTENT|TABLE_CONTENT)\b.*?<<<END_(?:IMAGE_CONTENT|TABLE_CONTENT)>>>",
    re.DOTALL,
)

# Pattern to extract block IDs
_BLOCK_ID = re.compile(r'id="([^"]+)"')
_BLOCK_TYPE = re.compile(r"<<<(IMAGE_CONTENT|TABLE_CONTENT)")


class StructuralChunker(BaseChunker):
    """Chunk text using structural boundaries."""

    def __init__(self, settings: Settings | None = None):
        self._target = 2000 if settings is None else settings.chunk_target_size
        self._overlap = 0 if settings is None else settings.chunk_overlap

    @property
    def strategy_name(self) -> str:
        return "structural"

    async def chunk(
        self,
        text: str,
        structure: DocumentStructure | None = None,
        source_file: str = "unknown",
    ) -> list[Chunk]:
        """Split text into chunks respecting structure and atomic blocks."""
        if not text.strip():
            return []

        # 1. Split into segments respecting atomic blocks
        segments = self._split_preserving_atomic(text)

        # 2. Merge segments into chunks of ~target size
        raw_chunks = self._merge_segments(segments)

        # 3. Build Chunk objects
        chunks: list[Chunk] = []
        offset = 0
        for i, content in enumerate(raw_chunks):
            chunk_id = f"chunk_{i:03d}"
            sections = self._find_sections(content, structure) if structure else []

            # Detect embedded images and tables
            images = [m.group(1) for m in _BLOCK_ID.finditer(content) if "IMAGE" in content[max(0, m.start()-50):m.start()]]
            tables = [m.group(1) for m in _BLOCK_ID.finditer(content) if "TABLE" in content[max(0, m.start()-50):m.start()]]

            # Simpler approach: scan for all IDs
            embedded_images = []
            embedded_tables = []
            for block_match in _ATOMIC_BLOCK.finditer(content):
                block_text = block_match.group()
                id_match = _BLOCK_ID.search(block_text)
                type_match = _BLOCK_TYPE.search(block_text)
                if id_match and type_match:
                    bid = id_match.group(1)
                    if type_match.group(1) == "IMAGE_CONTENT":
                        embedded_images.append(bid)
                    else:
                        embedded_tables.append(bid)

            content_type = "mixed" if embedded_images or embedded_tables else "text"
            fp = hashlib.sha256(content.encode()).hexdigest()[:16]
            end_offset = offset + len(content.encode())

            chunk = Chunk(
                id=chunk_id,
                position=i,
                content=content,
                content_type=content_type,
                embedded_images=embedded_images,
                embedded_tables=embedded_tables,
                source_file=source_file,
                source_sections=sections,
                byte_offset_start=offset,
                byte_offset_end=end_offset,
                char_count=len(content),
                word_count=len(content.split()),
                token_count_est=len(content) // 4,
                fingerprint=fp,
            )
            chunks.append(chunk)
            offset = end_offset

        # Link chunks
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.preceding_chunk_id = chunks[i - 1].id
            if i < len(chunks) - 1:
                chunk.following_chunk_id = chunks[i + 1].id

        return chunks

    def _split_preserving_atomic(self, text: str) -> list[str]:
        """Split text into segments, keeping atomic blocks intact."""
        segments: list[str] = []
        last_end = 0

        for match in _ATOMIC_BLOCK.finditer(text):
            # Add text before block (split by paragraphs)
            before = text[last_end:match.start()]
            if before.strip():
                for para in re.split(r"\n{2,}", before):
                    if para.strip():
                        segments.append(para.strip())
            # Add atomic block as single segment
            segments.append(match.group())
            last_end = match.end()

        # Remaining text after last block
        remaining = text[last_end:]
        if remaining.strip():
            for para in re.split(r"\n{2,}", remaining):
                if para.strip():
                    segments.append(para.strip())

        return segments

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """Merge segments into chunks of approximately target size."""
        if not segments:
            return []

        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for seg in segments:
            seg_len = len(seg)
            if current_len + seg_len > self._target and current_parts:
                chunks.append("\n\n".join(current_parts))
                # Overlap: keep last part if overlap > 0
                if self._overlap > 0 and current_parts:
                    last = current_parts[-1]
                    current_parts = [last] if len(last) <= self._overlap else []
                    current_len = len(last) if current_parts else 0
                else:
                    current_parts = []
                    current_len = 0
            current_parts.append(seg)
            current_len += seg_len

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    def _find_sections(
        self, content: str, structure: DocumentStructure,
    ) -> list[ChunkSourceSection]:
        """Find which sections a chunk belongs to."""
        result: list[ChunkSourceSection] = []
        content_lower = content[:200].lower()
        for section in structure.sections:
            if section.title.lower() in content_lower:
                result.append(ChunkSourceSection(title=section.title, level=section.level))
        return result[:3]  # Max 3 sections per chunk
