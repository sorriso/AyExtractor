# src/extraction/structure_detector.py — v2
"""Detect document structure: TOC, sections, annexes, bibliography, footnotes, index.

Operates on extracted raw text + optional format-specific metadata.
See spec §10 for detection methods and output.
"""

from __future__ import annotations

import re

from ayextractor.core.models import DocumentStructure, Footnote, Section


def detect_structure(text: str) -> DocumentStructure:
    """Analyze text to detect structural elements.

    Args:
        text: Raw extracted text from document.

    Returns:
        Populated DocumentStructure with detected elements.
    """
    sections = _detect_sections(text)
    footnotes = _detect_footnotes(text)
    has_toc = _detect_toc(text)
    has_bibliography, bib_pos = _detect_bibliography(text)
    has_annexes, annexes = _detect_annexes(text, sections)
    has_index = _detect_index(text)

    return DocumentStructure(
        has_toc=has_toc,
        sections=sections,
        has_bibliography=has_bibliography,
        bibliography_position=bib_pos,
        has_annexes=has_annexes,
        annexes=annexes,
        footnotes=footnotes,
        has_index=has_index,
    )


def _detect_sections(text: str) -> list[Section]:
    """Detect headings and sections via numbering and formatting patterns."""
    sections: list[Section] = []

    # Pattern: numbered headings like "1.", "1.1", "1.1.1", "Chapter 1"
    heading_patterns = [
        # Markdown headings: "# Title", "## Subtitle", etc.
        (r"^(#{1,6})\s+(.+)$", lambda m: len(m.group(1))),
        # "1. Title" or "1.2.3 Title"
        (r"^(\d+(?:\.\d+)*)\s+(.+)$", lambda m: m.group(1).count(".") + 1),
        # "Chapter N" / "Chapitre N"
        (r"^(?:chapter|chapitre)\s+(\d+)\s*[:\.\-—]?\s*(.*)$", lambda _: 1),
        # "SECTION N" / "ANNEX" style uppercase headings
        (r"^([A-Z][A-Z\s]{3,})$", lambda _: 1),
    ]

    for i, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if not stripped or len(stripped) > 200:
            continue
        for pattern, level_fn in heading_patterns:
            match = re.match(pattern, stripped, re.IGNORECASE)
            if match:
                title = match.group(0).strip()
                level = level_fn(match)
                sections.append(
                    Section(
                        title=title,
                        level=level,
                        start_position=text.find(line),
                        end_position=text.find(line) + len(line),
                    )
                )
                break

    return sections


def _detect_toc(text: str) -> bool:
    """Detect presence of a table of contents."""
    toc_markers = [
        r"table\s+of\s+contents",
        r"table\s+des\s+matières",
        r"sommaire",
        r"contents\s*$",
    ]
    sample = text[:5000].lower()
    return any(re.search(p, sample) for p in toc_markers)


def _detect_bibliography(text: str) -> tuple[bool, int | None]:
    """Detect bibliography section. Returns (has_bibliography, position)."""
    bib_heading = re.compile(
        r"^#{1,3}\s+(?:references|bibliography|bibliographie|références)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    # Check for a markdown heading anywhere in the document
    match = bib_heading.search(text)
    if match:
        return True, match.start()

    # Fallback: search in the last 30% of the document for non-heading markers
    bib_markers = [
        r"\b(?:references|bibliography|bibliographie|références)\b",
    ]
    search_start = int(len(text) * 0.7)
    tail = text[search_start:].lower()

    for pattern in bib_markers:
        match = re.search(pattern, tail)
        if match:
            return True, search_start + match.start()
    return False, None


def _detect_annexes(
    text: str, sections: list[Section]
) -> tuple[bool, list[Section]]:
    """Detect annex/appendix sections."""
    annex_pattern = re.compile(
        r"\b(?:annex|appendix|annexe)\b", re.IGNORECASE
    )
    annexes = [s for s in sections if annex_pattern.search(s.title)]
    return len(annexes) > 0, annexes


def _detect_footnotes(text: str) -> list[Footnote]:
    """Detect footnotes from superscript markers and footnote sections."""
    footnotes: list[Footnote] = []

    # Pattern: [1] or ¹ followed by content at end/bottom
    fn_pattern = re.compile(r"^\[(\d+)\]\s+(.+)$", re.MULTILINE)
    for match in fn_pattern.finditer(text):
        footnotes.append(
            Footnote(
                id=f"fn_{match.group(1)}",
                content=match.group(2).strip(),
                position=match.start(),
            )
        )

    return footnotes


def _detect_index(text: str) -> bool:
    """Detect alphabetical index section."""
    # Index is usually at the very end with alphabetical entries + page numbers
    tail = text[-3000:].lower() if len(text) > 3000 else text.lower()
    index_markers = [r"\bindex\b", r"\bindice\b"]
    if not any(re.search(p, tail) for p in index_markers):
        return False

    # Check for alphabetical pattern (A\n... B\n... C\n...)
    alpha_pattern = re.compile(r"^[A-Z]\s*$", re.MULTILINE)
    matches = alpha_pattern.findall(tail[-2000:])
    return len(matches) >= 3
