# tests/unit/extraction/test_epub_extractor.py — v1
"""Tests for extraction/epub_extractor.py."""

from __future__ import annotations

import pytest

from ayextractor.extraction.epub_extractor import EpubExtractor


class TestEpubExtractor:
    def test_supported_extensions(self):
        ext = EpubExtractor()
        assert ".epub" in ext.supported_extensions

    def test_requires_vision(self):
        ext = EpubExtractor()
        assert ext.requires_vision is False

    def test_import_error_message(self):
        """When ebooklib is not installed, a clear ImportError is raised."""
        import sys
        # Temporarily hide ebooklib if present
        ebooklib_mod = sys.modules.get("ebooklib")
        epub_mod = sys.modules.get("ebooklib.epub")
        sys.modules["ebooklib"] = None  # type: ignore[assignment]
        sys.modules["ebooklib.epub"] = None  # type: ignore[assignment]
        try:
            ext = EpubExtractor()
            with pytest.raises(ImportError, match="ebooklib"):
                import asyncio
                asyncio.get_event_loop().run_until_complete(ext.extract(b"fake"))
        finally:
            # Restore
            if ebooklib_mod is not None:
                sys.modules["ebooklib"] = ebooklib_mod
            else:
                sys.modules.pop("ebooklib", None)
            if epub_mod is not None:
                sys.modules["ebooklib.epub"] = epub_mod
            else:
                sys.modules.pop("ebooklib.epub", None)
