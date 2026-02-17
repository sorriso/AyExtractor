# tests/unit/api/test_facade.py — v1
"""Tests for api.facade — public entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ayextractor.api.facade import (
    _apply_overrides,
    _generate_document_id,
    _generate_run_id,
    _resolve_content_bytes,
    analyze,
)
from ayextractor.api.models import (
    AnalysisResult,
    ConfigOverrides,
    DocumentInput,
    Metadata,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_document(content: str = "test content", fmt: str = "md") -> DocumentInput:
    return DocumentInput(content=content, format=fmt, filename="test.md")


def _make_metadata(**kwargs) -> Metadata:
    defaults = {"document_type": "report", "output_path": Path("/tmp/out")}
    defaults.update(kwargs)
    return Metadata(**defaults)


# ---------------------------------------------------------------------------
# Unit tests — internal helpers
# ---------------------------------------------------------------------------

class TestGenerateDocumentId:
    def test_format(self):
        doc_id = _generate_document_id()
        parts = doc_id.split("_")
        assert len(parts) == 3  # date_time_uuid
        assert len(parts[0]) == 8  # yyyymmdd
        assert len(parts[1]) == 6  # hhmmss
        assert len(parts[2]) == 8  # uuid short

    def test_uniqueness(self):
        ids = {_generate_document_id() for _ in range(50)}
        assert len(ids) == 50


class TestGenerateRunId:
    def test_format(self):
        run_id = _generate_run_id("doc_123")
        parts = run_id.split("_")
        assert len(parts) >= 2
        assert len(parts[0]) == 8  # yyyymmdd

    def test_deterministic_for_same_input_and_time(self):
        # uuid5 is deterministic for the same name, so same doc+minute → same id
        rid1 = _generate_run_id("doc_a")
        rid2 = _generate_run_id("doc_a")
        assert rid1 == rid2


class TestResolveContentBytes:
    def test_bytes_passthrough(self):
        doc = DocumentInput(content=b"hello", format="txt", filename="t.txt")
        assert _resolve_content_bytes(doc) == b"hello"

    def test_str_encoded(self):
        doc = DocumentInput(content="héllo", format="md", filename="t.md")
        assert _resolve_content_bytes(doc) == "héllo".encode("utf-8")

    def test_path_read(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_bytes(b"content")
        doc = DocumentInput(content=f, format="txt", filename="doc.txt")
        assert _resolve_content_bytes(doc) == b"content"

    def test_multi_path(self, tmp_path: Path):
        f1 = tmp_path / "a.png"
        f2 = tmp_path / "b.png"
        f1.write_bytes(b"img1")
        f2.write_bytes(b"img2")
        doc = DocumentInput(content=[f1, f2], format="image", filename="imgs")
        assert _resolve_content_bytes(doc) == b"img1img2"


class TestApplyOverrides:
    def test_no_overrides(self):
        from ayextractor.config.settings import Settings
        s = Settings()
        m = _make_metadata()
        result = _apply_overrides(s, m)
        assert result is s  # Same object, not modified

    def test_with_overrides(self):
        from ayextractor.config.settings import Settings
        s = Settings()
        overrides = ConfigOverrides(chunk_target_size=500)
        m = _make_metadata(config_overrides=overrides)
        result = _apply_overrides(s, m)
        assert result.chunk_target_size == 500


# ---------------------------------------------------------------------------
# Integration test — analyze()
# ---------------------------------------------------------------------------

class TestAnalyze:
    @pytest.mark.asyncio
    async def test_analyze_returns_result(self, tmp_path: Path):
        """Verify analyze() returns AnalysisResult with correct fields."""
        mock_state = MagicMock()
        mock_state.synthesis = "Test summary"
        mock_state.dense_summary = "Dense"
        mock_state.community_summaries = []
        mock_state.agent_outputs = {}
        mock_state.total_llm_calls = 0
        mock_state.graph = None

        mock_pipeline_cls = MagicMock()
        mock_pipeline_inst = AsyncMock()
        mock_pipeline_inst.process.return_value = mock_state
        mock_pipeline_cls.return_value = mock_pipeline_inst

        with patch(
            "ayextractor.pipeline.document_pipeline.DocumentPipeline", mock_pipeline_cls,
        ):
            doc = _make_document()
            meta = _make_metadata(output_path=tmp_path)
            result = await analyze(doc, meta)

        assert isinstance(result, AnalysisResult)
        assert result.summary == "Test summary"
        assert result.community_count == 0

    @pytest.mark.asyncio
    async def test_analyze_with_custom_document_id(self, tmp_path: Path):
        mock_state = MagicMock()
        mock_state.synthesis = ""
        mock_state.dense_summary = ""
        mock_state.community_summaries = []
        mock_state.agent_outputs = {}
        mock_state.total_llm_calls = 0
        mock_state.graph = None

        mock_pipeline_cls = MagicMock()
        mock_pipeline_inst = AsyncMock()
        mock_pipeline_inst.process.return_value = mock_state
        mock_pipeline_cls.return_value = mock_pipeline_inst

        with patch(
            "ayextractor.pipeline.document_pipeline.DocumentPipeline", mock_pipeline_cls,
        ):
            doc = _make_document()
            meta = _make_metadata(
                output_path=tmp_path, document_id="custom_doc_42",
            )
            result = await analyze(doc, meta)

        assert result.document_id == "custom_doc_42"
