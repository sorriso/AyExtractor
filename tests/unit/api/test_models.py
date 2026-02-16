# tests/unit/api/test_models.py — v1
"""Tests for api/models.py — API-level input/output models."""

from __future__ import annotations

from pathlib import Path

import pytest

from ayextractor.api.models import (
    AnalysisResult,
    ConfigOverrides,
    DocumentInput,
    Metadata,
)


class TestDocumentInput:
    def test_create_with_string(self):
        d = DocumentInput(content="text content", format="md", filename="doc.md")
        assert d.format == "md"

    def test_create_with_path(self):
        d = DocumentInput(content=Path("/tmp/file.pdf"), format="pdf", filename="file.pdf")
        assert isinstance(d.content, Path)

    def test_create_with_bytes(self):
        d = DocumentInput(content=b"binary", format="pdf", filename="file.pdf")
        assert isinstance(d.content, bytes)


class TestMetadata:
    def test_defaults(self):
        m = Metadata()
        assert m.document_id is None
        assert m.document_type == "report"
        assert m.language is None

    def test_with_overrides(self):
        overrides = ConfigOverrides(chunk_target_size=4000, density_iterations=3)
        m = Metadata(config_overrides=overrides)
        assert m.config_overrides.chunk_target_size == 4000


class TestConfigOverrides:
    def test_all_none_by_default(self):
        co = ConfigOverrides()
        assert co.llm_assignments is None
        assert co.chunking_strategy is None

    def test_partial_override(self):
        co = ConfigOverrides(
            chunking_strategy="semantic",
            entity_similarity_threshold=0.9,
        )
        assert co.chunking_strategy == "semantic"
        assert co.density_iterations is None


class TestAnalysisResult:
    def test_create_minimal(self):
        ar = AnalysisResult(
            document_id="20260207_140000_abc",
            run_id="20260207_1615_xyz",
            summary="Test summary",
            graph_path=Path("/out/graph.json"),
            communities_path=Path("/out/communities.json"),
            profiles_path=Path("/out/profiles.json"),
            output_dir=Path("/out"),
            run_dir=Path("/out/runs/r1"),
        )
        assert ar.community_count == 0
        assert ar.themes == []
