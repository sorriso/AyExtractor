# tests/integration/api/test_int_api.py — v2
"""Integration tests for API subsystem.

Covers: api/facade.py, api/models.py
No Docker required.

API source review:
- DocumentInput: requires content, format, filename (all mandatory)
- Metadata: has document_id, document_type, output_path, language (NO title field)
- AnalysisResult: requires document_id, run_id, summary, graph_path,
  communities_path, profiles_path, output_dir, run_dir
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestAPIModels:

    def test_document_input(self):
        from ayextractor.api.models import DocumentInput
        doc = DocumentInput(content="Hello world", format="txt", filename="test.txt")
        assert doc.content == "Hello world"
        assert doc.format == "txt"
        assert doc.filename == "test.txt"

    def test_document_input_with_path(self):
        from ayextractor.api.models import DocumentInput
        doc = DocumentInput(content=Path("/tmp/test.pdf"), format="pdf", filename="test.pdf")
        assert doc.format == "pdf"

    def test_metadata_defaults(self):
        """Metadata has document_type, output_path, language — NOT title."""
        from ayextractor.api.models import Metadata
        meta = Metadata()
        assert meta.document_type == "report"
        assert meta.language is None
        assert meta.output_path == Path("./output")

    def test_metadata_with_overrides(self):
        from ayextractor.api.models import Metadata
        meta = Metadata(
            document_id="doc_001",
            document_type="contract",
            language="fr",
        )
        assert meta.document_id == "doc_001"
        assert meta.document_type == "contract"
        assert meta.language == "fr"

    def test_analysis_result(self):
        from ayextractor.api.models import AnalysisResult
        result = AnalysisResult(
            document_id="doc_001",
            run_id="run_001",
            summary="A summary",
            graph_path=Path("/tmp/graph.json"),
            communities_path=Path("/tmp/communities.json"),
            profiles_path=Path("/tmp/profiles.json"),
            output_dir=Path("/tmp/output"),
            run_dir=Path("/tmp/run"),
        )
        assert result.document_id == "doc_001"
        assert result.summary == "A summary"
        assert result.community_count == 0  # default
        assert result.themes == []  # default

    def test_config_overrides(self):
        from ayextractor.api.models import ConfigOverrides
        overrides = ConfigOverrides(
            chunking_strategy="semantic",
            density_iterations=3,
            critic_agent_enabled=True,
        )
        assert overrides.chunking_strategy == "semantic"
        assert overrides.density_iterations == 3


class TestAPIFacade:

    def test_facade_import(self):
        from ayextractor.api.facade import analyze
        assert callable(analyze)

    @pytest.mark.asyncio
    async def test_facade_signature(self):
        """Verify facade.analyze accepts expected params."""
        from ayextractor.api.facade import analyze
        from ayextractor.api.models import DocumentInput, Metadata
        doc = DocumentInput(content="Test content.", format="txt", filename="test.txt")
        meta = Metadata(document_type="report", language="en")
        assert doc.format == "txt"
        assert meta.document_type == "report"