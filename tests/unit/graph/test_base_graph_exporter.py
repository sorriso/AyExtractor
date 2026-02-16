# tests/unit/graph/test_base_graph_exporter.py — v1
"""Tests for graph/base_graph_exporter.py — BaseGraphExporter ABC."""

from __future__ import annotations

import pytest

from ayextractor.graph.base_graph_exporter import BaseGraphExporter


class TestBaseGraphExporter:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseGraphExporter()  # type: ignore[abstract]

    def test_has_required_methods(self):
        for attr in ["format_name", "file_extension", "export"]:
            assert hasattr(BaseGraphExporter, attr)
