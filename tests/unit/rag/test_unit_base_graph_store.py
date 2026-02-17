# tests/unit/rag/test_base_graph_store.py — v1
"""Tests for rag/graph_store/base_graph_store.py — BaseGraphStore ABC."""

from __future__ import annotations

import pytest

from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore


class TestBaseGraphStore:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseGraphStore()  # type: ignore[abstract]

    def test_has_required_methods(self):
        for attr in [
            "upsert_node", "get_node", "delete_node",
            "upsert_edge", "get_edges", "delete_edges",
            "query_neighbors", "query_by_properties",
            "import_graph", "node_count", "edge_count", "provider_name",
        ]:
            assert hasattr(BaseGraphStore, attr)
