# tests/unit/rag/graph_store/test_graph_stores.py — v1
"""Tests for graph store adapters — import error handling and factory."""

from __future__ import annotations

import sys

import pytest


class TestNeo4jStore:
    def test_import_error(self):
        mod = sys.modules.get("neo4j")
        sys.modules["neo4j"] = None  # type: ignore[assignment]
        try:
            from ayextractor.rag.graph_store.neo4j_store import Neo4jStore
            with pytest.raises(ImportError, match="neo4j"):
                Neo4jStore()
        finally:
            if mod is not None:
                sys.modules["neo4j"] = mod
            else:
                sys.modules.pop("neo4j", None)


class TestArangoDBStore:
    def test_import_error(self):
        mod = sys.modules.get("arango")
        sys.modules["arango"] = None  # type: ignore[assignment]
        try:
            from ayextractor.rag.graph_store.arangodb_store import ArangoDBStore
            with pytest.raises(ImportError, match="python-arango"):
                ArangoDBStore()
        finally:
            if mod is not None:
                sys.modules["arango"] = mod
            else:
                sys.modules.pop("arango", None)


class TestGraphStoreFactory:
    def test_none_raises(self):
        from ayextractor.config.settings import Settings
        from ayextractor.rag.graph_store.graph_store_factory import create_graph_store
        s = Settings(_env_file=None, graph_db_type="none")
        with pytest.raises(ValueError, match="none"):
            create_graph_store(s)

    def test_unsupported_type(self):
        """Settings validation rejects invalid types."""
        from ayextractor.config.settings import Settings
        with pytest.raises(Exception):
            Settings(_env_file=None, graph_db_type="invalid_db")
