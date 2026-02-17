# tests/unit/rag/vector_store/test_unit_vector_stores.py — v2
"""Tests for vector store adapters — properties and import error handling.

Changelog:
    v2: Catch RuntimeError for chromadb http-only client mode (devcontainer
        installs chromadb as http-only). Use pytest.skip instead of failing.
        Renamed from test_vector_stores.py per naming convention.
    v1: Initial tests.
"""

from __future__ import annotations

import sys

import pytest


class TestChromaDBStore:
    def test_provider_name(self):
        """ChromaDB in-memory client for test."""
        try:
            from ayextractor.rag.vector_store.chromadb_store import ChromaDBStore
            store = ChromaDBStore()  # In-memory
            assert store.provider_name == "chromadb"
        except ImportError:
            pytest.skip("chromadb not installed")
        except RuntimeError as exc:
            if "http-only" in str(exc).lower():
                pytest.skip("chromadb installed as http-only client")
            raise

    def test_import_error(self):
        mod = sys.modules.get("chromadb")
        sys.modules["chromadb"] = None  # type: ignore[assignment]
        try:
            from ayextractor.rag.vector_store.chromadb_store import ChromaDBStore
            with pytest.raises(ImportError, match="chromadb"):
                ChromaDBStore()
        finally:
            if mod is not None:
                sys.modules["chromadb"] = mod
            else:
                sys.modules.pop("chromadb", None)


class TestQdrantStore:
    def test_import_error(self):
        mod = sys.modules.get("qdrant_client")
        sys.modules["qdrant_client"] = None  # type: ignore[assignment]
        try:
            from ayextractor.rag.vector_store.qdrant_store import QdrantStore
            with pytest.raises(ImportError, match="qdrant-client"):
                QdrantStore()
        finally:
            if mod is not None:
                sys.modules["qdrant_client"] = mod
            else:
                sys.modules.pop("qdrant_client", None)


class TestVectorStoreFactory:
    def test_none_raises(self):
        from ayextractor.config.settings import Settings
        from ayextractor.rag.vector_store.vector_store_factory import create_vector_store
        s = Settings(_env_file=None, vector_db_type="none")
        with pytest.raises(ValueError, match="none"):
            create_vector_store(s)

    def test_unsupported_type(self):
        """Settings validation rejects invalid types."""
        from ayextractor.config.settings import Settings
        with pytest.raises(Exception):
            Settings(_env_file=None, vector_db_type="invalid_db")

    def test_chromadb_local(self, tmp_path):
        """ChromaDB with local persist path."""
        try:
            import chromadb  # noqa: F401
        except ImportError:
            pytest.skip("chromadb not installed")
        try:
            from ayextractor.config.settings import Settings
            from ayextractor.rag.vector_store.vector_store_factory import create_vector_store
            s = Settings(_env_file=None, vector_db_type="chromadb", vector_db_path=tmp_path / "chroma")
            store = create_vector_store(s)
            assert store.provider_name == "chromadb"
        except RuntimeError as exc:
            if "http-only" in str(exc).lower():
                pytest.skip("chromadb installed as http-only client")
            raise