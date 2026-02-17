# tests/integration/rag/test_int_store_factories.py — v2
"""Integration tests for store factory functions with real containers.

Coverage targets:
- vector_store_factory.py 31% → 90%+
- graph_store_factory.py 48% → 90%+

Changelog:
    v2: Fix test_qdrant_factory_creates_functional_store — replace deprecated
        asyncio.get_event_loop().run_until_complete() with @pytest.mark.asyncio
        (Python 3.12 no longer has a default event loop in MainThread).
    v1: Initial tests.
"""

from __future__ import annotations

import pytest

from ayextractor.config.settings import Settings


class TestVectorStoreFactory:
    """Factory creating real store instances from settings."""

    @pytest.mark.qdrant
    def test_create_qdrant_store(self, qdrant_url):
        from ayextractor.rag.vector_store.vector_store_factory import create_vector_store

        settings = Settings(
            vector_db_type="qdrant",
            vector_db_url=qdrant_url,
            _env_file=None,
        )
        store = create_vector_store(settings)
        assert store.provider_name == "qdrant"

    @pytest.mark.chromadb
    def test_create_chromadb_store(self, chromadb_config):
        from ayextractor.rag.vector_store.vector_store_factory import create_vector_store

        host = chromadb_config["host"]
        port = chromadb_config["port"]
        settings = Settings(
            vector_db_type="chromadb",
            vector_db_url=f"http://{host}:{port}",
            _env_file=None,
        )
        store = create_vector_store(settings)
        assert store.provider_name == "chromadb"

    def test_create_none_raises(self):
        from ayextractor.rag.vector_store.vector_store_factory import create_vector_store

        settings = Settings(vector_db_type="none", _env_file=None)
        with pytest.raises(ValueError, match="none"):
            create_vector_store(settings)

    def test_create_unsupported_raises(self):
        from ayextractor.rag.vector_store.vector_store_factory import (
            create_vector_store,
            UnsupportedVectorStoreError,
        )

        # Force an invalid type by overriding after init
        settings = Settings(vector_db_type="qdrant", _env_file=None)
        object.__setattr__(settings, "vector_db_type", "weaviate")
        with pytest.raises(UnsupportedVectorStoreError):
            create_vector_store(settings)

    @pytest.mark.qdrant
    @pytest.mark.asyncio
    async def test_qdrant_factory_creates_functional_store(self, qdrant_url, qdrant_collection):
        """Verify factory-created store is fully functional."""
        from ayextractor.rag.vector_store.vector_store_factory import create_vector_store

        settings = Settings(
            vector_db_type="qdrant", vector_db_url=qdrant_url, _env_file=None,
        )
        store = create_vector_store(settings)

        await store.create_collection(qdrant_collection, dimensions=64)
        assert await store.collection_exists(qdrant_collection)
        assert await store.count(qdrant_collection) == 0


class TestGraphStoreFactory:
    """Factory creating real graph store instances."""

    @pytest.mark.arangodb
    def test_create_arangodb_store(self, arangodb_url):
        from ayextractor.rag.graph_store.graph_store_factory import create_graph_store

        settings = Settings(
            graph_db_type="arangodb",
            graph_db_uri=arangodb_url,
            graph_db_database="_system",
            graph_db_user="root",
            graph_db_password="testpassword",
            _env_file=None,
        )
        store = create_graph_store(settings)
        assert store.provider_name == "arangodb"

    def test_create_none_raises(self):
        from ayextractor.rag.graph_store.graph_store_factory import create_graph_store

        settings = Settings(graph_db_type="none", _env_file=None)
        with pytest.raises(ValueError, match="none"):
            create_graph_store(settings)

    def test_create_unsupported_raises(self):
        from ayextractor.rag.graph_store.graph_store_factory import (
            create_graph_store,
            UnsupportedGraphStoreError,
        )

        settings = Settings(graph_db_type="arangodb", _env_file=None)
        object.__setattr__(settings, "graph_db_type", "dgraph")
        with pytest.raises(UnsupportedGraphStoreError):
            create_graph_store(settings)