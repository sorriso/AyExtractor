# tests/integration/rag/test_int_stores_memory.py — v3
"""Integration tests for vector stores using in-memory backends.

Tests Qdrant (:memory:) and ChromaDB (in-process) without Docker.
Covers: qdrant_store.py, chromadb_store.py, base_vector_store.py,
        vector_store_factory.py

Requires:
    - pip install qdrant-client chromadb (full package, not chromadb-client)

Source API reference (verified against deployed src/):
    - QdrantStore(url=None, api_key=None, path=None) — :memory: if all None
    - ChromaDBStore(persist_path=None, host=None, port=8000) — in-process if all None
    - async query(collection, query_embedding, top_k, filter) → list[SearchResult]
    - vector_store_factory.create_vector_store(settings) → BaseVectorStore
    - Settings.vector_db_type: Literal["none", "chromadb", "qdrant", "arangodb"]
    - SearchResult.source_type: Literal["chunk","entity_profile","relation_profile","community_summary"]

Changelog:
    v3: Fix missing Settings import (NameError in factory tests).
        Fix SearchResult source_type: "entity" → "entity_profile" (Literal).
    v2: ChromaDB http-only guard, Settings Literal, Qdrant v2.
"""

from __future__ import annotations

import math
import random
import uuid

import pytest

from ayextractor.config.settings import Settings

DIMS = 128


def _rand_vec(dims: int = DIMS, seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _similar_vec(base: list[float], noise: float = 0.1) -> list[float]:
    rng = random.Random(42)
    p = [x + rng.gauss(0, noise) for x in base]
    norm = math.sqrt(sum(x * x for x in p))
    return [x / norm for x in p]


# =====================================================================
#  QDRANT IN-MEMORY
# =====================================================================

class TestQdrantMemoryStore:
    """Full CRUD tests for Qdrant in-memory mode."""

    @pytest.mark.asyncio
    async def test_create_collection(self, qdrant_memory_store, unique_collection):
        store = qdrant_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        assert await store.collection_exists(unique_collection) is True

    @pytest.mark.asyncio
    async def test_create_idempotent(self, qdrant_memory_store, unique_collection):
        store = qdrant_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        await store.create_collection(unique_collection, dimensions=DIMS)
        assert await store.collection_exists(unique_collection) is True

    @pytest.mark.asyncio
    async def test_nonexistent_collection(self, qdrant_memory_store):
        assert await qdrant_memory_store.collection_exists("no_such") is False

    @pytest.mark.asyncio
    async def test_count_empty(self, qdrant_memory_store, unique_collection):
        store = qdrant_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        assert await store.count(unique_collection) == 0

    @pytest.mark.asyncio
    async def test_upsert_and_count(self, qdrant_memory_store, unique_collection):
        store = qdrant_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        await store.upsert(
            collection=unique_collection,
            ids=["a", "b", "c"],
            embeddings=[_rand_vec(seed=i) for i in range(3)],
            documents=["doc a", "doc b", "doc c"],
            metadatas=[{"source_type": "chunk"}] * 3,
        )
        assert await store.count(unique_collection) == 3

    @pytest.mark.asyncio
    async def test_similarity_order(self, qdrant_memory_store, unique_collection):
        store = qdrant_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        base = _rand_vec(seed=100)
        near = _similar_vec(base, noise=0.05)
        far = _rand_vec(seed=999)
        await store.upsert(
            collection=unique_collection,
            ids=["near", "far"],
            embeddings=[near, far],
            documents=["near doc", "far doc"],
        )
        results = await store.query(unique_collection, base, top_k=2)
        assert len(results) == 2
        assert results[0].content == "near doc"

    @pytest.mark.asyncio
    async def test_top_k_limit(self, qdrant_memory_store, unique_collection):
        store = qdrant_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        await store.upsert(
            collection=unique_collection,
            ids=[f"id_{i}" for i in range(10)],
            embeddings=[_rand_vec(seed=i) for i in range(10)],
            documents=[f"doc {i}" for i in range(10)],
        )
        results = await store.query(unique_collection, _rand_vec(seed=0), top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_metadata_filter(self, qdrant_memory_store, unique_collection):
        """Filter by metadata field.

        Uses valid Literal source_type: "entity_profile" not "entity".
        """
        store = qdrant_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        await store.upsert(
            collection=unique_collection,
            ids=["e1", "e2"],
            embeddings=[_rand_vec(seed=1), _rand_vec(seed=2)],
            documents=["entity one", "entity two"],
            metadatas=[{"source_type": "entity_profile"}, {"source_type": "chunk"}],
        )
        results = await store.query(
            unique_collection, _rand_vec(seed=1), top_k=10,
            filter={"source_type": "entity_profile"},
        )
        assert all(r.metadata.get("source_type") == "entity_profile" for r in results)

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, qdrant_memory_store, unique_collection):
        store = qdrant_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        vec = _rand_vec(seed=42)
        await store.upsert(unique_collection, ["x"], [vec], ["original"])
        await store.upsert(unique_collection, ["x"], [vec], ["updated"])
        assert await store.count(unique_collection) == 1
        results = await store.query(unique_collection, vec, top_k=1)
        assert results[0].content == "updated"

    @pytest.mark.asyncio
    async def test_delete_by_ids(self, qdrant_memory_store, unique_collection):
        store = qdrant_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        await store.upsert(
            collection=unique_collection,
            ids=["del1", "keep1"],
            embeddings=[_rand_vec(seed=1), _rand_vec(seed=2)],
            documents=["to delete", "to keep"],
        )
        await store.delete(unique_collection, ["del1"])
        assert await store.count(unique_collection) == 1

    @pytest.mark.asyncio
    async def test_provider_name(self, qdrant_memory_store):
        assert qdrant_memory_store.provider_name == "qdrant"

    @pytest.mark.asyncio
    async def test_query_with_score(self, qdrant_memory_store, unique_collection):
        store = qdrant_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        vec = _rand_vec(seed=50)
        await store.upsert(unique_collection, ["s1"], [vec], ["scored doc"])
        results = await store.query(unique_collection, vec, top_k=1)
        assert results[0].score >= 0.99


# =====================================================================
#  CHROMADB IN-MEMORY
# =====================================================================

class TestChromaDBMemoryStore:
    """Full CRUD tests for ChromaDB in-memory mode."""

    @pytest.mark.asyncio
    async def test_create_collection(self, chromadb_memory_store, unique_collection):
        store = chromadb_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        assert await store.collection_exists(unique_collection) is True

    @pytest.mark.asyncio
    async def test_nonexistent_collection(self, chromadb_memory_store):
        assert await chromadb_memory_store.collection_exists("no_such_col") is False

    @pytest.mark.asyncio
    async def test_count_empty(self, chromadb_memory_store, unique_collection):
        store = chromadb_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        assert await store.count(unique_collection) == 0

    @pytest.mark.asyncio
    async def test_upsert_and_count(self, chromadb_memory_store, unique_collection):
        store = chromadb_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        await store.upsert(
            collection=unique_collection,
            ids=["a", "b", "c"],
            embeddings=[_rand_vec(seed=i) for i in range(3)],
            documents=["doc a", "doc b", "doc c"],
        )
        assert await store.count(unique_collection) == 3

    @pytest.mark.asyncio
    async def test_query_returns_results(self, chromadb_memory_store, unique_collection):
        store = chromadb_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        base = _rand_vec(seed=100)
        near = _similar_vec(base, noise=0.05)
        far = _rand_vec(seed=999)
        await store.upsert(
            collection=unique_collection,
            ids=["near", "far"],
            embeddings=[near, far],
            documents=["near doc", "far doc"],
        )
        results = await store.query(unique_collection, base, top_k=2)
        assert len(results) == 2
        assert results[0].content == "near doc"

    @pytest.mark.asyncio
    async def test_query_with_metadata(self, chromadb_memory_store, unique_collection):
        store = chromadb_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        await store.upsert(
            collection=unique_collection,
            ids=["e1", "e2"],
            embeddings=[_rand_vec(seed=1), _rand_vec(seed=2)],
            documents=["entity", "chunk"],
            metadatas=[{"source_type": "entity_profile"}, {"source_type": "chunk"}],
        )
        results = await store.query(
            unique_collection, _rand_vec(seed=1), top_k=10,
            filter={"source_type": "entity_profile"},
        )
        assert all(r.metadata.get("source_type") == "entity_profile" for r in results)

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, chromadb_memory_store, unique_collection):
        store = chromadb_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        vec = _rand_vec(seed=42)
        await store.upsert(unique_collection, ["x"], [vec], ["original"])
        await store.upsert(unique_collection, ["x"], [vec], ["updated"])
        assert await store.count(unique_collection) == 1

    @pytest.mark.asyncio
    async def test_delete_by_ids(self, chromadb_memory_store, unique_collection):
        store = chromadb_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        await store.upsert(
            collection=unique_collection,
            ids=["del1", "keep1"],
            embeddings=[_rand_vec(seed=1), _rand_vec(seed=2)],
            documents=["to delete", "to keep"],
        )
        await store.delete(unique_collection, ["del1"])
        assert await store.count(unique_collection) == 1

    @pytest.mark.asyncio
    async def test_provider_name(self, chromadb_memory_store):
        assert chromadb_memory_store.provider_name == "chromadb"

    @pytest.mark.asyncio
    async def test_score_range(self, chromadb_memory_store, unique_collection):
        store = chromadb_memory_store
        await store.create_collection(unique_collection, dimensions=DIMS)
        vec = _rand_vec(seed=50)
        await store.upsert(unique_collection, ["s1"], [vec], ["scored doc"])
        results = await store.query(unique_collection, vec, top_k=1)
        assert 0.0 <= results[0].score <= 1.0


# =====================================================================
#  VECTOR STORE FACTORY
# =====================================================================

class TestVectorStoreFactory:
    """Test factory creation with in-memory/path backends."""

    def test_create_qdrant_from_path(self, tmp_path):
        from ayextractor.rag.vector_store.vector_store_factory import create_vector_store
        s = Settings(vector_db_type="qdrant", vector_db_path=str(tmp_path / "qdrant_data"))
        store = create_vector_store(s)
        assert store.provider_name == "qdrant"

    def test_create_chromadb_from_path(self, tmp_path):
        from ayextractor.rag.vector_store.vector_store_factory import create_vector_store
        try:
            s = Settings(vector_db_type="chromadb", vector_db_path=str(tmp_path / "chroma_data"))
            store = create_vector_store(s)
            assert store.provider_name == "chromadb"
        except RuntimeError as e:
            if "http-only" in str(e).lower():
                pytest.skip("chromadb http-only client cannot create persistent store")
            raise

    def test_create_none_raises(self):
        from ayextractor.rag.vector_store.vector_store_factory import create_vector_store
        s = Settings(vector_db_type="none")
        with pytest.raises(ValueError):
            create_vector_store(s)

    def test_create_unsupported_raises(self):
        """vector_db_type not in Literal → Pydantic ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(vector_db_type="nonexistent_db")