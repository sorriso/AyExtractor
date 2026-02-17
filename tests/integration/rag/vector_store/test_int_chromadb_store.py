# tests/integration/rag/vector_store/test_int_chromadb_store.py — v2
"""Integration tests for ChromaDB vector store via testcontainers.

Coverage target: chromadb_store.py 33% → 90%+

Source API reference:
- ChromaDBStore(persist_path, host, port) — in-process if all None
- SearchResult.source_type: Literal["chunk","entity_profile","relation_profile","community_summary"]

Changelog:
    v2: Fix source_type Literal — "entity" is not valid, must be "entity_profile".
"""

from __future__ import annotations

import math
import random

import pytest

pytestmark = [pytest.mark.chromadb]

DIMS = 128


def _rand_vec(dims: int = DIMS, seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class TestCollectionManagement:

    @pytest.mark.asyncio
    async def test_create_collection(self, chromadb_store, chromadb_collection):
        await chromadb_store.create_collection(chromadb_collection, dimensions=DIMS)
        assert await chromadb_store.collection_exists(chromadb_collection)

    @pytest.mark.asyncio
    async def test_nonexistent_collection(self, chromadb_store):
        assert not await chromadb_store.collection_exists("surely_missing_xyz")

    @pytest.mark.asyncio
    async def test_count_empty(self, chromadb_store, chromadb_collection):
        await chromadb_store.create_collection(chromadb_collection, dimensions=DIMS)
        assert await chromadb_store.count(chromadb_collection) == 0


class TestUpsertAndQuery:

    @pytest.mark.asyncio
    async def test_upsert_and_count(self, chromadb_store, chromadb_collection):
        await chromadb_store.create_collection(chromadb_collection, dimensions=DIMS)
        vecs = [_rand_vec(DIMS, seed=i) for i in range(5)]
        await chromadb_store.upsert(
            chromadb_collection,
            ids=[f"d{i}" for i in range(5)],
            embeddings=vecs,
            documents=[f"text_{i}" for i in range(5)],
        )
        assert await chromadb_store.count(chromadb_collection) == 5

    @pytest.mark.asyncio
    async def test_query_returns_results(self, chromadb_store, chromadb_collection):
        await chromadb_store.create_collection(chromadb_collection, dimensions=DIMS)
        vecs = [_rand_vec(DIMS, seed=i) for i in range(3)]
        await chromadb_store.upsert(
            chromadb_collection,
            ids=["a", "b", "c"], embeddings=vecs,
            documents=["doc a", "doc b", "doc c"],
        )
        results = await chromadb_store.query(
            chromadb_collection, query_embedding=vecs[0], top_k=2,
        )
        assert len(results) == 2
        assert results[0].source_id == "a"  # exact match first
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_query_with_metadata(self, chromadb_store, chromadb_collection):
        """Filter by source_type metadata using valid Literal values."""
        await chromadb_store.create_collection(chromadb_collection, dimensions=DIMS)
        vecs = [_rand_vec(DIMS, seed=i) for i in range(4)]
        await chromadb_store.upsert(
            chromadb_collection,
            ids=["e1", "e2", "c1", "c2"], embeddings=vecs,
            documents=["ent1", "ent2", "chunk1", "chunk2"],
            metadatas=[
                {"source_type": "entity_profile"}, {"source_type": "entity_profile"},
                {"source_type": "chunk"}, {"source_type": "chunk"},
            ],
        )
        results = await chromadb_store.query(
            chromadb_collection, vecs[0], top_k=10,
            filter={"source_type": "entity_profile"},
        )
        assert len(results) >= 1
        assert all(r.metadata.get("source_type") == "entity_profile" for r in results)

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, chromadb_store, chromadb_collection):
        await chromadb_store.create_collection(chromadb_collection, dimensions=DIMS)
        v1 = _rand_vec(DIMS, seed=1)
        await chromadb_store.upsert(chromadb_collection, ["x"], [v1], ["old"])
        v2 = _rand_vec(DIMS, seed=2)
        await chromadb_store.upsert(chromadb_collection, ["x"], [v2], ["new"])
        assert await chromadb_store.count(chromadb_collection) == 1
        results = await chromadb_store.query(chromadb_collection, v2, top_k=1)
        assert results[0].content == "new"


class TestDelete:

    @pytest.mark.asyncio
    async def test_delete_by_ids(self, chromadb_store, chromadb_collection):
        await chromadb_store.create_collection(chromadb_collection, dimensions=DIMS)
        vecs = [_rand_vec(DIMS, seed=i) for i in range(3)]
        await chromadb_store.upsert(chromadb_collection, ["a", "b", "c"], vecs, ["A", "B", "C"])
        await chromadb_store.delete(chromadb_collection, ids=["a", "b"])
        assert await chromadb_store.count(chromadb_collection) == 1

    @pytest.mark.asyncio
    async def test_provider_name(self, chromadb_store):
        assert chromadb_store.provider_name == "chromadb"