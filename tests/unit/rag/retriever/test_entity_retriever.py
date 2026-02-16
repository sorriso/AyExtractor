# tests/unit/rag/retriever/test_entity_retriever.py — v1
"""Tests for rag/retriever/entity_retriever.py."""

from __future__ import annotations

import asyncio

import pytest

from ayextractor.rag.retriever.entity_retriever import (
    _convert_results,
    retrieve_entities,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeVectorStore:
    def __init__(self, entity_results=None, relation_results=None):
        self._entity = entity_results or []
        self._relation = relation_results or []

    async def search(self, collection: str, query_vector: list, top_k: int = 5):
        if collection == "entity_profiles":
            return self._entity[:top_k]
        elif collection == "relation_profiles":
            return self._relation[:top_k]
        return []


class TestRetrieveEntities:
    def test_returns_combined_results(self):
        store = _FakeVectorStore(
            entity_results=[
                {"id": "e1", "content": "Entity A", "score": 0.9, "metadata": {}},
            ],
            relation_results=[
                {"id": "r1", "content": "Relation X", "score": 0.8, "metadata": {}},
            ],
        )
        results = _run(retrieve_entities("test", store, _FakeEmbedder()))
        types = {r.source_type for r in results}
        assert "entity_profile" in types
        assert "relation_profile" in types

    def test_prunes_low_scores(self):
        store = _FakeVectorStore(
            entity_results=[
                {"id": "e1", "content": "Good", "score": 0.9, "metadata": {}},
                {"id": "e2", "content": "Bad", "score": 0.1, "metadata": {}},
            ],
        )
        results = _run(retrieve_entities("test", store, _FakeEmbedder(), min_score=0.25))
        assert all(r.score >= 0.25 for r in results)

    def test_empty_store(self):
        store = _FakeVectorStore()
        results = _run(retrieve_entities("test", store, _FakeEmbedder()))
        assert results == []


class TestConvertResults:
    def test_dict_results(self):
        raw = [{"id": "x", "content": "text", "score": 0.5, "metadata": {"k": "v"}}]
        converted = _convert_results(raw, "entity_profile")
        assert len(converted) == 1
        assert converted[0].source_type == "entity_profile"
        assert converted[0].score == 0.5

    def test_empty_list(self):
        assert _convert_results([], "entity_profile") == []
