# tests/unit/rag/retriever/test_chunk_retriever.py — v1
"""Tests for rag/retriever/chunk_retriever.py."""

from __future__ import annotations

import asyncio

import pytest

from ayextractor.rag.models import SearchResult
from ayextractor.rag.retriever.chunk_retriever import (
    retrieve_chunks,
    should_retrieve_chunks,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2]


class _FakeVectorStore:
    def __init__(self, results=None):
        self._results = results or []

    async def search(self, collection: str, query_vector: list, top_k: int = 5):
        return self._results[:top_k]


class TestRetrieveChunks:
    def test_returns_chunk_results(self):
        store = _FakeVectorStore([
            {"id": "ch1", "content": "Chunk text", "score": 0.85, "metadata": {}},
        ])
        results = _run(retrieve_chunks("test", store, _FakeEmbedder()))
        assert len(results) == 1
        assert results[0].source_type == "chunk"

    def test_prunes_low_scores(self):
        store = _FakeVectorStore([
            {"id": "ch1", "content": "Good", "score": 0.8, "metadata": {}},
            {"id": "ch2", "content": "Low", "score": 0.05, "metadata": {}},
        ])
        results = _run(retrieve_chunks("test", store, _FakeEmbedder(), min_score=0.2))
        assert len(results) == 1

    def test_empty_store(self):
        store = _FakeVectorStore([])
        results = _run(retrieve_chunks("test", store, _FakeEmbedder()))
        assert results == []


class TestShouldRetrieveChunks:
    def test_empty_results_triggers_fallback(self):
        assert should_retrieve_chunks([]) is True

    def test_high_confidence_no_fallback(self):
        results = [
            SearchResult(source_type="entity_profile", source_id="e1", content="", score=0.9, metadata={}),
            SearchResult(source_type="entity_profile", source_id="e2", content="", score=0.8, metadata={}),
        ]
        assert should_retrieve_chunks(results, fallback_threshold=0.6) is False

    def test_low_confidence_triggers_fallback(self):
        results = [
            SearchResult(source_type="entity_profile", source_id="e1", content="", score=0.3, metadata={}),
            SearchResult(source_type="entity_profile", source_id="e2", content="", score=0.4, metadata={}),
        ]
        assert should_retrieve_chunks(results, fallback_threshold=0.6) is True
