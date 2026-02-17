# tests/unit/rag/retriever/test_community_retriever.py — v1
"""Tests for rag/retriever/community_retriever.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ayextractor.rag.retriever.community_retriever import retrieve_communities


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeVectorStore:
    def __init__(self, results: list[dict] | None = None):
        self._results = results or []

    async def search(self, collection: str, query_vector: list, top_k: int = 5) -> list[dict]:
        return self._results[:top_k]


class TestRetrieveCommunities:
    def test_returns_search_results(self):
        store = _FakeVectorStore([
            {"id": "c1", "content": "Community about AI", "score": 0.9, "metadata": {}},
            {"id": "c2", "content": "Community about ML", "score": 0.7, "metadata": {}},
        ])
        results = _run(retrieve_communities("AI", store, _FakeEmbedder()))
        assert len(results) == 2
        assert results[0].source_type == "community_summary"
        assert results[0].score >= results[1].score

    def test_prunes_low_scores(self):
        store = _FakeVectorStore([
            {"id": "c1", "content": "Good match", "score": 0.8, "metadata": {}},
            {"id": "c2", "content": "Poor match", "score": 0.1, "metadata": {}},
        ])
        results = _run(retrieve_communities("test", store, _FakeEmbedder(), min_score=0.3))
        assert len(results) == 1
        assert results[0].source_id == "c1"

    def test_respects_top_k(self):
        store = _FakeVectorStore([
            {"id": f"c{i}", "content": f"Community {i}", "score": 0.9 - i * 0.1, "metadata": {}}
            for i in range(10)
        ])
        results = _run(retrieve_communities("test", store, _FakeEmbedder(), top_k=3))
        assert len(results) <= 3

    def test_empty_results(self):
        store = _FakeVectorStore([])
        results = _run(retrieve_communities("test", store, _FakeEmbedder()))
        assert results == []
