# tests/unit/rag/retriever/test_pipeline.py — v1
"""Tests for rag/retriever/pipeline.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ayextractor.rag.models import RAGContext
from ayextractor.rag.retriever.pipeline import RetrievalPipeline


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeVectorStore:
    async def search(self, collection: str, query_vector: list, top_k: int = 5):
        return [
            {"id": f"{collection}_1", "content": f"Content from {collection}", "score": 0.8, "metadata": {}},
        ]


class TestRetrievalPipeline:
    def test_init_default(self):
        pipe = RetrievalPipeline()
        assert pipe._vector_store is None

    def test_init_with_stores(self):
        pipe = RetrievalPipeline(
            vector_store=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
        )
        assert pipe._vector_store is not None

    def test_retrieve_returns_rag_context(self):
        pipe = RetrievalPipeline(
            vector_store=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
        )
        result = _run(pipe.retrieve("What are the main topics?"))
        assert isinstance(result, RAGContext)
        assert result.assembled_text  # Should have some content

    def test_retrieve_without_stores_returns_empty(self):
        pipe = RetrievalPipeline()
        result = _run(pipe.retrieve("test"))
        assert isinstance(result, RAGContext)
        assert result.assembled_text == ""

    def test_retrieve_with_consolidator(self):
        pipe = RetrievalPipeline(
            vector_store=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
            graph_store=MagicMock(),
        )
        # Patch corpus retriever to avoid real graph store calls
        with patch(
            "ayextractor.rag.retriever.pipeline.RetrievalPipeline._retrieve_corpus",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = _run(pipe.retrieve("test", consolidator_enabled=True))
        assert isinstance(result, RAGContext)

    def test_config_overrides(self):
        pipe = RetrievalPipeline(
            config={
                "context_token_budget": 2000,
                "top_k_communities": 3,
                "top_k_entities": 10,
                "top_k_chunks": 5,
            }
        )
        assert pipe._assembler._max_tokens == 2000
