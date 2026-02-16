# tests/unit/rag/test_indexer.py — v1
"""Tests for rag/indexer.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import networkx as nx
import pytest

from ayextractor.rag.indexer import (
    IndexingReport,
    _chunk_metadata,
    _get_chunk_content,
    index_analysis_results,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeChunk:
    def __init__(self, chunk_id="c1", content="text", context_summary="ctx", global_summary="glob"):
        self.chunk_id = chunk_id
        self.content = content
        self.context_summary = context_summary
        self.global_summary = global_summary
        self.document_id = "doc1"
        self.chunk_index = 0


class _FakeVectorStore:
    def __init__(self):
        self.upserted = []

    async def upsert(self, collection, id, content, vector, metadata=None):
        self.upserted.append((collection, id))


class _FakeEmbedder:
    async def embed(self, text):
        return [0.1, 0.2, 0.3]


class TestIndexingReport:
    def test_defaults(self):
        r = IndexingReport()
        assert r.chunks_indexed == 0
        assert r.consolidator_linked is False
        assert r.errors == []


class TestGetChunkContent:
    def test_combines_fields(self):
        chunk = _FakeChunk()
        content = _get_chunk_content(chunk)
        assert "text" in content
        assert "ctx" in content
        assert "glob" in content

    def test_empty_chunk(self):
        chunk = MagicMock(spec=[])
        content = _get_chunk_content(chunk)
        assert content == ""


class TestChunkMetadata:
    def test_extracts_known_fields(self):
        chunk = _FakeChunk()
        meta = _chunk_metadata(chunk)
        assert meta["document_id"] == "doc1"
        assert meta["chunk_index"] == 0


class TestIndexAnalysisResults:
    def test_indexes_chunks_to_vector_store(self):
        state = MagicMock()
        state.chunks = [_FakeChunk("c1"), _FakeChunk("c2")]
        state.entity_profiles = []
        state.relation_profiles = []
        state.community_summaries = []

        store = _FakeVectorStore()
        embedder = _FakeEmbedder()

        report = _run(
            index_analysis_results(state, vector_store=store, embedder=embedder)
        )
        assert report.chunks_indexed == 2
        assert len(store.upserted) == 2

    def test_no_stores_returns_empty_report(self):
        state = MagicMock()
        state.chunks = [_FakeChunk()]
        state.entity_profiles = []
        state.relation_profiles = []
        state.community_summaries = []

        report = _run(index_analysis_results(state))
        assert report.chunks_indexed == 0

    def test_graph_db_import(self):
        state = MagicMock()
        state.chunks = []
        state.entity_profiles = []
        state.relation_profiles = []
        state.community_summaries = []

        g = nx.Graph()
        g.add_node("A", layer="L2")
        g.add_edge("A", "B", relation="knows")

        graph_store = MagicMock()
        graph_store.upsert_node = MagicMock()
        graph_store.upsert_edge = MagicMock()

        report = _run(
            index_analysis_results(state, graph_store=graph_store, document_graph=g)
        )
        assert report.graph_nodes_imported == 2
        assert report.graph_edges_imported == 1
