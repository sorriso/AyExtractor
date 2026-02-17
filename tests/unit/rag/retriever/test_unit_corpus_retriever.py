# tests/unit/rag/retriever/test_corpus_retriever.py — v1
"""Tests for rag/retriever/corpus_retriever.py."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from ayextractor.consolidator.models import CNode, TNode
from ayextractor.core.models import SourceProvenance
from ayextractor.rag.retriever.corpus_retriever import (
    corpus_context_to_search_results,
    retrieve_corpus_context,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


_NOW = datetime.now(timezone.utc)


def _make_cnode(name: str, **kwargs) -> CNode:
    defaults = dict(
        canonical_name=name,
        entity_type="ORG",
        confidence=0.8,
        salience=0.5,
        first_seen_at=_NOW,
        last_updated_at=_NOW,
    )
    defaults.update(kwargs)
    return CNode(**defaults)


class _FakeGraphStore:
    def __init__(self, cnodes=None, tnodes=None):
        self._cnodes = cnodes or []
        self._tnodes = tnodes or []

    def get_nodes_by_names(self, names):
        return [c.model_dump() for c in self._cnodes if c.canonical_name in names]

    def search_nodes(self, keyword, limit=10):
        return [
            c.model_dump()
            for c in self._cnodes
            if keyword.lower() in c.canonical_name.lower()
        ][:limit]

    def get_tnodes(self):
        return self._tnodes

    def get_edges_for_nodes(self, names, limit=50):
        return []


class TestRetrieveCorpusContext:
    def test_basic_retrieval(self):
        cnodes = [_make_cnode("Anthropic"), _make_cnode("OpenAI")]
        store = _FakeGraphStore(cnodes=cnodes)
        ctx = _run(retrieve_corpus_context("AI companies", store, seed_entities=["Anthropic"]))
        assert len(ctx.cnodes) >= 1
        assert ctx.source_document_count == 0  # no source_documents set

    def test_empty_store(self):
        store = _FakeGraphStore()
        ctx = _run(retrieve_corpus_context("test", store))
        assert ctx.cnodes == []
        assert ctx.tnodes == []

    def test_with_tnodes(self):
        cnodes = [_make_cnode("Anthropic")]
        tnodes = [
            TNode(
                canonical_name="AI_Companies",
                level="concept",
                classified_cnodes=["Anthropic"],
                created_by="consolidator_clustering",
                created_at=_NOW,
            )
        ]
        store = _FakeGraphStore(cnodes=cnodes, tnodes=tnodes)
        ctx = _run(retrieve_corpus_context("AI", store, seed_entities=["Anthropic"]))
        assert len(ctx.tnodes) >= 1


class TestCorpusContextToSearchResults:
    def test_converts_cnodes(self):
        from ayextractor.rag.models import CorpusContext

        ctx = CorpusContext(
            cnodes=[_make_cnode("TestCo", corroboration=3)],
            source_document_count=2,
        )
        results = corpus_context_to_search_results(ctx)
        assert len(results) == 1
        assert results[0].source_type == "entity_profile"
        assert results[0].metadata["source"] == "corpus_graph"

    def test_empty_context(self):
        from ayextractor.rag.models import CorpusContext

        ctx = CorpusContext()
        results = corpus_context_to_search_results(ctx)
        assert results == []
