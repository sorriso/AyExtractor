# tests/unit/rag/test_models.py — v1
"""Tests for rag/models.py — RAG retrieval models."""

from __future__ import annotations

from ayextractor.rag.models import (
    CorpusContext,
    RAGContext,
    RetrievalPlan,
    SearchResult,
)


class TestSearchResult:
    def test_create(self):
        sr = SearchResult(
            source_type="chunk", source_id="chunk_001",
            content="Some content", score=0.92,
        )
        assert sr.metadata == {}

    def test_all_source_types(self):
        for st in ["chunk", "entity_profile", "relation_profile", "community_summary"]:
            sr = SearchResult(source_type=st, source_id="x", content="y", score=0.5)
            assert sr.source_type == st


class TestRAGContext:
    def test_empty(self):
        ctx = RAGContext(assembled_text="")
        assert ctx.total_token_count == 0
        assert ctx.corpus_context is None

    def test_with_results(self):
        ctx = RAGContext(
            assembled_text="Context text here",
            community_summaries=["Summary 1"],
            entity_profiles=["Profile 1"],
            total_token_count=150,
        )
        assert len(ctx.community_summaries) == 1


class TestRetrievalPlan:
    def test_create(self):
        rp = RetrievalPlan(
            query_type="conceptual",
            levels_to_query=["community", "entity"],
            estimated_token_cost=2000,
        )
        assert len(rp.levels_to_query) == 2


class TestCorpusContext:
    def test_empty(self):
        cc = CorpusContext()
        assert cc.source_document_count == 0
