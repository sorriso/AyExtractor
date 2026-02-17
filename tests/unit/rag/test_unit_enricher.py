# tests/unit/rag/test_enricher.py — v1
"""Tests for rag/enricher.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ayextractor.rag.enricher import (
    DEFAULT_ENRICH_AGENTS,
    _derive_query,
    build_context,
    should_enrich,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestShouldEnrich:
    def test_disabled_rag(self):
        assert should_enrich("synthesizer", rag_enabled=False) is False

    def test_enabled_default_agents(self):
        for agent in DEFAULT_ENRICH_AGENTS:
            assert should_enrich(agent, rag_enabled=True) is True

    def test_non_enrichable_agent(self):
        assert should_enrich("densifier", rag_enabled=True) is False

    def test_custom_agent_set(self):
        assert should_enrich(
            "summarizer", rag_enabled=True, enrich_agents={"summarizer"}
        ) is True
        assert should_enrich(
            "synthesizer", rag_enabled=True, enrich_agents={"summarizer"}
        ) is False


class TestDeriveQuery:
    def test_with_dense_summary(self):
        state = MagicMock()
        state.dense_summary = "This document discusses AI safety"
        state.references = []
        state.qualified_triplets = []
        state.refine_summary = ""
        query = _derive_query("synthesizer", state)
        assert "AI safety" in query

    def test_concept_extractor_with_triplets(self):
        state = MagicMock()
        state.dense_summary = ""
        state.qualified_triplets = [
            MagicMock(subject="Anthropic", object="AI"),
            MagicMock(subject="OpenAI", object="GPT"),
        ]
        query = _derive_query("concept_extractor", state)
        assert "entities:" in query

    def test_empty_state(self):
        state = MagicMock(spec=[])  # No attributes
        query = _derive_query("synthesizer", state)
        assert query == ""


class TestBuildContext:
    def test_no_stores_returns_none(self):
        state = MagicMock()
        result = _run(build_context("synthesizer", state))
        assert result is None

    def test_vector_store_without_embedder_returns_none(self):
        state = MagicMock()
        result = _run(
            build_context(
                "synthesizer",
                state,
                vector_store=MagicMock(),
                embedder=None,
            )
        )
        assert result is None

    def test_with_stores_returns_context(self):
        state = MagicMock()
        state.dense_summary = "Test document summary"
        state.references = []
        state.qualified_triplets = []
        state.refine_summary = ""

        # Patch the pipeline to return a mock RAGContext
        from ayextractor.rag.models import RAGContext

        mock_ctx = RAGContext(assembled_text="test context", total_token_count=10)

        with patch(
            "ayextractor.rag.retriever.pipeline.RetrievalPipeline"
        ) as MockPipeline:
            instance = MockPipeline.return_value
            instance.retrieve = AsyncMock(return_value=mock_ctx)

            result = _run(
                build_context(
                    "synthesizer",
                    state,
                    vector_store=MagicMock(),
                    embedder=MagicMock(),
                )
            )

        assert result is not None
        assert result.assembled_text == "test context"
