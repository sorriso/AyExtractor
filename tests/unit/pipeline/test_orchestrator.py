# tests/unit/pipeline/test_orchestrator.py — v1
"""Tests for pipeline/orchestrator.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ayextractor.pipeline.orchestrator import PipelineOrchestrator, _should_run


class TestShouldRun:
    def test_no_resume_runs_all(self):
        assert _should_run("phase3_dag", None, []) is True
        assert _should_run("phase4_finalize", None, []) is True

    def test_already_completed_skips(self):
        assert _should_run("phase3_dag", None, ["phase3_dag"]) is False

    def test_resume_from_phase4(self):
        assert _should_run("phase3_dag", "phase4_finalize", []) is False
        assert _should_run("phase4_finalize", "phase4_finalize", []) is True


class TestPipelineOrchestrator:
    def _make_orchestrator(self, **kwargs):
        settings = MagicMock()
        settings.RAG_ENABLED = False
        settings.CONSOLIDATOR_ENABLED = False
        llm = MagicMock()
        return PipelineOrchestrator(settings=settings, llm_client=llm, **kwargs)

    def test_init(self):
        orch = self._make_orchestrator()
        assert orch._vector_store is None
        assert orch._graph_store is None

    def test_init_with_stores(self):
        vs = MagicMock()
        gs = MagicMock()
        orch = self._make_orchestrator(vector_store=vs, graph_store=gs)
        assert orch._vector_store is vs
        assert orch._graph_store is gs

    def test_enrich_if_needed_rag_disabled(self):
        orch = self._make_orchestrator()
        state = MagicMock()

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            orch._enrich_if_needed("synthesizer", state)
        )
        loop.close()

        assert result.rag_context is None

    def test_enrich_if_needed_rag_enabled(self):
        orch = self._make_orchestrator(
            vector_store=MagicMock(),
            embedder=MagicMock(),
        )
        orch._settings.RAG_ENABLED = True
        orch._settings.RAG_ENRICH_AGENTS = "synthesizer,concept_extractor"
        orch._settings.CONSOLIDATOR_ENABLED = False
        orch._settings.RAG_RETRIEVAL_TOP_K_COMMUNITIES = 5
        orch._settings.RAG_RETRIEVAL_TOP_K_ENTITIES = 20
        orch._settings.RAG_RETRIEVAL_TOP_K_CHUNKS = 10
        orch._settings.RAG_CHUNK_FALLBACK_THRESHOLD = 0.6
        orch._settings.RAG_COMPOSITE_WEIGHT = 0.3
        orch._settings.RAG_CONTEXT_TOKEN_BUDGET = 4000

        state = MagicMock()
        state.dense_summary = "test summary"
        state.references = []
        state.qualified_triplets = []
        state.refine_summary = ""
        state.graph = None

        from ayextractor.rag.models import RAGContext

        mock_ctx = RAGContext(assembled_text="enriched", total_token_count=5)

        with patch(
            "ayextractor.rag.retriever.pipeline.RetrievalPipeline"
        ) as MockPipeline:
            instance = MockPipeline.return_value
            instance.retrieve = AsyncMock(return_value=mock_ctx)

            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                orch._enrich_if_needed("synthesizer", state)
            )
            loop.close()

        # state.rag_context should have been set
        assert state.rag_context is not None


class TestPhase4Finalize:
    def test_finalize_calls_indexer(self):
        settings = MagicMock()
        settings.RAG_ENABLED = False
        settings.CONSOLIDATOR_ENABLED = False
        orch = PipelineOrchestrator(settings=settings, llm_client=MagicMock())

        state = MagicMock()
        state.graph = None
        state.chunks = []

        from ayextractor.rag.indexer import IndexingReport

        mock_report = IndexingReport()

        with patch(
            "ayextractor.rag.indexer.index_analysis_results",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(orch._run_phase4_finalize(state))
            loop.close()

        assert result is state
