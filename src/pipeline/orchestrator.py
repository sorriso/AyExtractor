# src/pipeline/orchestrator.py — v1
"""Pipeline orchestrator — LangGraph workflow definition.

Drives the full 4-phase pipeline:
  Phase 1: Extraction (sequential, orchestrator-driven)
  Phase 2: Chunking + Decontextualization + Summarization + Densification
  Phase 3: Agent DAG (LangGraph — concept extraction → graph → synthesis)
  Phase 4: Finalization (export, indexing, consolidation)

The orchestrator manages RAG enrichment injection before each eligible agent.

See spec §6 for full pipeline overview.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ayextractor.pipeline.state import PipelineState

if TYPE_CHECKING:
    from ayextractor.config.settings import Settings
    from ayextractor.llm.base_client import BaseLLMClient
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder
    from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore
    from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore
    from ayextractor.tracking.call_logger import CallLogger

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Top-level orchestrator for the full document analysis pipeline.

    Coordinates all 4 phases, manages RAG enrichment, and handles
    finalization (indexing + consolidation).

    Args:
        settings: Application settings.
        llm_client: LLM client for agent calls.
        call_logger: Optional tracking logger.
        vector_store: Optional vector store for RAG.
        graph_store: Optional graph store for RAG/consolidation.
        embedder: Optional embedder for RAG indexing.
    """

    def __init__(
        self,
        settings: Settings,
        llm_client: BaseLLMClient,
        call_logger: CallLogger | None = None,
        vector_store: BaseVectorStore | None = None,
        graph_store: BaseGraphStore | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self._settings = settings
        self._llm_client = llm_client
        self._call_logger = call_logger
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._embedder = embedder

    async def run(
        self,
        state: PipelineState,
        resume_from_step: str | None = None,
    ) -> PipelineState:
        """Execute the full pipeline on a prepared PipelineState.

        Args:
            state: PipelineState initialized with extraction context.
            resume_from_step: Optional step to resume from (for recovery).

        Returns:
            Completed PipelineState with all analysis results.
        """
        start_time = time.monotonic()
        steps_completed: list[str] = []

        try:
            # Phase 3: Agent DAG
            if _should_run("phase3_dag", resume_from_step, steps_completed):
                state = await self._run_phase3_dag(state)
                steps_completed.append("phase3_dag")

            # Phase 4: Finalization (export + indexing + consolidation)
            if _should_run("phase4_finalize", resume_from_step, steps_completed):
                state = await self._run_phase4_finalize(state)
                steps_completed.append("phase4_finalize")

        except Exception:
            logger.exception("Pipeline failed after steps: %s", steps_completed)
            raise

        elapsed = time.monotonic() - start_time
        logger.info(
            "Pipeline complete: %d steps in %.1fs",
            len(steps_completed), elapsed,
        )
        return state

    # ------------------------------------------------------------------
    # Phase 3: Agent DAG
    # ------------------------------------------------------------------

    async def _run_phase3_dag(self, state: PipelineState) -> PipelineState:
        """Execute Phase 3 agent DAG with RAG enrichment.

        DAG order:
        1. Concept Extractor (per-chunk triplets)
        2. Merger (entity norm + relation norm + triplet consolidation)
        3. Graph Builder (L2/L3)
        4. Community Detector (Leiden on L2)
        5. Community Integrator (inject L1)
        6. Community Summarizer (LLM)
        7. Profile Generator (LLM)
        8. Synthesizer (LLM)
        9. Critic (optional, LLM)
        """
        from ayextractor.pipeline.dag_builder import build_dag
        from ayextractor.pipeline.registry import AgentRegistry
        from ayextractor.pipeline.runner import PipelineRunner

        logger.info("Starting Phase 3 — Agent DAG")

        # Build DAG from registered agents
        registry = AgentRegistry()
        registry.auto_register()
        plan = build_dag(registry)

        # Create runner
        runner = PipelineRunner(
            llm_client=self._llm_client,
            call_logger=self._call_logger,
        )

        # Run DAG stages with RAG enrichment
        for stage in plan.stages:
            for agent_entry in stage.agents:
                # Inject RAG context if applicable
                state = await self._enrich_if_needed(agent_entry.name, state)

            # Execute stage
            result = await runner.run_stage(stage, state)
            if result.error:
                logger.error("Stage '%s' failed: %s", stage.name, result.error)
                raise RuntimeError(f"Stage '{stage.name}' failed: {result.error}")

        return state

    async def _enrich_if_needed(
        self, agent_name: str, state: PipelineState
    ) -> PipelineState:
        """Inject RAG context into state before agent execution."""
        from ayextractor.rag.enricher import build_context, should_enrich

        rag_enabled = getattr(self._settings, "RAG_ENABLED", False)
        enrich_agents_raw = getattr(self._settings, "RAG_ENRICH_AGENTS", None)
        enrich_agents = None
        if enrich_agents_raw and isinstance(enrich_agents_raw, str):
            enrich_agents = {a.strip() for a in enrich_agents_raw.split(",")}

        if not should_enrich(agent_name, rag_enabled, enrich_agents):
            state.rag_context = None
            return state

        knowledge_graph = None
        if hasattr(state, "graph") and state.graph is not None:
            knowledge_graph = state.graph

        rag_context = await build_context(
            agent_name=agent_name,
            state=state,
            vector_store=self._vector_store,
            graph_store=self._graph_store,
            embedder=self._embedder,
            knowledge_graph=knowledge_graph,
            settings=self._settings,
        )
        state.rag_context = rag_context
        return state

    # ------------------------------------------------------------------
    # Phase 4: Finalization
    # ------------------------------------------------------------------

    async def _run_phase4_finalize(self, state: PipelineState) -> PipelineState:
        """Execute Phase 4 — export, indexing, consolidation.

        1. Export graph to configured formats
        2. Index to vector DB (chunks + profiles + communities)
        3. Index to graph DB (Document Graph)
        4. Run consolidator Pass 1 (Linking) if enabled
        """
        logger.info("Starting Phase 4 — Finalization")

        # Build document graph if available
        document_graph = getattr(state, "graph", None)

        # Index analysis results
        from ayextractor.rag.indexer import index_analysis_results

        report = await index_analysis_results(
            state=state,
            vector_store=self._vector_store,
            graph_store=self._graph_store,
            embedder=self._embedder,
            settings=self._settings,
            document_graph=document_graph,
        )

        logger.info(
            "Phase 4 complete: %d chunks, %d entities, %d nodes indexed, consolidator=%s",
            report.chunks_indexed,
            report.entity_profiles_indexed,
            report.graph_nodes_imported,
            report.consolidator_linked,
        )

        return state


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _should_run(
    step: str,
    resume_from: str | None,
    completed: list[str],
) -> bool:
    """Check if a step should execute based on resume point."""
    if step in completed:
        return False
    if resume_from is None:
        return True
    # Skip steps before resume point
    all_steps = ["phase3_dag", "phase4_finalize"]
    if resume_from in all_steps:
        idx_resume = all_steps.index(resume_from)
        idx_current = all_steps.index(step) if step in all_steps else 0
        return idx_current >= idx_resume
    return True
