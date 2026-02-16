# src/pipeline/document_pipeline.py — v1
"""Document pipeline — top-level orchestrator for full document processing.

Chains all phases:
  Phase 1 — Extraction (adapter → enriched text + structure + references)
  Phase 2 — Chunking + Decontextualization + Summarization + Densification
  Phase 3 — Agent DAG (concept extraction → graph → communities → profiles → synthesis → critic)

Produces a fully populated PipelineState with graph, profiles, and synthesis.

See spec §6 for full pipeline overview.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ayextractor.pipeline.dag_builder import build_dag
from ayextractor.pipeline.registry import AgentRegistry
from ayextractor.pipeline.runner import PipelineRunner, RunResult
from ayextractor.pipeline.state import PipelineState

if TYPE_CHECKING:
    from ayextractor.config.settings import Settings
    from ayextractor.llm.base_client import BaseLLMClient
    from ayextractor.tracking.call_logger import CallLogger

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Top-level orchestrator for processing a single document.

    Usage:
        pipeline = DocumentPipeline(settings, llm_factory)
        result = await pipeline.process(extraction_result, document_id="doc1")
    """

    def __init__(
        self,
        settings: Settings | None = None,
        llm_factory: Any = None,
        call_logger: CallLogger | None = None,
        disabled_agents: set[str] | None = None,
        max_retries: int = 2,
        min_quality: float = 0.2,
        fail_fast: bool = False,
    ) -> None:
        self._settings = settings
        self._llm_factory = llm_factory
        self._call_logger = call_logger
        self._disabled_agents = disabled_agents or set()
        self._max_retries = max_retries
        self._min_quality = min_quality
        self._fail_fast = fail_fast

        # Initialize registry and plan
        self._registry = AgentRegistry()
        self._registry.load_all(disabled=self._disabled_agents)
        self._plan = self._build_plan()

    def _build_plan(self):
        """Build execution plan from registry."""
        dep_map = self._registry.get_dependency_map()
        return build_dag(dep_map)

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    @property
    def execution_order(self) -> list[str]:
        return self._plan.flat_order

    async def process(
        self,
        state: PipelineState | None = None,
        document_id: str = "",
        document_title: str = "",
        language: str = "en",
        **kwargs: Any,
    ) -> RunResult:
        """Process a document through the full agent pipeline.

        Phase 1 (extraction) and Phase 2 (chunking/decontextualization)
        are expected to be completed BEFORE calling this method.
        The state should already contain chunks and dense_summary.

        This method runs Phase 3 (agent DAG) and returns the result.

        Args:
            state: Pre-populated PipelineState (from Phase 1+2).
            document_id: Document identifier.
            document_title: Document title for agents.
            language: Document language.

        Returns:
            RunResult with final state and execution metadata.
        """
        if state is None:
            state = PipelineState()

        if document_id:
            state.document_id = document_id
        if document_title:
            state.document_title = document_title
        if language:
            state.language = language

        logger.info(
            "Starting document pipeline: doc_id=%s, %d chunks, %d agents planned",
            state.document_id,
            len(state.chunks),
            self._plan.total_agents,
        )

        runner = PipelineRunner(
            registry=self._registry,
            plan=self._plan,
            llm_factory=self._llm_factory,
            call_logger=self._call_logger,
            max_retries=self._max_retries,
            min_quality=self._min_quality,
            fail_fast=self._fail_fast,
        )

        result = await runner.run(state)

        logger.info(
            "Document pipeline complete: success=%s, %d LLM calls, %d tokens, %dms",
            result.success,
            state.total_llm_calls,
            state.total_tokens_used,
            result.duration_ms,
        )

        return result
