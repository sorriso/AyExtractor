# tests/unit/pipeline/test_document_pipeline.py — v1
"""Tests for pipeline/document_pipeline.py — DocumentPipeline orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ayextractor.pipeline.document_pipeline import DocumentPipeline
from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
from ayextractor.pipeline.state import PipelineState


def _mock_agent_output(name: str) -> AgentOutput:
    return AgentOutput(
        data={"result": name},
        confidence=0.9,
        metadata=AgentMetadata(
            agent_name=name, agent_version="1.0.0",
            execution_time_ms=50, llm_calls=1, tokens_used=100,
        ),
    )


class TestDocumentPipeline:
    def test_creation_loads_registry(self):
        """Pipeline should load agents from registry at init."""
        pipeline = DocumentPipeline(
            llm_factory=lambda n: MagicMock(),
        )
        assert len(pipeline.registry.agent_names) >= 2
        assert len(pipeline.execution_order) >= 2

    def test_creation_with_disabled_agents(self):
        pipeline = DocumentPipeline(
            llm_factory=lambda n: MagicMock(),
            disabled_agents={"critic"},
        )
        assert "critic" not in pipeline.registry.agent_names

    def test_execution_order_respects_deps(self):
        pipeline = DocumentPipeline(
            llm_factory=lambda n: MagicMock(),
        )
        order = pipeline.execution_order
        # Densifier depends on summarizer → must come after
        if "summarizer" in order and "densifier" in order:
            assert order.index("summarizer") < order.index("densifier")

    @pytest.mark.asyncio
    async def test_process_creates_state_if_none(self):
        """Should create new state if not provided."""
        pipeline = DocumentPipeline(
            llm_factory=lambda n: MagicMock(),
            disabled_agents=set(
                pipeline._registry.agent_names
                for pipeline in [DocumentPipeline(llm_factory=lambda n: MagicMock())]
            ).pop() if False else set(),  # noqa: keep all agents
        )
        # Patch runner to avoid actual agent execution
        with patch.object(
            pipeline, "_build_plan",
        ), patch(
            "ayextractor.pipeline.document_pipeline.PipelineRunner",
        ) as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run = AsyncMock(return_value=MagicMock(
                success=True, state=PipelineState(), duration_ms=100,
                failed_agents=[], skipped_agents=[],
            ))
            mock_runner_cls.return_value = mock_runner

            result = await pipeline.process(
                document_id="doc1",
                document_title="Test",
            )
            mock_runner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_passes_state_through(self):
        """State should flow through the pipeline."""
        state = PipelineState(document_id="existing_doc")
        pipeline = DocumentPipeline(
            llm_factory=lambda n: MagicMock(),
        )
        with patch(
            "ayextractor.pipeline.document_pipeline.PipelineRunner",
        ) as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.state = state
            mock_runner.run = AsyncMock(return_value=mock_result)
            mock_runner_cls.return_value = mock_runner

            result = await pipeline.process(
                state=state, document_title="Updated",
            )
            # State should have been updated
            passed_state = mock_runner.run.call_args[0][0]
            assert passed_state.document_id == "existing_doc"
            assert passed_state.document_title == "Updated"

    @pytest.mark.asyncio
    async def test_process_sets_document_metadata(self):
        pipeline = DocumentPipeline(
            llm_factory=lambda n: MagicMock(),
        )
        with patch(
            "ayextractor.pipeline.document_pipeline.PipelineRunner",
        ) as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run = AsyncMock(return_value=MagicMock(
                success=True, duration_ms=50,
            ))
            mock_runner_cls.return_value = mock_runner

            await pipeline.process(
                document_id="d1", document_title="Title", language="fr",
            )
            passed_state = mock_runner.run.call_args[0][0]
            assert passed_state.document_id == "d1"
            assert passed_state.document_title == "Title"
            assert passed_state.language == "fr"
