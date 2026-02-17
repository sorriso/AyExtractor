# tests/unit/pipeline/test_runner.py — v1
"""Tests for pipeline/runner.py — PipelineRunner execution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ayextractor.pipeline.dag_builder import ExecutionPlan
from ayextractor.pipeline.plugin_kit.base_agent import BaseAgent
from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
from ayextractor.pipeline.registry import AgentRegistry
from ayextractor.pipeline.runner import PipelineRunner, RunResult
from ayextractor.pipeline.state import PipelineState


# --- Helpers ---

def _make_output(name: str = "test", confidence: float = 0.9) -> AgentOutput:
    return AgentOutput(
        data={"result": f"from_{name}"},
        confidence=confidence,
        metadata=AgentMetadata(
            agent_name=name,
            agent_version="1.0.0",
            execution_time_ms=50,
            llm_calls=1,
            tokens_used=100,
        ),
    )


class StubAgent(BaseAgent):
    """Test agent that returns a predefined output."""

    def __init__(self, agent_name: str, confidence: float = 0.9, fail: bool = False):
        self._name = agent_name
        self._confidence = confidence
        self._fail = fail

    @property
    def name(self): return self._name
    @property
    def version(self): return "1.0.0"
    @property
    def description(self): return f"Stub {self._name}"
    @property
    def input_schema(self): return type("M", (), {})
    @property
    def output_schema(self): return type("M", (), {})
    @property
    def dependencies(self): return []

    async def execute(self, state, llm):
        if self._fail:
            raise RuntimeError(f"{self._name} failed")
        return _make_output(self._name, self._confidence)

    def validate_output(self, output):
        return output.confidence


def _build_registry(*agents: StubAgent) -> AgentRegistry:
    reg = AgentRegistry()
    for a in agents:
        reg.register(a)
    return reg


# --- Tests ---

class TestPipelineRunner:
    @pytest.mark.asyncio
    async def test_single_agent_success(self):
        reg = _build_registry(StubAgent("agent_a"))
        plan = ExecutionPlan(stages=[["agent_a"]], total_agents=1)
        runner = PipelineRunner(
            registry=reg, plan=plan, llm_factory=lambda n: MagicMock(),
        )
        state = PipelineState()
        result = await runner.run(state)

        assert result.success is True
        assert result.failed_agents == []
        assert "agent_a" in state.agent_outputs
        assert state.total_llm_calls == 1
        assert state.total_tokens_used == 100

    @pytest.mark.asyncio
    async def test_multi_stage_execution(self):
        reg = _build_registry(
            StubAgent("a"), StubAgent("b"), StubAgent("c"),
        )
        plan = ExecutionPlan(stages=[["a"], ["b"], ["c"]], total_agents=3)
        runner = PipelineRunner(
            registry=reg, plan=plan, llm_factory=lambda n: MagicMock(),
        )
        state = PipelineState()
        result = await runner.run(state)

        assert result.success is True
        assert state.total_llm_calls == 3
        assert result.stages_completed == 3

    @pytest.mark.asyncio
    async def test_parallel_stage(self):
        reg = _build_registry(StubAgent("a"), StubAgent("b"))
        plan = ExecutionPlan(stages=[["a", "b"]], total_agents=2)
        runner = PipelineRunner(
            registry=reg, plan=plan, llm_factory=lambda n: MagicMock(),
        )
        state = PipelineState()
        result = await runner.run(state)

        assert result.success is True
        assert len(state.agent_outputs) == 2

    @pytest.mark.asyncio
    async def test_agent_failure_recorded(self):
        reg = _build_registry(StubAgent("ok"), StubAgent("bad", fail=True))
        plan = ExecutionPlan(stages=[["ok", "bad"]], total_agents=2)
        runner = PipelineRunner(
            registry=reg, plan=plan, llm_factory=lambda n: MagicMock(),
            max_retries=0,
        )
        state = PipelineState()
        result = await runner.run(state)

        assert result.success is False
        assert "bad" in result.failed_agents
        assert "ok" in state.agent_outputs
        assert len(state.errors) == 1

    @pytest.mark.asyncio
    async def test_fail_fast_stops_pipeline(self):
        reg = _build_registry(
            StubAgent("a", fail=True), StubAgent("b"),
        )
        plan = ExecutionPlan(stages=[["a"], ["b"]], total_agents=2)
        runner = PipelineRunner(
            registry=reg, plan=plan, llm_factory=lambda n: MagicMock(),
            max_retries=0, fail_fast=True,
        )
        state = PipelineState()
        result = await runner.run(state)

        assert result.success is False
        assert "a" in result.failed_agents
        # "b" should not have run
        assert "b" not in state.agent_outputs

    @pytest.mark.asyncio
    async def test_missing_agent_skipped(self):
        reg = AgentRegistry()  # empty registry
        plan = ExecutionPlan(stages=[["missing"]], total_agents=1)
        runner = PipelineRunner(
            registry=reg, plan=plan, llm_factory=lambda n: MagicMock(),
        )
        state = PipelineState()
        result = await runner.run(state)

        assert "missing" in result.skipped_agents

    @pytest.mark.asyncio
    async def test_quality_gate_low_quality_accepted_on_final_retry(self):
        """Agent with quality below threshold accepted on last attempt with warning."""
        reg = _build_registry(StubAgent("low_q", confidence=0.05))
        plan = ExecutionPlan(stages=[["low_q"]], total_agents=1)
        runner = PipelineRunner(
            registry=reg, plan=plan, llm_factory=lambda n: MagicMock(),
            min_quality=0.5, max_retries=0,
        )
        state = PipelineState()
        result = await runner.run(state)

        assert "low_q" in state.agent_outputs
        assert len(state.agent_outputs["low_q"].warnings) > 0

    @pytest.mark.asyncio
    async def test_no_llm_factory_raises(self):
        reg = _build_registry(StubAgent("a"))
        plan = ExecutionPlan(stages=[["a"]], total_agents=1)
        runner = PipelineRunner(registry=reg, plan=plan, llm_factory=None)
        state = PipelineState()
        result = await runner.run(state)
        # Should fail because no LLM factory
        assert result.success is False
        assert "a" in result.failed_agents

    @pytest.mark.asyncio
    async def test_duration_tracked(self):
        reg = _build_registry(StubAgent("a"))
        plan = ExecutionPlan(stages=[["a"]], total_agents=1)
        runner = PipelineRunner(
            registry=reg, plan=plan, llm_factory=lambda n: MagicMock(),
        )
        state = PipelineState()
        result = await runner.run(state)

        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_llm_factory_as_direct_client(self):
        """llm_factory can be a single client instance (non-callable)."""
        mock_llm = MagicMock()
        reg = _build_registry(StubAgent("a"))
        plan = ExecutionPlan(stages=[["a"]], total_agents=1)
        runner = PipelineRunner(
            registry=reg, plan=plan, llm_factory=mock_llm,
        )
        state = PipelineState()
        result = await runner.run(state)
        assert result.success is True
