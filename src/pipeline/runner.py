# src/pipeline/runner.py — v1
"""Pipeline runner — execute agent DAG with error handling and tracking.

Walks the ExecutionPlan stage by stage, calling each agent's execute()
method with the shared PipelineState and per-agent LLM client.

Supports:
  - Sequential execution within stages (async-ready for future concurrency)
  - Per-agent retry with exponential backoff
  - Quality threshold gating (agent must pass validate_output)
  - Full tracking via CallLogger

See spec §25.6 for full documentation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ayextractor.pipeline.dag_builder import ExecutionPlan
from ayextractor.pipeline.plugin_kit.models import AgentOutput
from ayextractor.pipeline.state import PipelineState

if TYPE_CHECKING:
    from ayextractor.llm.base_client import BaseLLMClient
    from ayextractor.pipeline.plugin_kit.base_agent import BaseAgent
    from ayextractor.pipeline.registry import AgentRegistry
    from ayextractor.tracking.call_logger import CallLogger

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2
DEFAULT_MIN_QUALITY = 0.2
DEFAULT_RETRY_DELAY_S = 1.0


@dataclass
class RunResult:
    """Result of a full pipeline run."""

    state: PipelineState
    success: bool = True
    failed_agents: list[str] = field(default_factory=list)
    skipped_agents: list[str] = field(default_factory=list)
    duration_ms: int = 0
    stages_completed: int = 0


class PipelineRunner:
    """Execute an agent DAG against a PipelineState.

    Args:
        registry: Loaded AgentRegistry with all agents.
        plan: ExecutionPlan from DAGBuilder.
        llm_factory: Callable(agent_name) -> BaseLLMClient.
        call_logger: Optional call logger for tracking.
        max_retries: Max retry count per agent.
        min_quality: Minimum validate_output score to accept.
        fail_fast: Stop pipeline on first agent failure.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        plan: ExecutionPlan,
        llm_factory: Any = None,
        call_logger: CallLogger | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_quality: float = DEFAULT_MIN_QUALITY,
        fail_fast: bool = False,
    ) -> None:
        self._registry = registry
        self._plan = plan
        self._llm_factory = llm_factory
        self._call_logger = call_logger
        self._max_retries = max_retries
        self._min_quality = min_quality
        self._fail_fast = fail_fast

    async def run(self, state: PipelineState) -> RunResult:
        """Execute all agents in DAG order.

        Args:
            state: Mutable pipeline state to pass through agents.

        Returns:
            RunResult with final state and execution metadata.
        """
        start_ns = time.monotonic_ns()
        result = RunResult(state=state)

        for stage_idx, stage in enumerate(self._plan.stages):
            logger.info(
                "Stage %d/%d: executing %s",
                stage_idx + 1,
                len(self._plan.stages),
                stage,
            )

            for agent_name in stage:
                agent = self._registry.get(agent_name)
                if agent is None:
                    logger.warning("Agent '%s' not in registry, skipping", agent_name)
                    result.skipped_agents.append(agent_name)
                    continue

                success = await self._run_agent(agent, state, result)

                if not success and self._fail_fast:
                    logger.error(
                        "Fail-fast: stopping pipeline after '%s' failure",
                        agent_name,
                    )
                    result.success = False
                    result.duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
                    result.stages_completed = stage_idx
                    return result

            result.stages_completed = stage_idx + 1

        result.duration_ms = (time.monotonic_ns() - start_ns) // 1_000_000
        result.success = len(result.failed_agents) == 0

        logger.info(
            "Pipeline complete: %d agents, %d failed, %d skipped, %dms",
            self._plan.total_agents,
            len(result.failed_agents),
            len(result.skipped_agents),
            result.duration_ms,
        )
        return result

    async def _run_agent(
        self,
        agent: BaseAgent,
        state: PipelineState,
        result: RunResult,
    ) -> bool:
        """Run a single agent with retry logic.

        Returns True if agent succeeded, False if all retries exhausted.
        """
        for attempt in range(self._max_retries + 1):
            try:
                llm = self._get_llm(agent.name)
                logger.debug(
                    "Running agent '%s' (attempt %d/%d)",
                    agent.name,
                    attempt + 1,
                    self._max_retries + 1,
                )

                output = await agent.execute(state, llm)

                # Quality gate
                quality = agent.validate_output(output)
                if quality < self._min_quality:
                    logger.warning(
                        "Agent '%s' quality %.2f below threshold %.2f (attempt %d)",
                        agent.name,
                        quality,
                        self._min_quality,
                        attempt + 1,
                    )
                    if attempt < self._max_retries:
                        await asyncio.sleep(DEFAULT_RETRY_DELAY_S * (attempt + 1))
                        continue
                    # Accept low-quality on final attempt with warning
                    output.warnings.append(
                        f"Quality {quality:.2f} below threshold {self._min_quality:.2f}"
                    )

                # Record output
                state.record_agent_output(agent.name, output)

                logger.info(
                    "Agent '%s' completed: confidence=%.2f, tokens=%d, time=%dms",
                    agent.name,
                    output.confidence,
                    output.metadata.tokens_used,
                    output.metadata.execution_time_ms,
                )
                return True

            except Exception as exc:
                logger.error(
                    "Agent '%s' failed (attempt %d): %s",
                    agent.name,
                    attempt + 1,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(DEFAULT_RETRY_DELAY_S * (attempt + 1))
                else:
                    state.errors.append(f"{agent.name}: {exc}")
                    result.failed_agents.append(agent.name)
                    return False

        return False

    def _get_llm(self, agent_name: str) -> BaseLLMClient:
        """Get LLM client for a specific agent."""
        if self._llm_factory is not None:
            if callable(self._llm_factory):
                return self._llm_factory(agent_name)
            return self._llm_factory
        raise RuntimeError(
            f"No LLM factory configured for agent '{agent_name}'"
        )
