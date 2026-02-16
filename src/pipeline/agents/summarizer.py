# src/pipeline/agents/summarizer.py — v1
"""Refine incremental summarizer agent.

Maintains a running summary that grows with each new chunk.
Runs in phase 2c-ii, interleaved with the Decontextualizer.
See spec §6.1 (Summarizer row) and §28.4 for pipeline position.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ayextractor.core.models import Chunk
from ayextractor.llm.models import Message
from ayextractor.pipeline.plugin_kit.base_agent import BaseAgent
from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput

if TYPE_CHECKING:
    from ayextractor.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summarizer.txt"


class SummarizerInput(BaseModel):
    """Input schema for summarizer refine step."""

    chunk: Chunk
    current_summary: str
    document_title: str
    language: str


class SummarizerOutput(BaseModel):
    """Output schema for summarizer."""

    updated_summary: str
    new_information: list[str]
    confidence: float


class SummarizerAgent(BaseAgent):
    """Refine incremental summarizer.

    For each chunk, integrates new information into the running summary.
    The result is stored as chunk.context_summary (cumulative narrative).
    """

    def __init__(self):
        self._prompt_template: str | None = None

    @property
    def name(self) -> str:
        return "summarizer"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Incremental Refine summarizer — maintains running summary across chunks"

    @property
    def input_schema(self) -> type[BaseModel]:
        return SummarizerInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return SummarizerOutput

    @property
    def prompt_file(self) -> str | None:
        return str(_PROMPT_PATH)

    def _load_prompt(self) -> str:
        """Load and cache prompt template."""
        if self._prompt_template is None:
            self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_template

    def _format_prompt(self, inp: SummarizerInput) -> str:
        """Fill prompt template with input data."""
        template = self._load_prompt()
        return template.format(
            document_title=inp.document_title,
            language=inp.language,
            current_summary=inp.current_summary or "(empty — this is the first chunk)",
            chunk_content=inp.chunk.content,
        )

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse LLM JSON response, handling markdown fences."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    async def execute(self, state: object, llm: BaseLLMClient) -> AgentOutput:
        """Execute refine summarization on a single chunk.

        Called per-chunk by the orchestrator during the interleaved loop (2c-ii).
        """
        if isinstance(state, dict):
            inp = SummarizerInput(**state)
        elif isinstance(state, SummarizerInput):
            inp = state
        else:
            raise TypeError(f"Expected SummarizerInput or dict, got {type(state)}")

        start_ms = time.monotonic_ns() // 1_000_000
        prompt = self._format_prompt(inp)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        response = await llm.complete(
            messages=[Message(role="user", content=prompt)],
            system="You are a document summarization expert. Respond only with valid JSON.",
            temperature=0.2,
        )

        elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms

        try:
            parsed = self._parse_response(response.content)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Summarizer JSON parse failed for %s: %s", inp.chunk.id, exc)
            # Fallback: keep existing summary
            parsed = {
                "updated_summary": inp.current_summary or inp.chunk.content[:500],
                "new_information": [],
                "confidence": 0.3,
            }

        confidence = float(parsed.get("confidence", 0.5))

        return AgentOutput(
            data={
                "updated_summary": parsed.get("updated_summary", inp.current_summary),
                "new_information": parsed.get("new_information", []),
                "confidence": confidence,
                "chunk_id": inp.chunk.id,
            },
            confidence=confidence,
            metadata=AgentMetadata(
                agent_name=self.name,
                agent_version=self.version,
                execution_time_ms=elapsed_ms,
                llm_calls=1,
                tokens_used=response.input_tokens + response.output_tokens,
                prompt_hash=prompt_hash,
            ),
        )

    def apply_to_chunk(self, chunk: Chunk, summary: str) -> Chunk:
        """Store the cumulative summary into the chunk's context_summary.

        Args:
            chunk: The chunk to update.
            summary: The cumulative refine summary up to this chunk.

        Returns:
            The updated chunk.
        """
        chunk.context_summary = summary
        return chunk

    def validate_output(self, output: AgentOutput) -> float:
        """Validate summarizer output."""
        summary = output.data.get("updated_summary", "")
        if not summary.strip():
            return 0.0
        # Penalize very short summaries (< 50 chars) as likely incomplete
        if len(summary) < 50:
            return min(output.confidence, 0.5)
        return output.confidence
