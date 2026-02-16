# src/pipeline/agents/synthesizer.py — v1
"""Synthesizer agent — final document synthesis from graph, communities, profiles.

Produces a structured, human-readable summary of the entire knowledge graph.
Operates AFTER all extraction, normalization, and community detection is done.

See spec §6.1 (Synthesizer row) and §14.1 for full documentation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ayextractor.graph.layers.models import CommunitySummary
from ayextractor.llm.models import Message
from ayextractor.pipeline.plugin_kit.base_agent import BaseAgent
from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput

if TYPE_CHECKING:
    from ayextractor.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "synthesizer.txt"


class SynthesizerInput(BaseModel):
    """Input schema for synthesizer."""

    document_title: str
    dense_summary: str
    community_summaries: list[dict[str, Any]] = Field(default_factory=list)
    top_entities: list[dict[str, Any]] = Field(default_factory=list)
    graph_stats: dict[str, Any] = Field(default_factory=dict)
    language: str = "en"


class SynthesizerOutput(BaseModel):
    """Output schema for synthesizer."""

    synthesis: str
    key_findings: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class SynthesizerAgent(BaseAgent):
    """Produce a structured synthesis of the entire document knowledge graph.

    Combines dense_summary, community summaries, and top entity profiles
    into a coherent, thematic overview.
    """

    def __init__(self) -> None:
        self._prompt_template: str | None = None

    @property
    def name(self) -> str:
        return "synthesizer"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Produce a structured synthesis from graph, communities, and profiles"

    @property
    def input_schema(self) -> type[BaseModel]:
        return SynthesizerInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return SynthesizerOutput

    @property
    def dependencies(self) -> list[str]:
        return ["densifier", "community_summarizer", "profile_generator"]

    @property
    def prompt_file(self) -> str | None:
        return str(_PROMPT_PATH)

    def _load_prompt(self) -> str:
        if self._prompt_template is None:
            self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_template

    def _format_prompt(self, inp: SynthesizerInput) -> str:
        template = self._load_prompt()
        comm_text = "\n".join(
            f"- **{c.get('title', 'Untitled')}**: {c.get('summary', '')}"
            for c in inp.community_summaries[:10]
        ) or "(no communities)"

        entities_text = "\n".join(
            f"- {e.get('canonical_name', '?')}: {e.get('profile_text', '')[:120]}"
            for e in inp.top_entities[:10]
        ) or "(no entity profiles)"

        stats_text = json.dumps(inp.graph_stats, indent=2) if inp.graph_stats else "{}"

        return template.format(
            document_title=inp.document_title,
            dense_summary=inp.dense_summary,
            community_summaries=comm_text,
            top_entities=entities_text,
            graph_stats=stats_text,
            language=inp.language,
        )

    def _parse_response(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    async def execute(self, state: object, llm: BaseLLMClient) -> AgentOutput:
        if isinstance(state, dict):
            inp = SynthesizerInput(**state)
        elif isinstance(state, SynthesizerInput):
            inp = state
        else:
            raise TypeError(f"Expected SynthesizerInput or dict, got {type(state)}")

        start_ms = time.monotonic_ns() // 1_000_000
        prompt = self._format_prompt(inp)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        response = await llm.complete(
            messages=[Message(role="user", content=prompt)],
            system=(
                "You are a document synthesis expert. "
                "Produce a comprehensive, structured overview. "
                "Respond only with valid JSON."
            ),
            temperature=0.4,
        )
        elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms

        try:
            parsed = self._parse_response(response.content)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Synthesizer JSON parse failed: %s", exc)
            parsed = {
                "synthesis": f"Analysis of '{inp.document_title}' with {len(inp.community_summaries)} themes identified.",
                "key_findings": [],
                "confidence": 0.3,
            }

        confidence = float(parsed.get("confidence", 0.7))

        return AgentOutput(
            data={
                "synthesis": parsed.get("synthesis", ""),
                "key_findings": parsed.get("key_findings", []),
                "confidence": confidence,
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

    def validate_output(self, output: AgentOutput) -> float:
        synthesis = output.data.get("synthesis", "")
        if not synthesis.strip():
            return 0.0
        # Longer synthesis = higher confidence (crude heuristic)
        length_factor = min(len(synthesis) / 500, 1.0)
        return min(output.confidence * length_factor, 1.0)
