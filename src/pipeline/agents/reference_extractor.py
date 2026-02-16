# src/pipeline/agents/reference_extractor.py — v1
"""Cross-reference and citation extraction agent.

Operates on the full enriched text (not individual chunks) to extract
citations, footnotes, bibliography entries, and internal cross-references.
Runs in phase 1, step 1g (after content merger, before chunking).

See spec §12 for full documentation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ayextractor.core.models import DocumentStructure, Reference
from ayextractor.llm.models import Message
from ayextractor.pipeline.plugin_kit.base_agent import BaseAgent
from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput

if TYPE_CHECKING:
    from ayextractor.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "reference_extractor.txt"


class ReferenceExtractorInput(BaseModel):
    """Input schema for reference extractor."""

    enriched_text: str
    document_title: str
    language: str
    structure: DocumentStructure | None = None


class ReferenceExtractorOutput(BaseModel):
    """Output schema for reference extractor."""

    references: list[Reference] = Field(default_factory=list)
    confidence: float = 0.0


class ReferenceExtractorAgent(BaseAgent):
    """Extract cross-references and citations from full document text.

    Identifies inline citations, footnotes, bibliography entries,
    and internal cross-references (e.g. "see chapter 3").
    """

    def __init__(self) -> None:
        self._prompt_template: str | None = None

    @property
    def name(self) -> str:
        return "reference_extractor"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Extract citations, footnotes, bibliography, and internal cross-references"

    @property
    def input_schema(self) -> type[BaseModel]:
        return ReferenceExtractorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return ReferenceExtractorOutput

    @property
    def prompt_file(self) -> str | None:
        return str(_PROMPT_PATH)

    def _load_prompt(self) -> str:
        if self._prompt_template is None:
            self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_template

    def _build_structure_hint(self, structure: DocumentStructure | None) -> str:
        if structure is None:
            return "(not available)"
        parts: list[str] = []
        if structure.has_bibliography:
            parts.append(f"Bibliography detected at position {structure.bibliography_position or 'unknown'}")
        if structure.has_annexes:
            parts.append(f"Annexes: {len(structure.annexes)}")
        if structure.footnotes:
            parts.append(f"Footnotes detected: {len(structure.footnotes)}")
        return "\n".join(parts) if parts else "(no structural hints)"

    def _format_prompt(self, inp: ReferenceExtractorInput) -> str:
        template = self._load_prompt()
        # Truncate text for LLM context window (keep first ~8000 chars)
        text_excerpt = inp.enriched_text[:8000]
        if len(inp.enriched_text) > 8000:
            text_excerpt += "\n\n[... truncated ...]"
        return template.format(
            document_title=inp.document_title,
            language=inp.language,
            structure_hints=self._build_structure_hint(inp.structure),
            enriched_text=text_excerpt,
        )

    def _parse_response(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    def _build_reference(self, raw: dict[str, Any]) -> Reference | None:
        try:
            return Reference(
                type=raw["type"],
                text=str(raw["text"]),
                target=raw.get("target"),
                source_chunk_id=raw.get("source_chunk_id", "document"),
            )
        except (KeyError, ValueError) as exc:
            logger.debug("Skipping malformed reference: %s", exc)
            return None

    async def execute(self, state: object, llm: BaseLLMClient) -> AgentOutput:
        """Extract references from the full enriched text."""
        if isinstance(state, dict):
            inp = ReferenceExtractorInput(**state)
        elif isinstance(state, ReferenceExtractorInput):
            inp = state
        else:
            raise TypeError(
                f"Expected ReferenceExtractorInput or dict, got {type(state)}"
            )

        start_ms = time.monotonic_ns() // 1_000_000
        prompt = self._format_prompt(inp)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        response = await llm.complete(
            messages=[Message(role="user", content=prompt)],
            system=(
                "You are a bibliographic and cross-reference extraction specialist. "
                "Respond only with valid JSON."
            ),
            temperature=0.1,
        )
        elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms

        try:
            parsed = self._parse_response(response.content)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("ReferenceExtractor JSON parse failed: %s", exc)
            parsed = {"references": [], "confidence": 0.0}

        references: list[Reference] = []
        for raw_r in parsed.get("references", []):
            r = self._build_reference(raw_r)
            if r is not None:
                references.append(r)

        confidence = float(parsed.get("confidence", 0.5))

        return AgentOutput(
            data={
                "references": [r.model_dump() for r in references],
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
        refs = output.data.get("references", [])
        if not refs:
            return 0.5  # No refs can be valid for some documents
        return output.confidence
