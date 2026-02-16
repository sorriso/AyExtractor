# src/pipeline/agents/concept_extractor.py — v2
"""Per-chunk qualified triplet extraction agent.

Extracts (subject, predicate, object) triplets with qualifiers and
temporal scope from decontextualized chunks. Each triplet captures
a factual claim with provenance.

Runs in phase 3 (DAG), after decontextualization.
See spec §6.1 (Concept Extractor row) and §13.6.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ayextractor.core.models import Chunk, QualifiedTriplet, TemporalScope
from ayextractor.llm.models import Message
from ayextractor.pipeline.plugin_kit.base_agent import BaseAgent
from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput

if TYPE_CHECKING:
    from ayextractor.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "concept_extractor.txt"


class ConceptExtractorInput(BaseModel):
    """Input schema for concept extractor."""

    chunk: Chunk
    document_title: str
    language: str
    dense_summary: str = ""


class ConceptExtractorOutput(BaseModel):
    """Output schema for concept extractor."""

    triplets: list[QualifiedTriplet] = Field(default_factory=list)
    extraction_confidence: float = 0.0


class ConceptExtractorAgent(BaseAgent):
    """Extract qualified triplets from a single chunk via LLM.

    Each triplet includes subject, predicate, object, qualifiers,
    temporal scope, and a confidence score. Triplets are raw (not
    yet normalized); normalization is handled by graph/merger.py.
    """

    def __init__(self) -> None:
        self._prompt_template: str | None = None

    @property
    def name(self) -> str:
        return "concept_extractor"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Extract qualified (subject, predicate, object) triplets from chunks"

    @property
    def input_schema(self) -> type[BaseModel]:
        return ConceptExtractorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return ConceptExtractorOutput

    @property
    def dependencies(self) -> list[str]:
        return []  # decontextualizer runs in Phase 2 before the DAG

    @property
    def prompt_file(self) -> str | None:
        return str(_PROMPT_PATH)

    def _load_prompt(self) -> str:
        if self._prompt_template is None:
            self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_template

    def _format_prompt(self, inp: ConceptExtractorInput) -> str:
        template = self._load_prompt()
        return template.format(
            document_title=inp.document_title,
            language=inp.language,
            dense_summary=inp.dense_summary or "(not available)",
            chunk_id=inp.chunk.id,
            chunk_content=inp.chunk.content,
        )

    def _parse_response(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    def _build_triplet(
        self, raw: dict[str, Any], chunk_id: str
    ) -> QualifiedTriplet | None:
        """Build a QualifiedTriplet from raw LLM output dict."""
        try:
            ts_raw = raw.get("temporal_scope")
            temporal_scope = None
            if ts_raw and isinstance(ts_raw, dict):
                temporal_scope = TemporalScope(**ts_raw)

            return QualifiedTriplet(
                subject=str(raw["subject"]).strip(),
                predicate=str(raw["predicate"]).strip(),
                object=str(raw["object"]).strip(),
                source_chunk_id=chunk_id,
                confidence=float(raw.get("confidence", 0.5)),
                context_sentence=str(raw.get("context_sentence", "")),
                qualifiers=raw.get("qualifiers"),
                temporal_scope=temporal_scope,
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Skipping malformed triplet: %s", exc)
            return None

    async def execute(self, state: object, llm: BaseLLMClient) -> AgentOutput:
        """Extract triplets from a single chunk."""
        if isinstance(state, dict):
            inp = ConceptExtractorInput(**state)
        elif isinstance(state, ConceptExtractorInput):
            inp = state
        else:
            raise TypeError(
                f"Expected ConceptExtractorInput or dict, got {type(state)}"
            )

        start_ms = time.monotonic_ns() // 1_000_000
        prompt = self._format_prompt(inp)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        response = await llm.complete(
            messages=[Message(role="user", content=prompt)],
            system=(
                "You are a knowledge graph extraction specialist. "
                "Respond only with valid JSON."
            ),
            temperature=0.1,
        )
        elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms

        try:
            parsed = self._parse_response(response.content)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "ConceptExtractor JSON parse failed for %s: %s",
                inp.chunk.id,
                exc,
            )
            parsed = {"triplets": [], "extraction_confidence": 0.0}

        triplets: list[QualifiedTriplet] = []
        for raw_t in parsed.get("triplets", []):
            t = self._build_triplet(raw_t, inp.chunk.id)
            if t is not None:
                triplets.append(t)

        extraction_confidence = float(
            parsed.get("extraction_confidence", 0.5)
        )

        return AgentOutput(
            data={
                "triplets": [t.model_dump() for t in triplets],
                "extraction_confidence": extraction_confidence,
                "chunk_id": inp.chunk.id,
            },
            confidence=extraction_confidence,
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
        triplets = output.data.get("triplets", [])
        if not triplets:
            return 0.3  # Empty extraction is valid but low confidence
        avg_conf = sum(t.get("confidence", 0) for t in triplets) / len(triplets)
        return min(avg_conf, output.confidence)
