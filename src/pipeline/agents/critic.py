# src/pipeline/agents/critic.py — v1
"""Critic agent — quality assessment and self-correction of extraction pipeline.

Reviews the full extraction output (triplets, communities, synthesis) and
flags potential issues: low-confidence triplets, missing entities,
contradictions, orphan nodes, and coverage gaps.

Runs LAST in the pipeline. See spec §6.1 (Critic row) and §14.2.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ayextractor.llm.models import Message
from ayextractor.pipeline.plugin_kit.base_agent import BaseAgent
from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput

if TYPE_CHECKING:
    from ayextractor.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "critic.txt"


class CriticInput(BaseModel):
    """Input schema for critic."""

    document_title: str
    dense_summary: str
    synthesis: str
    graph_stats: dict[str, Any] = Field(default_factory=dict)
    sample_triplets: list[dict[str, Any]] = Field(default_factory=list)
    community_count: int = 0
    entity_count: int = 0
    language: str = "en"


class QualityIssue(BaseModel):
    """A single quality issue detected by the critic."""

    severity: str  # "low", "medium", "high"
    category: str  # e.g. "coverage_gap", "contradiction", "low_confidence"
    description: str
    affected_entities: list[str] = Field(default_factory=list)
    suggestion: str = ""


class CriticOutput(BaseModel):
    """Output schema for critic."""

    overall_quality: float  # 0.0 - 1.0
    issues: list[QualityIssue] = Field(default_factory=list)
    summary: str = ""


class CriticAgent(BaseAgent):
    """Assess extraction quality and flag issues."""

    def __init__(self) -> None:
        self._prompt_template: str | None = None

    @property
    def name(self) -> str:
        return "critic"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Assess extraction quality and flag issues"

    @property
    def input_schema(self) -> type[BaseModel]:
        return CriticInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return CriticOutput

    @property
    def dependencies(self) -> list[str]:
        return ["synthesizer"]

    @property
    def prompt_file(self) -> str | None:
        return str(_PROMPT_PATH)

    def _load_prompt(self) -> str:
        if self._prompt_template is None:
            self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_template

    def _format_prompt(self, inp: CriticInput) -> str:
        template = self._load_prompt()
        triplets_text = "\n".join(
            f"- ({t.get('subject', '?')}, {t.get('predicate', '?')}, {t.get('object', '?')}) "
            f"[conf={t.get('confidence', 0):.2f}]"
            for t in inp.sample_triplets[:15]
        ) or "(no triplets)"

        return template.format(
            document_title=inp.document_title,
            dense_summary=inp.dense_summary[:500],
            synthesis=inp.synthesis[:800],
            graph_stats=json.dumps(inp.graph_stats, indent=2),
            sample_triplets=triplets_text,
            community_count=inp.community_count,
            entity_count=inp.entity_count,
            language=inp.language,
        )

    def _parse_response(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    def _build_issue(self, raw: dict[str, Any]) -> QualityIssue | None:
        try:
            return QualityIssue(
                severity=raw.get("severity", "low"),
                category=raw.get("category", "unknown"),
                description=raw.get("description", ""),
                affected_entities=raw.get("affected_entities", []),
                suggestion=raw.get("suggestion", ""),
            )
        except (KeyError, ValueError) as exc:
            logger.debug("Skipping malformed issue: %s", exc)
            return None

    async def execute(self, state: object, llm: BaseLLMClient) -> AgentOutput:
        if isinstance(state, dict):
            inp = CriticInput(**state)
        elif isinstance(state, CriticInput):
            inp = state
        else:
            raise TypeError(f"Expected CriticInput or dict, got {type(state)}")

        start_ms = time.monotonic_ns() // 1_000_000
        prompt = self._format_prompt(inp)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        response = await llm.complete(
            messages=[Message(role="user", content=prompt)],
            system=(
                "You are a knowledge graph quality analyst. "
                "Critically assess the extraction quality. "
                "Respond only with valid JSON."
            ),
            temperature=0.2,
        )
        elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms

        try:
            parsed = self._parse_response(response.content)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Critic JSON parse failed: %s", exc)
            parsed = {
                "overall_quality": 0.5,
                "issues": [],
                "summary": "Quality assessment could not be completed.",
            }

        issues: list[QualityIssue] = []
        for raw_issue in parsed.get("issues", []):
            issue = self._build_issue(raw_issue)
            if issue:
                issues.append(issue)

        overall_quality = float(parsed.get("overall_quality", 0.5))

        return AgentOutput(
            data={
                "overall_quality": overall_quality,
                "issues": [i.model_dump() for i in issues],
                "summary": parsed.get("summary", ""),
                "issue_count": len(issues),
                "high_severity_count": sum(1 for i in issues if i.severity == "high"),
            },
            confidence=overall_quality,
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
        return output.data.get("overall_quality", 0.5)
