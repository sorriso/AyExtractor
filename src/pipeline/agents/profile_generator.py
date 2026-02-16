# src/pipeline/agents/profile_generator.py — v1
"""Entity and relation profile generator agent.

Generates concise textual profiles for L2 entities and key relations.
Profiles serve as first-level RAG context — more compact than chunks.

See spec §13.11.2 for full documentation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
from pydantic import BaseModel, Field

from ayextractor.graph.profiles.models import EntityProfile, RelationProfile
from ayextractor.llm.models import Message
from ayextractor.pipeline.plugin_kit.base_agent import BaseAgent
from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput

if TYPE_CHECKING:
    from ayextractor.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "profile_generator.txt"

DEFAULT_MIN_RELATIONS = 2


class ProfileGeneratorInput(BaseModel):
    """Input schema for profile generator."""

    entity_name: str
    graph_data: dict[str, Any]
    language: str = "en"


class ProfileGeneratorOutput(BaseModel):
    """Output schema for profile generator."""

    entity_profiles: list[EntityProfile] = Field(default_factory=list)
    relation_profiles: list[RelationProfile] = Field(default_factory=list)


class ProfileGeneratorAgent(BaseAgent):
    """Generate textual profiles for entities and relations via LLM."""

    def __init__(self, min_relations: int = DEFAULT_MIN_RELATIONS) -> None:
        self._min_relations = min_relations
        self._prompt_template: str | None = None

    @property
    def name(self) -> str:
        return "profile_generator"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Generate concise textual profiles for entities and key relations"

    @property
    def input_schema(self) -> type[BaseModel]:
        return ProfileGeneratorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return ProfileGeneratorOutput

    @property
    def dependencies(self) -> list[str]:
        return ["concept_extractor"]

    @property
    def prompt_file(self) -> str | None:
        return str(_PROMPT_PATH)

    def _load_prompt(self) -> str:
        if self._prompt_template is None:
            self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_template

    def _extract_entity_context(
        self, entity_name: str, graph_data: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Extract node data and relation descriptions for an entity."""
        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", [])
        node = nodes.get(entity_name, {})

        relations: list[str] = []
        for edge in edges:
            src, tgt = edge.get("source", ""), edge.get("target", "")
            rel = edge.get("relation_type", "related_to")
            if src == entity_name:
                relations.append(f"{entity_name} → {rel} → {tgt}")
            elif tgt == entity_name:
                relations.append(f"{src} → {rel} → {entity_name}")

        return node, relations

    def _format_prompt(
        self, entity_name: str, node: dict, relations: list[str], language: str,
    ) -> str:
        template = self._load_prompt()
        rels_text = "\n".join(f"- {r}" for r in relations[:15]) or "(none)"
        return template.format(
            entity_name=entity_name,
            entity_type=node.get("entity_type", "unknown"),
            aliases=", ".join(node.get("aliases", [])) or "(none)",
            relations=rels_text,
            language=language,
        )

    def _parse_response(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    async def generate_entity_profile(
        self,
        entity_name: str,
        graph: nx.Graph,
        llm: BaseLLMClient,
        language: str = "en",
    ) -> EntityProfile | None:
        """Generate a profile for a single entity.

        Returns None if entity has fewer relations than min_relations.
        """
        from ayextractor.pipeline.agents.community_summarizer import _graph_to_dict

        graph_data = _graph_to_dict(graph)
        node, relations = self._extract_entity_context(entity_name, graph_data)

        if len(relations) < self._min_relations:
            return None

        prompt = self._format_prompt(entity_name, node, relations, language)

        response = await llm.complete(
            messages=[Message(role="user", content=prompt)],
            system="You are a knowledge graph profiling expert. Respond only with valid JSON.",
            temperature=0.3,
        )

        try:
            parsed = self._parse_response(response.content)
        except (json.JSONDecodeError, KeyError):
            parsed = {
                "profile_text": f"{entity_name} is a {node.get('entity_type', 'concept')}.",
                "key_relations": [r.split(" → ")[1] + " " + r.split(" → ")[2]
                                  for r in relations[:5] if " → " in r],
            }

        return EntityProfile(
            canonical_name=entity_name,
            entity_type=node.get("entity_type", "concept"),
            profile_text=parsed.get("profile_text", ""),
            key_relations=parsed.get("key_relations", []),
            community_id=node.get("community_id"),
        )

    async def execute(self, state: object, llm: BaseLLMClient) -> AgentOutput:
        """Execute profile generation for a single entity."""
        if isinstance(state, dict):
            inp = ProfileGeneratorInput(**state)
        elif isinstance(state, ProfileGeneratorInput):
            inp = state
        else:
            raise TypeError(
                f"Expected ProfileGeneratorInput or dict, got {type(state)}"
            )

        start_ms = time.monotonic_ns() // 1_000_000
        node, relations = self._extract_entity_context(inp.entity_name, inp.graph_data)

        if len(relations) < self._min_relations:
            elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms
            return AgentOutput(
                data={"entity_profile": None, "skipped": True, "reason": "too_few_relations"},
                confidence=1.0,
                metadata=AgentMetadata(
                    agent_name=self.name, agent_version=self.version,
                    execution_time_ms=elapsed_ms, llm_calls=0, tokens_used=0,
                ),
            )

        prompt = self._format_prompt(inp.entity_name, node, relations, inp.language)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        response = await llm.complete(
            messages=[Message(role="user", content=prompt)],
            system="You are a knowledge graph profiling expert. Respond only with valid JSON.",
            temperature=0.3,
        )
        elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms

        try:
            parsed = self._parse_response(response.content)
        except (json.JSONDecodeError, KeyError):
            parsed = {
                "profile_text": f"{inp.entity_name} is mentioned in the document.",
                "key_relations": [],
                "confidence": 0.3,
            }

        profile = EntityProfile(
            canonical_name=inp.entity_name,
            entity_type=node.get("entity_type", "concept"),
            profile_text=parsed.get("profile_text", ""),
            key_relations=parsed.get("key_relations", []),
            community_id=node.get("community_id"),
        )

        confidence = float(parsed.get("confidence", 0.7))

        return AgentOutput(
            data={"entity_profile": profile.model_dump(), "confidence": confidence},
            confidence=confidence,
            metadata=AgentMetadata(
                agent_name=self.name, agent_version=self.version,
                execution_time_ms=elapsed_ms, llm_calls=1,
                tokens_used=response.input_tokens + response.output_tokens,
                prompt_hash=prompt_hash,
            ),
        )

    def validate_output(self, output: AgentOutput) -> float:
        if output.data.get("skipped"):
            return 1.0
        profile = output.data.get("entity_profile", {})
        if not profile or not profile.get("profile_text", "").strip():
            return 0.0
        return output.confidence
