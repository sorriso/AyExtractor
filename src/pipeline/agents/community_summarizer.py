# src/pipeline/agents/community_summarizer.py — v1
"""Community summarizer agent — LLM-generated summary per community.

For each detected community, generates a concise textual summary
from member entities, their relations, and representative source sentences.

Runs in phase 3, after community detection + integration.
See spec §13.10.3 for full documentation.
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

from ayextractor.graph.layers.models import Community, CommunitySummary
from ayextractor.llm.models import Message
from ayextractor.pipeline.plugin_kit.base_agent import BaseAgent
from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput

if TYPE_CHECKING:
    from ayextractor.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "community_summarizer.txt"


class CommunitySummarizerInput(BaseModel):
    """Input schema for community summarizer."""

    community: Community
    graph_data: dict[str, Any]  # serialized graph node/edge data
    language: str = "en"


class CommunitySummarizerOutput(BaseModel):
    """Output schema for community summarizer."""

    summaries: list[CommunitySummary] = Field(default_factory=list)


class CommunitySummarizerAgent(BaseAgent):
    """Generate LLM-based summaries for detected communities."""

    def __init__(self) -> None:
        self._prompt_template: str | None = None

    @property
    def name(self) -> str:
        return "community_summarizer"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Generate concise textual summaries for graph communities"

    @property
    def input_schema(self) -> type[BaseModel]:
        return CommunitySummarizerInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return CommunitySummarizerOutput

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

    def _extract_community_context(
        self, community: Community, graph_data: dict[str, Any],
    ) -> str:
        """Build textual context for a community from graph data."""
        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", [])

        members_info: list[str] = []
        for member in community.members[:20]:  # cap for prompt length
            node = nodes.get(member, {})
            entity_type = node.get("entity_type", "unknown")
            members_info.append(f"- {member} ({entity_type})")

        relations_info: list[str] = []
        member_set = set(community.members)
        for edge in edges:
            src, tgt = edge.get("source", ""), edge.get("target", "")
            if src in member_set and tgt in member_set:
                rel = edge.get("relation_type", "related_to")
                relations_info.append(f"- {src} → {rel} → {tgt}")
            if len(relations_info) >= 15:
                break

        return (
            f"Members ({len(community.members)}):\n"
            + "\n".join(members_info)
            + f"\n\nIntra-community relations ({len(relations_info)}):\n"
            + "\n".join(relations_info) if relations_info else "(none detected)"
        )

    def _format_prompt(self, community: Community, context: str, language: str) -> str:
        template = self._load_prompt()
        return template.format(
            community_id=community.community_id,
            member_count=len(community.members),
            context=context,
            language=language,
        )

    def _parse_response(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    async def summarize_community(
        self,
        community: Community,
        graph: nx.Graph,
        llm: BaseLLMClient,
        language: str = "en",
    ) -> CommunitySummary:
        """Generate summary for a single community.

        Convenience method for direct usage outside the agent framework.
        """
        graph_data = _graph_to_dict(graph)
        context = self._extract_community_context(community, graph_data)
        prompt = self._format_prompt(community, context, language)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        response = await llm.complete(
            messages=[Message(role="user", content=prompt)],
            system="You are a knowledge graph analyst. Respond only with valid JSON.",
            temperature=0.3,
        )

        try:
            parsed = self._parse_response(response.content)
        except (json.JSONDecodeError, KeyError):
            parsed = {
                "title": f"Community {community.community_id}",
                "summary": f"Community of {len(community.members)} entities.",
                "key_entities": community.members[:5],
            }

        return CommunitySummary(
            community_id=community.community_id,
            level=community.level,
            title=parsed.get("title", community.community_id),
            summary=parsed.get("summary", ""),
            key_entities=parsed.get("key_entities", community.members[:5]),
            chunk_coverage=community.chunk_coverage,
            member_count=len(community.members),
        )

    async def execute(self, state: object, llm: BaseLLMClient) -> AgentOutput:
        """Execute community summarization.

        State should be a CommunitySummarizerInput or dict.
        """
        if isinstance(state, dict):
            inp = CommunitySummarizerInput(**state)
        elif isinstance(state, CommunitySummarizerInput):
            inp = state
        else:
            raise TypeError(
                f"Expected CommunitySummarizerInput or dict, got {type(state)}"
            )

        start_ms = time.monotonic_ns() // 1_000_000
        context = self._extract_community_context(inp.community, inp.graph_data)
        prompt = self._format_prompt(inp.community, context, inp.language)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        response = await llm.complete(
            messages=[Message(role="user", content=prompt)],
            system="You are a knowledge graph analyst. Respond only with valid JSON.",
            temperature=0.3,
        )
        elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms

        try:
            parsed = self._parse_response(response.content)
        except (json.JSONDecodeError, KeyError):
            parsed = {
                "title": inp.community.community_id,
                "summary": f"Community of {len(inp.community.members)} entities.",
                "key_entities": inp.community.members[:5],
                "confidence": 0.3,
            }

        summary = CommunitySummary(
            community_id=inp.community.community_id,
            level=inp.community.level,
            title=parsed.get("title", ""),
            summary=parsed.get("summary", ""),
            key_entities=parsed.get("key_entities", []),
            chunk_coverage=inp.community.chunk_coverage,
            member_count=len(inp.community.members),
        )

        confidence = float(parsed.get("confidence", 0.7))

        return AgentOutput(
            data={"summary": summary.model_dump(), "confidence": confidence},
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
        summary = output.data.get("summary", {})
        if not summary.get("summary", "").strip():
            return 0.0
        return output.confidence


def _graph_to_dict(graph: nx.Graph) -> dict[str, Any]:
    """Convert NetworkX graph to a simple dict for prompt building."""
    nodes = {n: dict(d) for n, d in graph.nodes(data=True)}
    edges = []
    for u, v, d in graph.edges(data=True):
        edge = dict(d)
        edge["source"] = u
        edge["target"] = v
        edges.append(edge)
    return {"nodes": nodes, "edges": edges}
