# src/pipeline/state.py — v1
"""Mutable pipeline state flowing through all agents.

Accumulates results from each phase: extraction, chunking,
decontextualization, graph building, community detection, etc.

See spec §25.1 for full documentation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from ayextractor.core.models import (
    Chunk,
    ConsolidatedTriplet,
    DocumentStructure,
    EntityNormalization,
    ExtractionResult,
    QualifiedTriplet,
    Reference,
)
from ayextractor.graph.layers.models import CommunityHierarchy, CommunitySummary
from ayextractor.graph.profiles.models import EntityProfile, RelationProfile
from ayextractor.pipeline.plugin_kit.models import AgentOutput


class PipelineState(BaseModel):
    """Mutable state accumulating results across all pipeline phases.

    Each agent reads from and writes to this state. The pipeline runner
    passes it between agents according to DAG order.
    """

    model_config = {"arbitrary_types_allowed": True}

    # === IDENTITY ===
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    document_title: str = ""
    language: str = "en"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # === PHASE 1 — EXTRACTION ===
    extraction_result: ExtractionResult | None = None
    enriched_text: str = ""
    structure: DocumentStructure | None = None
    references: list[Reference] = Field(default_factory=list)

    # === PHASE 2 — CHUNKING + DECONTEXTUALIZATION ===
    chunks: list[Chunk] = Field(default_factory=list)
    refine_summary: str = ""
    dense_summary: str = ""

    # === PHASE 3 — AGENTS ===
    # Per-chunk raw triplets
    raw_triplets: list[QualifiedTriplet] = Field(default_factory=list)
    # Consolidated triplets after merger
    entity_normalizations: list[EntityNormalization] = Field(default_factory=list)
    consolidated_triplets: list[ConsolidatedTriplet] = Field(default_factory=list)
    merger_stats: dict[str, Any] = Field(default_factory=dict)

    # === GRAPH ===
    graph: nx.Graph | None = Field(default=None, exclude=True)
    community_hierarchy: CommunityHierarchy | None = None
    community_summaries: list[CommunitySummary] = Field(default_factory=list)

    # === PROFILES ===
    entity_profiles: list[EntityProfile] = Field(default_factory=list)
    relation_profiles: list[RelationProfile] = Field(default_factory=list)

    # === SYNTHESIS ===
    synthesis: str = ""
    key_findings: list[str] = Field(default_factory=list)

    # === CRITIC ===
    quality_score: float = 0.0
    quality_issues: list[dict[str, Any]] = Field(default_factory=list)

    # === AGENT OUTPUTS (raw) ===
    agent_outputs: dict[str, AgentOutput] = Field(default_factory=dict)

    # === STATS ===
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    errors: list[str] = Field(default_factory=list)

    def record_agent_output(self, agent_name: str, output: AgentOutput) -> None:
        """Record an agent's output and update running stats."""
        self.agent_outputs[agent_name] = output
        self.total_llm_calls += output.metadata.llm_calls
        self.total_tokens_used += output.metadata.tokens_used

    def get_graph_stats(self) -> dict[str, Any]:
        """Return summary stats about the current graph."""
        if self.graph is None:
            return {"nodes": 0, "edges": 0}
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "l1_nodes": sum(
                1 for _, d in self.graph.nodes(data=True) if d.get("layer") == 1
            ),
            "l2_nodes": sum(
                1 for _, d in self.graph.nodes(data=True) if d.get("layer") == 2
            ),
            "l3_nodes": sum(
                1 for _, d in self.graph.nodes(data=True) if d.get("layer") == 3
            ),
            "communities": len(self.community_summaries),
        }
