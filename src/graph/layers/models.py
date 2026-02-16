# src/graph/layers/models.py — v1
"""Community detection models: Community, CommunityHierarchy, CommunitySummary.

See spec §13.10 for full documentation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Community(BaseModel):
    """Single community detected by Leiden algorithm."""

    community_id: str
    level: int
    members: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    modularity_score: float = 0.0
    chunk_coverage: list[str] = Field(default_factory=list)


class CommunityHierarchy(BaseModel):
    """Full hierarchical community structure from Leiden."""

    communities: list[Community] = Field(default_factory=list)
    num_levels: int = 0
    resolution: float = 1.0
    seed: int | None = None
    total_communities: int = 0
    modularity: float = 0.0


class CommunitySummary(BaseModel):
    """LLM-generated summary for a single community."""

    community_id: str
    level: int
    title: str
    summary: str
    key_entities: list[str] = Field(default_factory=list)
    chunk_coverage: list[str] = Field(default_factory=list)
    member_count: int = 0
