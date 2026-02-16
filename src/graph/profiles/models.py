# src/graph/profiles/models.py — v1
"""Entity and relation profile models.

See spec §13.11 for full documentation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ayextractor.core.models import TemporalScope


class EntityProfile(BaseModel):
    """Textual profile for an L2 entity, used for RAG vector search."""

    canonical_name: str
    entity_type: str
    profile_text: str
    key_relations: list[str] = Field(default_factory=list)
    community_id: str | None = None
    embedding: list[float] | None = None


class RelationProfile(BaseModel):
    """Textual profile for a key relation."""

    subject: str
    predicate: str
    object: str
    profile_text: str
    qualifiers: dict[str, str] | None = None
    temporal_scope: TemporalScope | None = None
    embedding: list[float] | None = None
