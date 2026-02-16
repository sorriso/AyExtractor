# src/consolidator/models.py — v1
"""Corpus Graph consolidation models.

See spec §13.12 (TNode, CNode, XEdge) and §15 (consolidation reports).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ayextractor.core.models import SourceProvenance, TemporalScope


class TNode(BaseModel):
    """Taxonomy node — domain/sub-domain/concept hierarchy."""

    canonical_name: str
    level: Literal["domain", "subdomain", "concept"]
    parent: str | None = None
    children: list[str] = Field(default_factory=list)
    classified_cnodes: list[str] = Field(default_factory=list)
    created_by: Literal["manual", "consolidator_clustering"]
    created_at: datetime


class CNode(BaseModel):
    """Canonical concept — entity merged across multiple documents."""

    canonical_name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    source_documents: list[SourceProvenance] = Field(default_factory=list)
    corroboration: int = 0
    consolidated_attributes: dict[str, Any] = Field(default_factory=dict)
    taxonomy_path: str | None = None
    confidence: float = 0.0
    salience: float = 0.0
    first_seen_at: datetime
    last_updated_at: datetime


class XEdge(BaseModel):
    """Cross-document relation between C-nodes."""

    source: str
    relation_type: str
    target: str
    source_documents: list[SourceProvenance] = Field(default_factory=list)
    corroboration: int = 0
    confidence: float = 0.0
    qualifiers: dict[str, str] | None = None
    temporal_scope: TemporalScope | None = None
    inferred: bool = False
    first_seen_at: datetime
    last_updated_at: datetime


class Contradiction(BaseModel):
    """Conflicting claims detected in Corpus Graph."""

    contradiction_id: str
    edge_a_subject: str
    edge_a_predicate: str
    edge_a_object: str
    edge_b_subject: str
    edge_b_predicate: str
    edge_b_object: str
    conflict_type: Literal["value", "temporal", "negation"]
    source_documents_a: list[str] = Field(default_factory=list)
    source_documents_b: list[str] = Field(default_factory=list)


class PassResult(BaseModel):
    """Result of a single consolidation pass."""

    pass_name: str
    duration_ms: int
    items_processed: int
    items_modified: int
    details: dict[str, Any] = Field(default_factory=dict)


class LinkingReport(BaseModel):
    new_cnodes: int
    updated_cnodes: int
    new_xedges: int
    updated_xedges: int
    documents_linked: int


class ClusteringReport(BaseModel):
    new_tnodes: int
    updated_tnodes: int
    clusters_found: int


class InferenceReport(BaseModel):
    proposed: int
    accepted: int
    rejected: int


class DecayReport(BaseModel):
    recalculated: int
    marked_stale: int
    pruned: int


class ContradictionReport(BaseModel):
    found: int
    details: list[Contradiction] = Field(default_factory=list)


class ConsolidationReport(BaseModel):
    """Top-level report for a full consolidation run."""

    consolidation_id: str
    timestamp: datetime
    trigger: Literal["on_ingestion", "scheduled", "manual"]
    passes_executed: list[str] = Field(default_factory=list)
    results: dict[str, PassResult] = Field(default_factory=dict)
    corpus_stats: dict[str, int] = Field(default_factory=dict)
