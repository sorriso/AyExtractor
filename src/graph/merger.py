# src/graph/merger.py — v1
"""Triplet consolidation pipeline orchestrator.

Runs three sequential passes:
  1. Entity Normalization (entity_normalizer.py)
  2. Relation Normalization (relation_normalizer.py)
  3. Triplet Dedup & Merge (triplet_consolidator.py)

See spec §13.2 for pipeline overview.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ayextractor.core.models import (
    ConsolidatedTriplet,
    EntityNormalization,
    QualifiedTriplet,
    RelationTaxonomyEntry,
)
from ayextractor.graph.entity_normalizer import (
    build_entity_mapping,
    normalize_entities,
)
from ayextractor.graph.relation_normalizer import (
    build_relation_mapping,
    normalize_relations,
)
from ayextractor.graph.triplet_consolidator import consolidate_triplets

if TYPE_CHECKING:
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class MergerResult:
    """Result of the full 3-pass consolidation pipeline."""

    entity_normalizations: list[EntityNormalization] = field(default_factory=list)
    relation_taxonomy: list[RelationTaxonomyEntry] = field(default_factory=list)
    consolidated_triplets: list[ConsolidatedTriplet] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


async def run_merger_pipeline(
    raw_triplets: list[QualifiedTriplet],
    embedder: BaseEmbedder | None = None,
    entity_similarity_threshold: float = 0.85,
    relation_extensible: bool = True,
    boost_confidence: bool = True,
) -> MergerResult:
    """Execute the full 3-pass consolidation pipeline.

    Args:
        raw_triplets: All qualified triplets from all chunks.
        embedder: Optional embedder for entity similarity clustering.
        entity_similarity_threshold: Cosine threshold for entity clustering.
        relation_extensible: Allow new taxonomy entries for unknown relations.
        boost_confidence: Boost confidence for multi-occurrence triplets.

    Returns:
        MergerResult with normalizations, taxonomy, and consolidated triplets.
    """
    if not raw_triplets:
        return MergerResult(stats={"total_raw_triplets": 0})

    # Collect unique entity/relation counts before
    unique_entities_before = len(
        {t.subject.strip() for t in raw_triplets}
        | {t.object.strip() for t in raw_triplets}
    )
    unique_relations_before = len(
        {t.predicate.strip() for t in raw_triplets}
    )

    # Pass 1 — Entity Normalization
    logger.info("Merger Pass 1: Entity normalization (%d triplets)", len(raw_triplets))
    entity_norms = await normalize_entities(
        raw_triplets,
        embedder=embedder,
        threshold=entity_similarity_threshold,
    )
    entity_mapping = build_entity_mapping(entity_norms)

    # Pass 2 — Relation Normalization
    logger.info("Merger Pass 2: Relation normalization")
    relation_taxonomy = normalize_relations(
        raw_triplets,
        extensible=relation_extensible,
    )
    relation_mapping = build_relation_mapping(
        raw_triplets,
        extensible=relation_extensible,
    )

    # Pass 3 — Triplet Consolidation
    logger.info("Merger Pass 3: Triplet consolidation")
    consolidated = consolidate_triplets(
        raw_triplets,
        entity_mapping=entity_mapping,
        relation_mapping=relation_mapping,
        boost_confidence=boost_confidence,
    )

    unique_entities_after = len(entity_norms)
    unique_relations_after = len(
        {t.predicate for t in consolidated}
    )

    stats = {
        "total_raw_triplets": len(raw_triplets),
        "unique_entities_before": unique_entities_before,
        "unique_entities_after": unique_entities_after,
        "unique_relations_before": unique_relations_before,
        "unique_relations_after": unique_relations_after,
        "consolidated_triplets": len(consolidated),
        "dedup_ratio": round(
            1 - len(consolidated) / max(len(raw_triplets), 1), 4
        ),
    }

    logger.info("Merger complete: %s", stats)

    return MergerResult(
        entity_normalizations=entity_norms,
        relation_taxonomy=relation_taxonomy,
        consolidated_triplets=consolidated,
        stats=stats,
    )
