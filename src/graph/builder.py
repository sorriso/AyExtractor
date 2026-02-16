# src/graph/builder.py — v1
"""Knowledge graph builder — constructs NetworkX graph from consolidated triplets.

Builds L2 (ACTORS) and L3 (EVIDENCE) nodes and edges. L1 (TOPICS)
are added later by community_integrator.py after Leiden clustering.

See spec §13.8 for builder role, §13.9.4 / §13.9.5 for node/edge schema.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import networkx as nx

from ayextractor.core.models import (
    ConsolidatedTriplet,
    EntityNormalization,
)

logger = logging.getLogger(__name__)


def build_document_graph(
    consolidated_triplets: list[ConsolidatedTriplet],
    entity_normalizations: list[EntityNormalization],
) -> nx.Graph:
    """Build an undirected NetworkX graph from consolidated triplets.

    Creates nodes for every unique entity (subject or object) and
    edges for every consolidated triplet. Node attributes follow
    the spec §13.9.4 schema; edge attributes follow §13.9.5.

    L3 (EVIDENCE) nodes are detected by the ``is_literal`` heuristic:
    values that look like numbers, percentages, dates, or measures.

    Args:
        consolidated_triplets: Normalized triplets from merger pipeline.
        entity_normalizations: Entity normalization table for metadata.

    Returns:
        NetworkX Graph with L2/L3 nodes and edges.
    """
    graph = nx.Graph()
    now = datetime.now(timezone.utc)

    # Build entity lookup for metadata
    entity_map: dict[str, EntityNormalization] = {
        e.canonical_name: e for e in entity_normalizations
    }

    # Collect per-entity stats from triplets
    entity_chunks: dict[str, set[str]] = {}
    entity_counts: dict[str, int] = {}

    for t in consolidated_triplets:
        for entity in (t.subject, t.object):
            entity_chunks.setdefault(entity, set()).update(t.source_chunk_ids)
            entity_counts[entity] = entity_counts.get(entity, 0) + t.occurrence_count

    # Create nodes
    all_entities = set(entity_chunks.keys())
    for entity in all_entities:
        meta = entity_map.get(entity)
        is_lit = _is_literal(entity)

        attrs: dict[str, Any] = {
            "canonical_name": entity,
            "layer": 3 if is_lit else 2,
            "entity_type": (meta.entity_type if meta else ("value" if is_lit else "concept")),
            "aliases": list(meta.aliases) if meta else [],
            "occurrence_count": entity_counts.get(entity, 0),
            "source_chunk_ids": sorted(entity_chunks.get(entity, set())),
            "confidence": 0.0,  # updated below from edges
            "salience": 0.0,
            "first_seen_at": now.isoformat(),
            "last_updated_at": now.isoformat(),
            "community_id": None,
            "is_literal": is_lit,
        }
        graph.add_node(entity, **attrs)

    # Create edges
    for t in consolidated_triplets:
        if not graph.has_node(t.subject) or not graph.has_node(t.object):
            logger.debug("Skipping edge with missing node: %s -> %s", t.subject, t.object)
            continue

        edge_attrs: dict[str, Any] = {
            "relation_type": t.predicate,
            "confidence": t.confidence,
            "occurrence_count": t.occurrence_count,
            "source_chunk_ids": t.source_chunk_ids,
            "qualifiers": t.qualifiers,
            "temporal_scope": t.temporal_scope.model_dump() if t.temporal_scope else None,
            "context_sentences": t.context_sentences,
            "original_forms": t.original_forms,
            "first_seen_at": now.isoformat(),
            "last_updated_at": now.isoformat(),
            "corroboration": 1,
        }

        # NetworkX multigraph-like: use relation as key in edge data
        if graph.has_edge(t.subject, t.object):
            existing = graph[t.subject][t.object]
            # Append as list of relations if multiple between same pair
            relations = existing.get("_relations", [existing.copy()])
            relations.append(edge_attrs)
            graph[t.subject][t.object]["_relations"] = relations
            # Keep highest confidence on main edge
            if t.confidence > existing.get("confidence", 0):
                graph[t.subject][t.object].update(edge_attrs)
        else:
            graph.add_edge(t.subject, t.object, **edge_attrs)

    # Update node confidence from max edge confidence
    for node in graph.nodes:
        max_conf = 0.0
        for _, _, data in graph.edges(node, data=True):
            max_conf = max(max_conf, data.get("confidence", 0.0))
        graph.nodes[node]["confidence"] = max_conf

    logger.info(
        "Built document graph: %d nodes (%d L2, %d L3), %d edges",
        graph.number_of_nodes(),
        sum(1 for _, d in graph.nodes(data=True) if d.get("layer") == 2),
        sum(1 for _, d in graph.nodes(data=True) if d.get("layer") == 3),
        graph.number_of_edges(),
    )

    return graph


def _is_literal(value: str) -> bool:
    """Heuristic: detect if entity name looks like a literal value (L3).

    Literal values include numbers, percentages, dates, currency amounts,
    and measurements. They only make sense attached to an L2 entity.
    """
    import re

    stripped = value.strip()
    if not stripped:
        return False
    # Pure numbers, percentages, currency
    if re.match(r"^[\d.,]+\s*[%€$£¥]?$", stripped):
        return True
    # Measurements like "500 employees", "15 kg", "3.5 GHz"
    if re.match(r"^[\d.,]+\s+\w{1,10}$", stripped):
        return True
    # Date patterns like "2025-01-15", "January 2025"
    if re.match(r"^\d{4}[-/]\d{2}([-/]\d{2})?$", stripped):
        return True
    return False
