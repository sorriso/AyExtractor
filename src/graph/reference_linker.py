# src/graph/reference_linker.py — v1
"""Reference linker — inject extracted references into the knowledge graph.

Links citations, footnotes, bibliography entries, and internal cross-references
to existing graph nodes. Creates 'cites' and 'references' edges.

See spec §12.2 for integration rules.
"""

from __future__ import annotations

import logging

import networkx as nx

from ayextractor.core.models import EntityNormalization, Reference

logger = logging.getLogger(__name__)


def link_references(
    graph: nx.Graph,
    references: list[Reference],
    entity_normalizations: list[EntityNormalization],
) -> nx.Graph:
    """Link extracted references to graph nodes.

    For each reference, attempts to find matching nodes in the graph and
    creates appropriate edges:
      - citation -> 'cites' edge from document node to source node
      - internal_ref -> 'references' edge between section nodes
      - footnote -> enriches related concept nodes (no new edge)
      - bibliography -> creates source node if not present

    Args:
        graph: NetworkX graph with L2/L3 nodes.
        references: Extracted references from ReferenceExtractorAgent.
        entity_normalizations: Entity normalization table for name matching.

    Returns:
        Modified graph with reference edges added.
    """
    if not references:
        return graph

    # Build alias lookup for fuzzy matching
    alias_to_canonical: dict[str, str] = {}
    for norm in entity_normalizations:
        alias_to_canonical[norm.canonical_name.lower()] = norm.canonical_name
        for alias in norm.aliases:
            alias_to_canonical[alias.lower()] = norm.canonical_name

    linked_count = 0
    created_count = 0

    for ref in references:
        if ref.type == "citation":
            linked_count += _link_citation(graph, ref, alias_to_canonical)
        elif ref.type == "bibliography":
            created_count += _link_bibliography(graph, ref)
        elif ref.type == "internal_ref":
            linked_count += _link_internal_ref(graph, ref, alias_to_canonical)
        # footnotes are informational, no graph edges needed

    logger.info(
        "Reference linking: %d linked, %d bibliography nodes created from %d references",
        linked_count,
        created_count,
        len(references),
    )
    return graph


def _find_node(
    graph: nx.Graph,
    text: str,
    alias_map: dict[str, str],
) -> str | None:
    """Find a graph node matching the text (exact or alias match)."""
    # Direct match
    if graph.has_node(text):
        return text
    # Alias match
    canonical = alias_map.get(text.lower())
    if canonical and graph.has_node(canonical):
        return canonical
    # Substring match on node names (last resort)
    text_lower = text.lower()
    for node in graph.nodes:
        if text_lower in node.lower() or node.lower() in text_lower:
            return node
    return None


def _link_citation(
    graph: nx.Graph,
    ref: Reference,
    alias_map: dict[str, str],
) -> int:
    """Create a 'cites' edge for a citation reference."""
    if ref.target is None:
        return 0
    target_node = _find_node(graph, ref.target, alias_map)
    if target_node is None:
        return 0

    # Find source node (the entity that cites)
    source_node = _find_node(graph, ref.text, alias_map)
    if source_node is None:
        # Use a generic document-level node
        source_node = "document"
        if not graph.has_node(source_node):
            graph.add_node(
                source_node,
                canonical_name="document",
                layer=2,
                entity_type="document",
                is_literal=False,
            )

    if not graph.has_edge(source_node, target_node):
        graph.add_edge(
            source_node,
            target_node,
            relation_type="cites",
            confidence=0.8,
            occurrence_count=1,
            source_chunk_ids=[ref.source_chunk_id],
            reference_type=ref.type,
        )
        return 1
    return 0


def _link_bibliography(graph: nx.Graph, ref: Reference) -> int:
    """Create a bibliography source node if not present."""
    node_name = ref.target or ref.text[:80]
    if graph.has_node(node_name):
        return 0
    graph.add_node(
        node_name,
        canonical_name=node_name,
        layer=2,
        entity_type="document",
        aliases=[],
        occurrence_count=1,
        source_chunk_ids=[ref.source_chunk_id],
        is_literal=False,
        bibliography_text=ref.text,
    )
    return 1


def _link_internal_ref(
    graph: nx.Graph,
    ref: Reference,
    alias_map: dict[str, str],
) -> int:
    """Create a 'references' edge for internal cross-references."""
    if ref.target is None:
        return 0
    target_node = _find_node(graph, ref.target, alias_map)
    if target_node is None:
        return 0

    source_node = _find_node(graph, ref.text, alias_map)
    if source_node is None or source_node == target_node:
        return 0

    if not graph.has_edge(source_node, target_node):
        graph.add_edge(
            source_node,
            target_node,
            relation_type="references",
            confidence=0.7,
            occurrence_count=1,
            source_chunk_ids=[ref.source_chunk_id],
            reference_type=ref.type,
        )
        return 1
    return 0
