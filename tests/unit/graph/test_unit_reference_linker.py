# tests/unit/graph/test_reference_linker.py — v1
"""Tests for graph/reference_linker.py — injecting references into the graph."""

from __future__ import annotations

import networkx as nx
import pytest

from ayextractor.core.models import EntityNormalization, Reference
from ayextractor.graph.reference_linker import link_references


def _build_graph() -> nx.Graph:
    """Build a small test graph."""
    g = nx.Graph()
    g.add_node("European Union", canonical_name="European Union", layer=2,
               entity_type="organization", is_literal=False)
    g.add_node("AI Act", canonical_name="AI Act", layer=2,
               entity_type="document", is_literal=False)
    g.add_node("ISO 21434", canonical_name="ISO 21434", layer=2,
               entity_type="document", is_literal=False)
    g.add_edge("European Union", "AI Act", relation_type="regulates")
    return g


def _build_norms() -> list[EntityNormalization]:
    return [
        EntityNormalization(
            canonical_name="European Union",
            aliases=["EU", "l'UE"],
        ),
        EntityNormalization(
            canonical_name="AI Act",
            aliases=["Artificial Intelligence Act"],
        ),
    ]


class TestLinkReferences:
    def test_empty_references_no_change(self):
        g = _build_graph()
        edges_before = g.number_of_edges()
        link_references(g, [], _build_norms())
        assert g.number_of_edges() == edges_before

    def test_citation_creates_cites_edge(self):
        g = _build_graph()
        refs = [
            Reference(
                type="citation",
                text="EU",
                target="ISO 21434",
                source_chunk_id="c1",
            ),
        ]
        link_references(g, refs, _build_norms())
        assert g.has_edge("European Union", "ISO 21434")
        edge = g["European Union"]["ISO 21434"]
        assert edge["relation_type"] == "cites"

    def test_bibliography_creates_node(self):
        g = _build_graph()
        refs = [
            Reference(
                type="bibliography",
                text="Smith et al. (2023). AI Safety. Nature.",
                target="Smith2023",
                source_chunk_id="c1",
            ),
        ]
        nodes_before = g.number_of_nodes()
        link_references(g, refs, _build_norms())
        assert g.number_of_nodes() == nodes_before + 1
        assert g.has_node("Smith2023")
        assert g.nodes["Smith2023"]["entity_type"] == "document"

    def test_internal_ref_creates_references_edge(self):
        g = _build_graph()
        refs = [
            Reference(
                type="internal_ref",
                text="European Union",
                target="AI Act",
                source_chunk_id="c1",
            ),
        ]
        # There's already an edge, so this won't create a duplicate
        link_references(g, refs, _build_norms())
        # But the original edge is still there
        assert g.has_edge("European Union", "AI Act")

    def test_alias_matching(self):
        """References using aliases should match canonical nodes."""
        g = _build_graph()
        refs = [
            Reference(
                type="citation",
                text="l'UE",
                target="ISO 21434",
                source_chunk_id="c1",
            ),
        ]
        link_references(g, refs, _build_norms())
        assert g.has_edge("European Union", "ISO 21434")

    def test_unresolvable_reference_ignored(self):
        g = _build_graph()
        refs = [
            Reference(
                type="citation",
                text="Unknown Entity",
                target="Also Unknown",
                source_chunk_id="c1",
            ),
        ]
        edges_before = g.number_of_edges()
        link_references(g, refs, _build_norms())
        assert g.number_of_edges() == edges_before

    def test_footnote_no_edge_created(self):
        g = _build_graph()
        refs = [
            Reference(
                type="footnote",
                text="See annex for details.",
                target=None,
                source_chunk_id="c1",
            ),
        ]
        edges_before = g.number_of_edges()
        link_references(g, refs, _build_norms())
        assert g.number_of_edges() == edges_before

    def test_no_duplicate_bibliography_nodes(self):
        g = _build_graph()
        refs = [
            Reference(type="bibliography", text="Smith 2023", target="Smith2023",
                      source_chunk_id="c1"),
            Reference(type="bibliography", text="Smith 2023 v2", target="Smith2023",
                      source_chunk_id="c2"),
        ]
        link_references(g, refs, _build_norms())
        # Smith2023 created only once
        count = sum(1 for n in g.nodes if n == "Smith2023")
        assert count == 1
