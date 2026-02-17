# tests/unit/graph/layers/test_layer_classifier.py — v1
"""Tests for graph/layers/layer_classifier.py — L2/L3 classification."""

from __future__ import annotations

import networkx as nx
import pytest

from ayextractor.graph.layers.layer_classifier import (
    apply_layers,
    classify_layers,
    get_l2_subgraph,
)


def _build_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("European Union", is_literal=False, layer=2)
    g.add_node("AI Act", is_literal=False, layer=2)
    g.add_node("500", is_literal=True, layer=3)
    g.add_node("15%", is_literal=True, layer=3)
    g.add_edge("European Union", "AI Act", relation_type="regulates")
    g.add_edge("European Union", "500", relation_type="has_attribute")
    g.add_edge("AI Act", "15%", relation_type="measured_at")
    return g


class TestClassifyLayers:
    def test_literal_gets_l3(self):
        g = _build_graph()
        layers = classify_layers(g)
        assert layers["500"] == 3
        assert layers["15%"] == 3

    def test_non_literal_gets_l2(self):
        g = _build_graph()
        layers = classify_layers(g)
        assert layers["European Union"] == 2
        assert layers["AI Act"] == 2

    def test_empty_graph(self):
        g = nx.Graph()
        layers = classify_layers(g)
        assert layers == {}

    def test_all_non_literal(self):
        g = nx.Graph()
        g.add_node("A", is_literal=False)
        g.add_node("B", is_literal=False)
        layers = classify_layers(g)
        assert all(v == 2 for v in layers.values())

    def test_all_literal(self):
        g = nx.Graph()
        g.add_node("42", is_literal=True)
        g.add_node("100%", is_literal=True)
        layers = classify_layers(g)
        assert all(v == 3 for v in layers.values())

    def test_missing_is_literal_defaults_l2(self):
        """Nodes without is_literal attribute default to L2."""
        g = nx.Graph()
        g.add_node("Unknown")
        layers = classify_layers(g)
        assert layers["Unknown"] == 2


class TestApplyLayers:
    def test_sets_layer_attribute(self):
        g = nx.Graph()
        g.add_node("A", layer=0)
        g.add_node("B", layer=0)
        apply_layers(g, {"A": 2, "B": 3})
        assert g.nodes["A"]["layer"] == 2
        assert g.nodes["B"]["layer"] == 3

    def test_ignores_missing_nodes(self):
        g = nx.Graph()
        g.add_node("A", layer=0)
        apply_layers(g, {"A": 2, "MISSING": 3})
        assert g.nodes["A"]["layer"] == 2
        assert "MISSING" not in g.nodes


class TestGetL2Subgraph:
    def test_excludes_l3_nodes(self):
        g = _build_graph()
        sub = get_l2_subgraph(g)
        assert "European Union" in sub.nodes
        assert "AI Act" in sub.nodes
        assert "500" not in sub.nodes
        assert "15%" not in sub.nodes

    def test_preserves_l2_edges(self):
        g = _build_graph()
        sub = get_l2_subgraph(g)
        assert sub.has_edge("European Union", "AI Act")

    def test_removes_l2_l3_edges(self):
        g = _build_graph()
        sub = get_l2_subgraph(g)
        assert not sub.has_edge("European Union", "500")

    def test_empty_graph(self):
        g = nx.Graph()
        sub = get_l2_subgraph(g)
        assert sub.number_of_nodes() == 0

    def test_returns_copy(self):
        """Subgraph should be a copy, not a view."""
        g = _build_graph()
        sub = get_l2_subgraph(g)
        sub.add_node("NEW")
        assert "NEW" not in g.nodes
