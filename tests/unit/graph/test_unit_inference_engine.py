# tests/unit/graph/test_inference_engine.py — v1
"""Tests for graph/inference_engine.py — implicit triplet inference."""

from __future__ import annotations

import networkx as nx
import pytest

from ayextractor.graph.inference_engine import (
    InferenceResult,
    run_inference,
)


def _build_transitive_graph() -> nx.Graph:
    """A part_of B part_of C → should infer A part_of C."""
    g = nx.Graph()
    g.add_node("A")
    g.add_node("B")
    g.add_node("C")
    g.add_edge("A", "B", relation_type="part_of", confidence=0.8)
    g.add_edge("B", "C", relation_type="part_of", confidence=0.9)
    return g


def _build_inverse_graph() -> nx.Graph:
    """A contains B → should infer B part_of A."""
    g = nx.Graph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", relation_type="contains", confidence=0.85)
    return g


class TestRunInference:
    def test_empty_graph(self):
        g = nx.Graph()
        result = run_inference(g)
        assert isinstance(result, InferenceResult)
        assert result.stats["total_inferred"] == 0

    def test_transitive_inference(self):
        g = _build_transitive_graph()
        result = run_inference(g, apply_to_graph=False)
        transitive = [e for e in result.inferred_edges if e.rule == "transitivity"]
        assert len(transitive) >= 1
        # Should infer A part_of C
        found = any(
            e.source == "A" and e.target == "C" and e.relation_type == "part_of"
            for e in transitive
        )
        assert found

    def test_transitive_confidence_discounted(self):
        g = _build_transitive_graph()
        result = run_inference(g)
        edge = next(
            (e for e in result.inferred_edges if e.source == "A" and e.target == "C"),
            None,
        )
        assert edge is not None
        # Confidence should be < min(0.8, 0.9) due to discount
        assert edge.confidence < 0.8

    def test_inverse_inference(self):
        g = _build_inverse_graph()
        result = run_inference(g)
        inverses = [e for e in result.inferred_edges if e.rule == "inverse"]
        assert len(inverses) >= 1
        found = any(
            e.source == "B" and e.target == "A" and e.relation_type == "part_of"
            for e in inverses
        )
        assert found

    def test_no_duplicate_with_existing(self):
        """If inverse already exists, don't infer it."""
        g = nx.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", relation_type="contains", confidence=0.8)
        g.add_edge("B", "A", relation_type="part_of", confidence=0.7)
        result = run_inference(g)
        # Should not re-infer B→A part_of since it exists
        part_of_inferred = [
            e for e in result.inferred_edges
            if e.source == "B" and e.target == "A" and e.relation_type == "part_of"
        ]
        assert len(part_of_inferred) == 0

    def test_apply_to_graph(self):
        g = _build_transitive_graph()
        edges_before = g.number_of_edges()
        result = run_inference(g, apply_to_graph=True)
        assert g.number_of_edges() > edges_before
        # Check inferred edge has is_inferred flag
        for u, v, data in g.edges(data=True):
            if data.get("is_inferred"):
                assert data["inference_rule"] in ("transitivity", "inverse")

    def test_max_depth_limits_chains(self):
        """Long transitive chains should be capped."""
        g = nx.Graph()
        for i in range(10):
            g.add_node(f"N{i}")
        for i in range(9):
            g.add_edge(f"N{i}", f"N{i+1}", relation_type="part_of", confidence=0.9)
        result = run_inference(g, max_depth=2)
        # Should not infer N0→N9 (depth 9)
        far = [e for e in result.inferred_edges if e.source == "N0" and e.target == "N9"]
        assert len(far) == 0

    def test_non_transitive_not_chained(self):
        """Non-transitive predicates should not produce chains."""
        g = nx.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", relation_type="regulates", confidence=0.8)
        g.add_edge("B", "C", relation_type="regulates", confidence=0.9)
        result = run_inference(g)
        transitive = [e for e in result.inferred_edges if e.rule == "transitivity"]
        # regulates is NOT transitive
        a_to_c = [e for e in transitive if e.source == "A" and e.target == "C"]
        assert len(a_to_c) == 0

    def test_evidence_path_recorded(self):
        g = _build_transitive_graph()
        result = run_inference(g)
        edge = next(
            (e for e in result.inferred_edges if e.source == "A" and e.target == "C"),
            None,
        )
        assert edge is not None
        assert len(edge.evidence_path) >= 3
        assert "A" in edge.evidence_path
        assert "C" in edge.evidence_path

    def test_stats_populated(self):
        g = _build_transitive_graph()
        result = run_inference(g)
        assert "total_inferred" in result.stats
        assert "transitive" in result.stats
        assert "inverse" in result.stats
