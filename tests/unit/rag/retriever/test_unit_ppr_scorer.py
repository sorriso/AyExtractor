# tests/unit/rag/retriever/test_ppr_scorer.py — v1
"""Tests for rag/retriever/ppr_scorer.py."""

from __future__ import annotations

import networkx as nx
import pytest

from ayextractor.rag.retriever.ppr_scorer import combined_score, ppr_score


class TestPPRScore:
    def test_basic_scoring(self):
        g = nx.path_graph(5)  # 0-1-2-3-4
        scores = ppr_score(g, seed_entities=[0])
        assert len(scores) == 5
        # Seed node should have highest score
        assert scores[0] >= scores[4]

    def test_normalized_to_unit_range(self):
        g = nx.complete_graph(10)
        scores = ppr_score(g, seed_entities=[0, 1])
        assert all(0.0 <= v <= 1.0 for v in scores.values())
        # At least one node should have score 1.0 (max normalized)
        assert max(scores.values()) == pytest.approx(1.0)

    def test_empty_graph(self):
        g = nx.Graph()
        scores = ppr_score(g, seed_entities=["x"])
        assert scores == {}

    def test_no_seeds(self):
        g = nx.complete_graph(5)
        scores = ppr_score(g, seed_entities=[])
        assert scores == {}

    def test_invalid_seeds_ignored(self):
        g = nx.path_graph(3)
        scores = ppr_score(g, seed_entities=["nonexistent"])
        assert scores == {}

    def test_multiple_seeds(self):
        g = nx.star_graph(5)  # center=0, leaves=1-5
        scores = ppr_score(g, seed_entities=[1, 2])
        assert len(scores) == 6
        # Center (0) should benefit from being hub
        assert scores[0] > 0

    def test_disconnected_graph(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2)])
        g.add_edges_from([(10, 11)])
        scores = ppr_score(g, seed_entities=[0])
        # Disconnected component should have ~0 score
        assert scores.get(10, 0) < scores.get(0, 1)


class TestCombinedScore:
    def test_default_weights(self):
        # α=0.3 → 0.3*0.8 + 0.7*0.6 = 0.24 + 0.42 = 0.66
        result = combined_score(0.8, 0.6)
        assert result == pytest.approx(0.66)

    def test_custom_weight(self):
        result = combined_score(1.0, 0.0, composite_weight=0.5)
        assert result == pytest.approx(0.5)

    def test_zero_inputs(self):
        assert combined_score(0.0, 0.0) == 0.0
