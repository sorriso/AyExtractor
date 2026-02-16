# tests/unit/graph/test_decay_manager.py — v1
"""Tests for graph/decay_manager.py — temporal confidence decay."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import networkx as nx
import pytest

from ayextractor.graph.decay_manager import (
    MIN_CONFIDENCE,
    apply_decay,
    compute_decay_factor,
)


class TestComputeDecayFactor:
    def test_zero_age_returns_one(self):
        assert compute_decay_factor(0) == 1.0

    def test_negative_age_returns_one(self):
        assert compute_decay_factor(-10) == 1.0

    def test_half_life_returns_half(self):
        factor = compute_decay_factor(365, half_life_days=365)
        assert abs(factor - 0.5) < 0.01

    def test_two_half_lives_returns_quarter(self):
        factor = compute_decay_factor(730, half_life_days=365)
        assert abs(factor - 0.25) < 0.01

    def test_very_old_clamps_to_min(self):
        factor = compute_decay_factor(10000, half_life_days=30)
        assert factor == MIN_CONFIDENCE

    def test_zero_half_life_returns_min(self):
        assert compute_decay_factor(100, half_life_days=0) == MIN_CONFIDENCE


class TestApplyDecay:
    def test_recent_node_minimal_decay(self):
        g = nx.Graph()
        now = datetime.now(timezone.utc)
        g.add_node("A", confidence=0.9,
                    last_updated_at=(now - timedelta(days=1)).isoformat())
        apply_decay(g, reference_date=now, half_life_days=365)
        assert g.nodes["A"]["confidence"] > 0.89

    def test_old_node_significant_decay(self):
        g = nx.Graph()
        now = datetime.now(timezone.utc)
        g.add_node("A", confidence=0.9,
                    last_updated_at=(now - timedelta(days=365)).isoformat())
        apply_decay(g, reference_date=now, half_life_days=365)
        assert g.nodes["A"]["confidence"] < 0.5

    def test_edge_decay(self):
        g = nx.Graph()
        now = datetime.now(timezone.utc)
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", confidence=0.8,
                    last_updated_at=(now - timedelta(days=365)).isoformat())
        stats = apply_decay(g, reference_date=now, half_life_days=365)
        assert g["A"]["B"]["confidence"] < 0.5
        assert stats.edges_decayed == 1

    def test_no_last_updated_skipped(self):
        g = nx.Graph()
        g.add_node("A", confidence=0.9)
        apply_decay(g)
        assert g.nodes["A"]["confidence"] == 0.9

    def test_prune_below_threshold(self):
        g = nx.Graph()
        now = datetime.now(timezone.utc)
        g.add_node("Fresh", confidence=0.9,
                    last_updated_at=(now - timedelta(days=1)).isoformat())
        g.add_node("Stale", confidence=0.1,
                    last_updated_at=(now - timedelta(days=3000)).isoformat())
        stats = apply_decay(g, reference_date=now, half_life_days=365,
                           prune_threshold=0.06)
        assert g.has_node("Fresh")
        assert not g.has_node("Stale")
        assert stats.nodes_below_threshold >= 1

    def test_decay_factor_stored(self):
        g = nx.Graph()
        now = datetime.now(timezone.utc)
        g.add_node("A", confidence=0.9,
                    last_updated_at=(now - timedelta(days=100)).isoformat())
        apply_decay(g, reference_date=now, half_life_days=365)
        assert "decay_factor" in g.nodes["A"]
        assert 0 < g.nodes["A"]["decay_factor"] < 1.0

    def test_empty_graph(self):
        g = nx.Graph()
        stats = apply_decay(g)
        assert stats.nodes_decayed == 0
        assert stats.edges_decayed == 0

    def test_stats_returned(self):
        g = nx.Graph()
        now = datetime.now(timezone.utc)
        g.add_node("A", confidence=0.9,
                    last_updated_at=now.isoformat())
        g.add_node("B", confidence=0.5,
                    last_updated_at=now.isoformat())
        g.add_edge("A", "B", confidence=0.7,
                    last_updated_at=now.isoformat())
        stats = apply_decay(g, reference_date=now)
        assert stats.nodes_decayed == 2
        assert stats.edges_decayed == 1
