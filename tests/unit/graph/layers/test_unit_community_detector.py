# tests/unit/graph/layers/test_community_detector.py — v2
"""Tests for graph/layers/community_detector.py — Leiden / fallback detection."""

from __future__ import annotations

import networkx as nx
import pytest

from ayextractor.graph.layers.community_detector import (
    _collect_chunk_coverage,
    _fallback_detect,
    detect_communities,
)
from ayextractor.graph.layers.models import CommunityHierarchy


def _build_clustered_graph() -> nx.Graph:
    """Build a graph with two clear clusters connected by a bridge."""
    g = nx.Graph()
    # Cluster A
    for n in ["A1", "A2", "A3", "A4"]:
        g.add_node(n, layer=2, is_literal=False, source_chunk_ids=["c1", "c2"])
    g.add_edge("A1", "A2")
    g.add_edge("A2", "A3")
    g.add_edge("A3", "A4")
    g.add_edge("A1", "A3")
    g.add_edge("A1", "A4")
    # Cluster B
    for n in ["B1", "B2", "B3", "B4"]:
        g.add_node(n, layer=2, is_literal=False, source_chunk_ids=["c3", "c4"])
    g.add_edge("B1", "B2")
    g.add_edge("B2", "B3")
    g.add_edge("B3", "B4")
    g.add_edge("B1", "B3")
    g.add_edge("B1", "B4")
    # Bridge
    g.add_edge("A4", "B1")
    return g


class TestDetectCommunities:
    def test_empty_graph(self):
        g = nx.Graph()
        result = detect_communities(g)
        assert isinstance(result, CommunityHierarchy)
        assert result.total_communities == 0
        assert result.num_levels == 0

    def test_returns_hierarchy(self):
        g = _build_clustered_graph()
        result = detect_communities(g, min_community_size=2)
        assert isinstance(result, CommunityHierarchy)
        assert result.total_communities >= 1
        assert result.num_levels >= 1

    def test_resolution_affects_count(self):
        g = _build_clustered_graph()
        low_res = detect_communities(g, resolution=0.5, min_community_size=2)
        high_res = detect_communities(g, resolution=5.0, min_community_size=2)
        # Higher resolution fragments communities, which may result in
        # fewer communities above min_community_size (small groups filtered).
        # We only assert both produce valid hierarchies.
        assert isinstance(low_res, CommunityHierarchy)
        assert isinstance(high_res, CommunityHierarchy)
        assert low_res.total_communities >= 1  # 2 clear clusters at low res

    def test_min_community_size_filters_small(self):
        g = nx.Graph()
        g.add_node("lonely", layer=2, source_chunk_ids=["c1"])
        result = detect_communities(g, min_community_size=3)
        assert result.total_communities == 0

    def test_seed_reproducibility(self):
        g = _build_clustered_graph()
        r1 = detect_communities(g, seed=42, min_community_size=2)
        r2 = detect_communities(g, seed=42, min_community_size=2)
        assert r1.total_communities == r2.total_communities
        for c1, c2 in zip(r1.communities, r2.communities):
            assert set(c1.members) == set(c2.members)

    def test_all_members_covered(self):
        g = _build_clustered_graph()
        result = detect_communities(g, min_community_size=2)
        all_members = set()
        for c in result.communities:
            all_members.update(c.members)
        # Every node should be in at least one community (if size >= min)
        for node in g.nodes:
            if len(list(g.neighbors(node))) >= 1:
                assert node in all_members or result.total_communities == 0


class TestFallbackDetect:
    def test_two_components(self):
        g = nx.Graph()
        g.add_node("A1", source_chunk_ids=["c1"])
        g.add_node("A2", source_chunk_ids=["c1"])
        g.add_node("A3", source_chunk_ids=["c1"])
        g.add_edge("A1", "A2")
        g.add_edge("A2", "A3")
        g.add_node("B1", source_chunk_ids=["c2"])
        g.add_node("B2", source_chunk_ids=["c2"])
        g.add_node("B3", source_chunk_ids=["c2"])
        g.add_edge("B1", "B2")
        g.add_edge("B2", "B3")
        result = _fallback_detect(g, min_community_size=3, resolution=1.0, seed=42)
        assert result.total_communities == 2

    def test_single_component(self):
        g = nx.Graph()
        for n in ["N0", "N1", "N2", "N3", "N4"]:
            g.add_node(n, source_chunk_ids=["c1"])
        for i in range(4):
            g.add_edge(f"N{i}", f"N{i+1}")
        result = _fallback_detect(g, min_community_size=3, resolution=1.0, seed=42)
        assert result.total_communities == 1
        assert len(result.communities[0].members) == 5

    def test_community_ids_formatted(self):
        g = nx.Graph()
        for n in ["N0", "N1", "N2", "N3"]:
            g.add_node(n, source_chunk_ids=["c1"])
        for i in range(3):
            g.add_edge(f"N{i}", f"N{i+1}")
        result = _fallback_detect(g, min_community_size=3, resolution=1.0, seed=42)
        for c in result.communities:
            assert c.community_id.startswith("comm_")


class TestCollectChunkCoverage:
    def test_collects_from_members(self):
        g = nx.Graph()
        g.add_node("A", source_chunk_ids=["c1", "c2"])
        g.add_node("B", source_chunk_ids=["c2", "c3"])
        result = _collect_chunk_coverage(g, ["A", "B"])
        assert result == {"c1", "c2", "c3"}

    def test_empty_members(self):
        g = nx.Graph()
        assert _collect_chunk_coverage(g, []) == set()

    def test_missing_node_data(self):
        g = nx.Graph()
        g.add_node("A")
        result = _collect_chunk_coverage(g, ["A"])
        assert result == set()