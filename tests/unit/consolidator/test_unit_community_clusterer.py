# tests/unit/consolidator/test_community_clusterer.py — v1
"""Tests for consolidator/community_clusterer.py."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import networkx as nx
import pytest

from ayextractor.consolidator.community_clusterer import (
    _build_cnode_graph,
    _generate_cluster_name,
    _run_leiden,
    cluster_corpus,
)
from ayextractor.consolidator.models import ClusteringReport, TNode


class _FakeStore:
    """Fake graph store that wraps a NetworkX graph."""

    def __init__(self, graph: nx.Graph | None = None) -> None:
        self._graph = graph or nx.Graph()
        self._tnodes: list[TNode] = []

    def to_networkx(self) -> nx.Graph:
        return self._graph

    def get_tnodes(self) -> list[TNode]:
        return self._tnodes

    def upsert_node(self, name: str, data: dict) -> None:
        pass  # no-op for tests


class TestClusterCorpus:
    def test_empty_graph_returns_zero(self):
        store = _FakeStore(nx.Graph())
        report = cluster_corpus(store, min_cluster_size=3)
        assert isinstance(report, ClusteringReport)
        assert report.new_tnodes == 0
        assert report.clusters_found == 0

    def test_small_graph_below_threshold(self):
        g = nx.Graph()
        g.add_nodes_from(["A", "B"], node_type="cnode")
        store = _FakeStore(g)
        report = cluster_corpus(store, min_cluster_size=3)
        assert report.new_tnodes == 0

    def test_clusters_above_min_size(self):
        g = nx.Graph()
        # Create two clear clusters
        for i in range(5):
            g.add_node(f"a{i}", node_type="cnode")
        for i in range(5):
            g.add_node(f"b{i}", node_type="cnode")
        # Dense edges within clusters
        for i in range(5):
            for j in range(i + 1, 5):
                g.add_edge(f"a{i}", f"a{j}")
                g.add_edge(f"b{i}", f"b{j}")
        # Sparse cross-cluster edge
        g.add_edge("a0", "b0")

        store = _FakeStore(g)
        report = cluster_corpus(store, min_cluster_size=3, seed=42)
        assert report.clusters_found >= 1

    def test_min_cluster_size_filter(self):
        g = nx.Graph()
        # One big cluster, one small (2 nodes)
        for i in range(4):
            g.add_node(f"big{i}", node_type="cnode")
            for j in range(i):
                g.add_edge(f"big{j}", f"big{i}")
        g.add_node("small0", node_type="cnode")
        g.add_node("small1", node_type="cnode")
        g.add_edge("small0", "small1")

        store = _FakeStore(g)
        report = cluster_corpus(store, min_cluster_size=3, seed=42)
        # Small cluster should be filtered out
        assert report.clusters_found >= 1


class TestRunLeiden:
    def test_empty_graph(self):
        g = nx.Graph()
        result = _run_leiden(g)
        assert result == {}

    def test_connected_graph(self):
        g = nx.complete_graph(5)
        result = _run_leiden(g, seed=42)
        # Should produce at least 1 community
        assert len(result) >= 1
        # All nodes should be assigned
        all_nodes = set()
        for members in result.values():
            all_nodes.update(members)
        assert len(all_nodes) == 5

    def test_disconnected_components(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2)])
        g.add_edges_from([(10, 11), (11, 12)])
        result = _run_leiden(g, seed=42)
        assert len(result) >= 2


class TestBuildCnodeGraph:
    def test_filters_cnode_type(self):
        store = MagicMock()
        g = nx.Graph()
        g.add_node("c1", node_type="cnode")
        g.add_node("t1", node_type="tnode")
        g.add_edge("c1", "t1")
        store.to_networkx.return_value = g

        result = _build_cnode_graph(store)
        assert "c1" in result.nodes()
        assert "t1" not in result.nodes()

    def test_no_to_networkx_returns_empty(self):
        store = MagicMock(spec=[])  # No to_networkx
        del store.to_networkx
        result = _build_cnode_graph(store)
        assert result.number_of_nodes() == 0


class TestGenerateClusterName:
    def test_small_cluster(self):
        name = _generate_cluster_name(["alpha", "beta", "gamma"])
        assert "cluster_" in name

    def test_large_cluster(self):
        name = _generate_cluster_name([f"n{i}" for i in range(10)])
        assert "cluster_10" in name
