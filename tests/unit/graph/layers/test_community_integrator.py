# tests/unit/graph/layers/test_community_integrator.py — v1
"""Tests for graph/layers/community_integrator.py — L1 node injection."""

from __future__ import annotations

import networkx as nx
import pytest

from ayextractor.graph.layers.community_integrator import integrate_communities
from ayextractor.graph.layers.models import Community, CommunityHierarchy


def _build_l2_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("EU", layer=2, entity_type="organization", source_chunk_ids=["c1", "c2", "c3"])
    g.add_node("AI Act", layer=2, entity_type="document", source_chunk_ids=["c1", "c2"])
    g.add_node("ISO 21434", layer=2, entity_type="document", source_chunk_ids=["c2", "c3", "c4"])
    g.add_node("OEM", layer=2, entity_type="organization", source_chunk_ids=["c3", "c4"])
    g.add_edge("EU", "AI Act", relation_type="regulates")
    g.add_edge("ISO 21434", "OEM", relation_type="requires")
    return g


def _build_hierarchy() -> CommunityHierarchy:
    return CommunityHierarchy(
        communities=[
            Community(
                community_id="comm_000", level=0,
                members=["EU", "AI Act"],
                chunk_coverage=["c1", "c2", "c3"],
            ),
            Community(
                community_id="comm_001", level=0,
                members=["ISO 21434", "OEM"],
                chunk_coverage=["c2", "c3", "c4"],
            ),
        ],
        num_levels=1, resolution=1.0, seed=42,
        total_communities=2, modularity=0.5,
    )


class TestIntegrateCommunities:
    def test_creates_l1_nodes(self):
        g = _build_l2_graph()
        h = _build_hierarchy()
        integrate_communities(g, h)
        assert g.has_node("comm_000")
        assert g.has_node("comm_001")
        assert g.nodes["comm_000"]["layer"] == 1
        assert g.nodes["comm_001"]["layer"] == 1

    def test_creates_encompasses_edges(self):
        g = _build_l2_graph()
        h = _build_hierarchy()
        integrate_communities(g, h)
        assert g.has_edge("comm_000", "EU")
        assert g.has_edge("comm_000", "AI Act")
        assert g["comm_000"]["EU"]["relation_type"] == "encompasses"

    def test_sets_community_id_on_members(self):
        g = _build_l2_graph()
        h = _build_hierarchy()
        integrate_communities(g, h)
        assert g.nodes["EU"]["community_id"] == "comm_000"
        assert g.nodes["OEM"]["community_id"] == "comm_001"

    def test_related_to_edges_on_shared_chunks(self):
        """Communities sharing >=3 chunks get related_to edge."""
        g = _build_l2_graph()
        h = _build_hierarchy()
        integrate_communities(g, h)
        # comm_000 has c1,c2,c3 and comm_001 has c2,c3,c4 → 2 shared (c2,c3)
        # MIN_SHARED_CHUNKS=3, so no edge
        assert not g.has_edge("comm_000", "comm_001")

    def test_related_to_edges_when_enough_shared(self):
        g = _build_l2_graph()
        h = CommunityHierarchy(
            communities=[
                Community(community_id="comm_000", level=0, members=["EU", "AI Act"],
                          chunk_coverage=["c1", "c2", "c3", "c4"]),
                Community(community_id="comm_001", level=0, members=["ISO 21434", "OEM"],
                          chunk_coverage=["c2", "c3", "c4", "c5"]),
            ],
            num_levels=1, resolution=1.0, total_communities=2, modularity=0.5,
        )
        integrate_communities(g, h)
        # 3 shared: c2, c3, c4
        assert g.has_edge("comm_000", "comm_001")
        assert g["comm_000"]["comm_001"]["relation_type"] == "related_to"

    def test_empty_hierarchy_no_change(self):
        g = _build_l2_graph()
        nodes_before = g.number_of_nodes()
        integrate_communities(g, CommunityHierarchy())
        assert g.number_of_nodes() == nodes_before

    def test_l1_node_attributes(self):
        g = _build_l2_graph()
        h = _build_hierarchy()
        integrate_communities(g, h)
        node = g.nodes["comm_000"]
        assert node["entity_type"] == "community_topic"
        assert node["is_literal"] is False
        assert "EU" in node["members"]
        assert "AI Act" in node["members"]

    def test_missing_member_skipped(self):
        """If a member doesn't exist in graph, skip encompasses edge."""
        g = _build_l2_graph()
        h = CommunityHierarchy(
            communities=[
                Community(community_id="comm_000", level=0,
                          members=["EU", "NONEXISTENT"],
                          chunk_coverage=["c1"]),
            ],
            num_levels=1, total_communities=1, modularity=0.3,
        )
        integrate_communities(g, h)
        assert g.has_edge("comm_000", "EU")
        assert not g.has_edge("comm_000", "NONEXISTENT")
