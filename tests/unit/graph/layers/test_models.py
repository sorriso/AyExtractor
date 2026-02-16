# tests/unit/graph/layers/test_models.py — v1
"""Tests for graph/layers/models.py — community detection models."""

from __future__ import annotations

from ayextractor.graph.layers.models import Community, CommunityHierarchy, CommunitySummary


class TestCommunity:
    def test_create(self):
        c = Community(community_id="comm_001", level=0, members=["A", "B", "C"])
        assert len(c.members) == 3
        assert c.parent_id is None


class TestCommunityHierarchy:
    def test_empty(self):
        h = CommunityHierarchy()
        assert h.total_communities == 0

    def test_with_communities(self):
        h = CommunityHierarchy(
            communities=[
                Community(community_id="c1", level=0, members=["A", "B"]),
                Community(community_id="c2", level=0, members=["C", "D"]),
            ],
            num_levels=1, resolution=1.0, seed=42,
            total_communities=2, modularity=0.45,
        )
        assert h.seed == 42


class TestCommunitySummary:
    def test_create(self):
        cs = CommunitySummary(
            community_id="c1", level=0, title="Cybersecurity Governance",
            summary="This community covers...", key_entities=["EU", "NIS2"],
            member_count=5,
        )
        assert cs.title == "Cybersecurity Governance"
