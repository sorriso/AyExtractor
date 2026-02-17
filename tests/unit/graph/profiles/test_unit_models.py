# tests/unit/graph/profiles/test_models.py — v1
"""Tests for graph/profiles/models.py — entity and relation profiles."""

from __future__ import annotations

from ayextractor.graph.profiles.models import EntityProfile, RelationProfile


class TestEntityProfile:
    def test_create(self):
        ep = EntityProfile(
            canonical_name="European Union", entity_type="organization",
            profile_text="The EU is a political and economic union.",
            key_relations=["regulates cybersecurity", "member of UN"],
            community_id="c1",
        )
        assert ep.embedding is None

    def test_with_embedding(self):
        ep = EntityProfile(
            canonical_name="Test", entity_type="concept",
            profile_text="A test entity.", embedding=[0.1, 0.2, 0.3],
        )
        assert len(ep.embedding) == 3


class TestRelationProfile:
    def test_create(self):
        rp = RelationProfile(
            subject="EU", predicate="regulates", object="cybersecurity",
            profile_text="The EU regulates cybersecurity through directives.",
        )
        assert rp.qualifiers is None
        assert rp.temporal_scope is None
