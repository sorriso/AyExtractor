# tests/unit/graph/test_relation_normalizer.py — v1
"""Tests for graph/relation_normalizer.py — predicate taxonomy mapping."""

from __future__ import annotations

import pytest

from ayextractor.core.models import QualifiedTriplet
from ayextractor.graph.relation_normalizer import (
    build_relation_mapping,
    extract_unique_predicates,
    normalize_relations,
)


def _make_triplet(predicate: str):
    return QualifiedTriplet(
        subject="A",
        predicate=predicate,
        object="B",
        source_chunk_id="c1",
        confidence=0.8,
        context_sentence="test",
    )


class TestExtractUniquePredicates:
    def test_extracts_unique(self):
        triplets = [_make_triplet("regulates"), _make_triplet("regulates"), _make_triplet("requires")]
        result = extract_unique_predicates(triplets)
        assert result == ["regulates", "requires"]

    def test_empty_input(self):
        assert extract_unique_predicates([]) == []

    def test_strips_whitespace(self):
        triplets = [_make_triplet("  regulates  ")]
        result = extract_unique_predicates(triplets)
        assert result == ["regulates"]


class TestNormalizeRelations:
    def test_known_relations_stay_in_taxonomy(self):
        triplets = [_make_triplet("regulates"), _make_triplet("requires")]
        result = normalize_relations(triplets)
        canonical_names = {e.canonical_relation for e in result}
        assert "regulates" in canonical_names
        assert "requires" in canonical_names

    def test_unknown_relation_extensible_creates_entry(self):
        triplets = [_make_triplet("collaborates_with")]
        result = normalize_relations(triplets, extensible=True)
        canonical_names = {e.canonical_relation for e in result}
        assert "collaborates_with" in canonical_names
        # Check it's in the extended category
        new_entries = [e for e in result if e.category == "extended"]
        assert len(new_entries) >= 1

    def test_unknown_relation_not_extensible_maps_to_related_to(self):
        triplets = [_make_triplet("collaborates_with")]
        # Not extensible: no new entries created
        result = normalize_relations(triplets, extensible=False)
        extended = [e for e in result if e.category == "extended"]
        assert len(extended) == 0

    def test_empty_input_returns_base_taxonomy(self):
        from ayextractor.graph.taxonomy import DEFAULT_RELATION_TAXONOMY
        result = normalize_relations([])
        assert len(result) == len(DEFAULT_RELATION_TAXONOMY)


class TestBuildRelationMapping:
    def test_known_predicate_maps_to_canonical(self):
        triplets = [_make_triplet("regulates"), _make_triplet("réglemente")]
        mapping = build_relation_mapping(triplets)
        assert mapping["regulates"] == "regulates"
        assert mapping["réglemente"] == "regulates"

    def test_unknown_extensible_maps_to_normalized_form(self):
        triplets = [_make_triplet("works with")]
        mapping = build_relation_mapping(triplets, extensible=True)
        assert mapping["works with"] == "works_with"

    def test_unknown_not_extensible_maps_to_related_to(self):
        triplets = [_make_triplet("works with")]
        mapping = build_relation_mapping(triplets, extensible=False)
        assert mapping["works with"] == "related_to"

    def test_empty_input(self):
        assert build_relation_mapping([]) == {}
