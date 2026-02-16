# tests/unit/graph/test_triplet_consolidator.py — v1
"""Tests for graph/triplet_consolidator.py — dedup and merge pass."""

from __future__ import annotations

import pytest

from ayextractor.core.models import QualifiedTriplet, TemporalScope
from ayextractor.graph.triplet_consolidator import (
    consolidate_triplets,
    _merge_qualifiers,
    _pick_temporal_scope,
    _triplet_hash,
)


def _make_triplet(
    subject="A", predicate="rel", obj="B", chunk_id="c1",
    confidence=0.8, qualifiers=None, temporal_scope=None,
):
    return QualifiedTriplet(
        subject=subject,
        predicate=predicate,
        object=obj,
        source_chunk_id=chunk_id,
        confidence=confidence,
        context_sentence=f"{subject} {predicate} {obj}",
        qualifiers=qualifiers,
        temporal_scope=temporal_scope,
    )


class TestTripletHash:
    def test_deterministic(self):
        h1 = _triplet_hash("A", "rel", "B")
        h2 = _triplet_hash("A", "rel", "B")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = _triplet_hash("EU", "Regulates", "AI")
        h2 = _triplet_hash("eu", "regulates", "ai")
        assert h1 == h2

    def test_different_triplets_different_hash(self):
        h1 = _triplet_hash("A", "rel", "B")
        h2 = _triplet_hash("A", "rel", "C")
        assert h1 != h2


class TestMergeQualifiers:
    def test_merge_same_key_same_value(self):
        result = _merge_qualifiers([{"scope": "high-risk"}, {"scope": "high-risk"}])
        assert result == {"scope": "high-risk"}

    def test_merge_same_key_different_values(self):
        result = _merge_qualifiers([{"scope": "high-risk"}, {"scope": "all systems"}])
        assert result == {"scope": ["high-risk", "all systems"]}

    def test_merge_different_keys(self):
        result = _merge_qualifiers([{"scope": "A"}, {"instrument": "B"}])
        assert result == {"scope": "A", "instrument": "B"}

    def test_merge_with_none(self):
        result = _merge_qualifiers([None, {"scope": "A"}, None])
        assert result == {"scope": "A"}

    def test_merge_all_none(self):
        result = _merge_qualifiers([None, None])
        assert result is None


class TestPickTemporalScope:
    def test_picks_finest_granularity(self):
        scopes = [
            TemporalScope(type="point", start="2025", granularity="year", raw_expression="2025"),
            TemporalScope(type="point", start="2025-06", granularity="month", raw_expression="June 2025"),
        ]
        result = _pick_temporal_scope(scopes)
        assert result is not None
        assert result.granularity == "month"

    def test_all_none(self):
        assert _pick_temporal_scope([None, None]) is None

    def test_single_scope(self):
        scope = TemporalScope(type="point", start="2025", granularity="year", raw_expression="2025")
        result = _pick_temporal_scope([scope])
        assert result == scope


class TestConsolidateTriplets:
    def test_deduplication(self):
        """Identical triplets from different chunks are merged."""
        triplets = [
            _make_triplet("EU", "regulates", "AI", "c1", 0.8),
            _make_triplet("EU", "regulates", "AI", "c2", 0.9),
        ]
        mapping_e = {"EU": "EU", "AI": "AI"}
        mapping_r = {"regulates": "regulates"}
        result = consolidate_triplets(triplets, mapping_e, mapping_r)
        assert len(result) == 1
        assert result[0].occurrence_count == 2
        assert set(result[0].source_chunk_ids) == {"c1", "c2"}

    def test_confidence_boost(self):
        """Multi-occurrence triplets get boosted confidence."""
        triplets = [
            _make_triplet("A", "rel", "B", "c1", 0.8),
            _make_triplet("A", "rel", "B", "c2", 0.8),
        ]
        mapping_e = {"A": "A", "B": "B"}
        mapping_r = {"rel": "rel"}
        result = consolidate_triplets(triplets, mapping_e, mapping_r, boost_confidence=True)
        assert result[0].confidence > 0.8

    def test_no_boost_when_disabled(self):
        triplets = [
            _make_triplet("A", "rel", "B", "c1", 0.8),
            _make_triplet("A", "rel", "B", "c2", 0.8),
        ]
        mapping_e = {"A": "A", "B": "B"}
        mapping_r = {"rel": "rel"}
        result = consolidate_triplets(triplets, mapping_e, mapping_r, boost_confidence=False)
        assert result[0].confidence == 0.8

    def test_entity_normalization_applied(self):
        """Aliases are replaced by canonical names before dedup."""
        triplets = [
            _make_triplet("EU", "regulates", "AI", "c1"),
            _make_triplet("European Union", "regulates", "AI", "c2"),
        ]
        mapping_e = {"EU": "European Union", "European Union": "European Union", "AI": "AI"}
        mapping_r = {"regulates": "regulates"}
        result = consolidate_triplets(triplets, mapping_e, mapping_r)
        assert len(result) == 1
        assert result[0].subject == "European Union"
        assert result[0].occurrence_count == 2

    def test_relation_normalization_applied(self):
        triplets = [
            _make_triplet("A", "regulates", "B", "c1"),
            _make_triplet("A", "réglemente", "B", "c2"),
        ]
        mapping_e = {"A": "A", "B": "B"}
        mapping_r = {"regulates": "regulates", "réglemente": "regulates"}
        result = consolidate_triplets(triplets, mapping_e, mapping_r)
        assert len(result) == 1
        assert result[0].predicate == "regulates"

    def test_context_sentences_collected(self):
        triplets = [
            QualifiedTriplet(
                subject="A", predicate="rel", object="B", source_chunk_id="c1",
                confidence=0.8, context_sentence="A relates to B in chunk 1",
            ),
            QualifiedTriplet(
                subject="A", predicate="rel", object="B", source_chunk_id="c2",
                confidence=0.8, context_sentence="A relates to B in chunk 2",
            ),
        ]
        mapping_e = {"A": "A", "B": "B"}
        mapping_r = {"rel": "rel"}
        result = consolidate_triplets(triplets, mapping_e, mapping_r)
        assert len(result[0].context_sentences) == 2

    def test_original_forms_collected(self):
        triplets = [
            _make_triplet("A", "regulates", "B", "c1"),
            _make_triplet("A", "réglemente", "B", "c2"),
        ]
        mapping_e = {"A": "A", "B": "B"}
        mapping_r = {"regulates": "regulates", "réglemente": "regulates"}
        result = consolidate_triplets(triplets, mapping_e, mapping_r)
        assert "regulates" in result[0].original_forms
        assert "réglemente" in result[0].original_forms

    def test_different_triplets_not_merged(self):
        triplets = [
            _make_triplet("A", "rel1", "B"),
            _make_triplet("A", "rel2", "C"),
        ]
        mapping_e = {"A": "A", "B": "B", "C": "C"}
        mapping_r = {"rel1": "rel1", "rel2": "rel2"}
        result = consolidate_triplets(triplets, mapping_e, mapping_r)
        assert len(result) == 2

    def test_empty_input(self):
        assert consolidate_triplets([], {}, {}) == []

    def test_confidence_capped_at_one(self):
        """Boosted confidence must not exceed 1.0."""
        triplets = [_make_triplet("A", "rel", "B", f"c{i}", 0.95) for i in range(10)]
        mapping_e = {"A": "A", "B": "B"}
        mapping_r = {"rel": "rel"}
        result = consolidate_triplets(triplets, mapping_e, mapping_r)
        assert result[0].confidence <= 1.0
