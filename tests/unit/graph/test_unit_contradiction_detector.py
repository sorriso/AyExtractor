# tests/unit/graph/test_contradiction_detector.py — v1
"""Tests for graph/contradiction_detector.py."""

from __future__ import annotations

import pytest

from ayextractor.core.models import ConsolidatedTriplet
from ayextractor.graph.contradiction_detector import (
    ContradictionReport,
    detect_contradictions,
    _looks_like_literal,
)


def _ct(subject, predicate, obj, confidence=0.8):
    return ConsolidatedTriplet(
        subject=subject, predicate=predicate, object=obj,
        source_chunk_ids=["c1"], occurrence_count=1, confidence=confidence,
    )


class TestDetectContradictions:
    def test_empty_input(self):
        result = detect_contradictions([])
        assert isinstance(result, ContradictionReport)
        assert result.stats["total_triplets"] == 0

    def test_no_contradictions(self):
        triplets = [
            _ct("EU", "regulates", "AI"),
            _ct("ISO", "requires", "audit"),
        ]
        result = detect_contradictions(triplets)
        assert len(result.contradictions) == 0

    def test_opposing_predicates_detected(self):
        triplets = [
            _ct("X", "enables", "Y"),
            _ct("X", "prevents", "Y"),
        ]
        result = detect_contradictions(triplets)
        assert len(result.contradictions) == 1
        assert result.contradictions[0].contradiction_type == "opposing_predicate"
        assert result.contradictions[0].severity == "high"

    def test_value_conflict_detected(self):
        triplets = [
            _ct("Company", "has_attribute", "500"),
            _ct("Company", "has_attribute", "1000"),
        ]
        result = detect_contradictions(triplets)
        assert len(result.contradictions) == 1
        assert result.contradictions[0].contradiction_type == "value_conflict"
        assert result.contradictions[0].severity == "medium"

    def test_same_literal_not_conflict(self):
        triplets = [
            _ct("Company", "has_attribute", "500"),
            _ct("Company", "has_attribute", "500"),
        ]
        result = detect_contradictions(triplets)
        assert len(result.contradictions) == 0

    def test_non_literal_objects_not_value_conflict(self):
        """Named entities as objects should NOT trigger value conflict."""
        triplets = [
            _ct("EU", "regulates", "AI systems"),
            _ct("EU", "regulates", "financial markets"),
        ]
        result = detect_contradictions(triplets)
        # Non-literal objects = not a value conflict
        assert len(result.contradictions) == 0

    def test_multiple_opposing_pairs(self):
        triplets = [
            _ct("A", "enables", "B"),
            _ct("A", "prevents", "B"),
            _ct("C", "causes", "D"),
            _ct("C", "prevents", "D"),
        ]
        result = detect_contradictions(triplets)
        assert len(result.contradictions) == 2
        assert result.stats["high_severity"] == 2

    def test_stats_populated(self):
        triplets = [
            _ct("A", "enables", "B"),
            _ct("A", "prevents", "B"),
        ]
        result = detect_contradictions(triplets)
        assert result.stats["total_triplets"] == 2
        assert result.stats["contradictions"] == 1


class TestLooksLikeLiteral:
    def test_number(self):
        assert _looks_like_literal("500") is True

    def test_percentage(self):
        assert _looks_like_literal("15%") is True

    def test_named_entity(self):
        assert _looks_like_literal("European Union") is False

    def test_date(self):
        assert _looks_like_literal("2025-01") is True

    def test_empty(self):
        assert _looks_like_literal("") is False
