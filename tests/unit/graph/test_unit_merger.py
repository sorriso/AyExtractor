# tests/unit/graph/test_merger.py — v1
"""Tests for graph/merger.py — full 3-pass consolidation pipeline orchestrator."""

from __future__ import annotations

import pytest

from ayextractor.core.models import QualifiedTriplet
from ayextractor.graph.merger import MergerResult, run_merger_pipeline


def _make_triplet(subject, predicate, obj, chunk_id="c1", confidence=0.8):
    return QualifiedTriplet(
        subject=subject,
        predicate=predicate,
        object=obj,
        source_chunk_id=chunk_id,
        confidence=confidence,
        context_sentence=f"{subject} {predicate} {obj}",
    )


class TestRunMergerPipeline:
    @pytest.mark.asyncio
    async def test_empty_input(self):
        result = await run_merger_pipeline([])
        assert isinstance(result, MergerResult)
        assert result.stats["total_raw_triplets"] == 0
        assert result.consolidated_triplets == []

    @pytest.mark.asyncio
    async def test_single_triplet_no_dedup(self):
        triplets = [_make_triplet("A", "regulates", "B")]
        result = await run_merger_pipeline(triplets)
        assert len(result.consolidated_triplets) == 1
        assert result.consolidated_triplets[0].subject == "A"

    @pytest.mark.asyncio
    async def test_duplicate_triplets_merged(self):
        triplets = [
            _make_triplet("EU", "regulates", "AI", "c1"),
            _make_triplet("EU", "regulates", "AI", "c2"),
        ]
        result = await run_merger_pipeline(triplets)
        assert len(result.consolidated_triplets) == 1
        assert result.consolidated_triplets[0].occurrence_count == 2

    @pytest.mark.asyncio
    async def test_entity_normalizations_produced(self):
        triplets = [_make_triplet("A", "rel", "B")]
        result = await run_merger_pipeline(triplets)
        assert len(result.entity_normalizations) >= 2  # A and B

    @pytest.mark.asyncio
    async def test_relation_taxonomy_produced(self):
        triplets = [_make_triplet("A", "regulates", "B")]
        result = await run_merger_pipeline(triplets)
        assert len(result.relation_taxonomy) > 0
        canonical_names = {e.canonical_relation for e in result.relation_taxonomy}
        assert "regulates" in canonical_names

    @pytest.mark.asyncio
    async def test_stats_populated(self):
        triplets = [
            _make_triplet("A", "rel", "B", "c1"),
            _make_triplet("C", "rel", "D", "c2"),
        ]
        result = await run_merger_pipeline(triplets)
        assert result.stats["total_raw_triplets"] == 2
        assert result.stats["unique_entities_before"] == 4
        assert result.stats["consolidated_triplets"] == 2

    @pytest.mark.asyncio
    async def test_known_relation_normalized(self):
        """Known taxonomy relation maps correctly through the pipeline."""
        triplets = [_make_triplet("X", "réglemente", "Y")]
        result = await run_merger_pipeline(triplets)
        assert result.consolidated_triplets[0].predicate == "regulates"

    @pytest.mark.asyncio
    async def test_dedup_ratio_computed(self):
        triplets = [
            _make_triplet("A", "rel", "B", "c1"),
            _make_triplet("A", "rel", "B", "c2"),
            _make_triplet("A", "rel", "B", "c3"),
        ]
        result = await run_merger_pipeline(triplets)
        assert result.stats["dedup_ratio"] > 0
