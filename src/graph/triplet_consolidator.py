# src/graph/triplet_consolidator.py — v1
"""Triplet deduplication and merge — Pass 3 of the consolidation pipeline.

Applies entity and relation normalizations, then groups and merges
identical triplets. Confidence is boosted for multi-occurrence triplets.

See spec §13.5 for full documentation.
"""

from __future__ import annotations

import hashlib
import logging

from ayextractor.core.models import (
    ConsolidatedTriplet,
    QualifiedTriplet,
    TemporalScope,
)

logger = logging.getLogger(__name__)

CONFIDENCE_BOOST_FACTOR = 0.1


def _triplet_hash(subject: str, predicate: str, obj: str) -> str:
    """Compute deterministic hash for a normalized triplet key."""
    key = f"{subject.lower().strip()}|{predicate.lower().strip()}|{obj.lower().strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _merge_qualifiers(
    groups: list[dict[str, str] | None],
) -> dict[str, str | list[str]] | None:
    """Merge qualifier dicts from multiple triplets.

    Same key + same value -> keep single value.
    Same key + different values -> store as list.
    """
    merged: dict[str, str | list[str]] = {}
    for quals in groups:
        if quals is None:
            continue
        for k, v in quals.items():
            if k not in merged:
                merged[k] = v
            else:
                existing = merged[k]
                if isinstance(existing, list):
                    if v not in existing:
                        existing.append(v)
                elif existing != v:
                    merged[k] = [existing, v]
    return merged if merged else None


def _pick_temporal_scope(
    scopes: list[TemporalScope | None],
) -> TemporalScope | None:
    """Pick the most precise temporal scope from a group.

    Prefers finer granularity (day > month > year).
    """
    granularity_rank = {"day": 5, "month": 4, "quarter": 3, "year": 2, "decade": 1}
    valid = [s for s in scopes if s is not None]
    if not valid:
        return None
    return max(
        valid,
        key=lambda s: granularity_rank.get(s.granularity or "", 0),
    )


def consolidate_triplets(
    triplets: list[QualifiedTriplet],
    entity_mapping: dict[str, str],
    relation_mapping: dict[str, str],
    boost_confidence: bool = True,
) -> list[ConsolidatedTriplet]:
    """Run Pass 3: normalize, group, and merge triplets.

    Args:
        triplets: Raw qualified triplets from all chunks.
        entity_mapping: alias -> canonical_name mapping from Pass 1.
        relation_mapping: raw_predicate -> canonical_relation from Pass 2.
        boost_confidence: Apply confidence boost for multi-occurrence triplets.

    Returns:
        List of consolidated (deduplicated) triplets.
    """
    if not triplets:
        return []

    # Group by normalized hash
    groups: dict[str, list[QualifiedTriplet]] = {}
    normalized_keys: dict[str, tuple[str, str, str]] = {}

    for t in triplets:
        subj = entity_mapping.get(t.subject.strip(), t.subject.strip())
        pred = relation_mapping.get(t.predicate.strip(), t.predicate.strip())
        obj = entity_mapping.get(t.object.strip(), t.object.strip())

        h = _triplet_hash(subj, pred, obj)
        groups.setdefault(h, []).append(t)
        if h not in normalized_keys:
            normalized_keys[h] = (subj, pred, obj)

    # Merge groups
    results: list[ConsolidatedTriplet] = []

    for h, group in groups.items():
        subj, pred, obj = normalized_keys[h]

        # Collect data from all members
        all_chunks: list[str] = []
        original_forms: set[str] = set()
        context_sentences: list[str] = []
        max_confidence = 0.0
        all_qualifiers: list[dict[str, str] | None] = []
        all_scopes: list[TemporalScope | None] = []

        for t in group:
            all_chunks.append(t.source_chunk_id)
            original_forms.add(t.predicate)
            if t.context_sentence:
                context_sentences.append(t.context_sentence)
            max_confidence = max(max_confidence, t.confidence)
            all_qualifiers.append(t.qualifiers)
            all_scopes.append(t.temporal_scope)

        occurrence_count = len(group)

        # Confidence boost: spec formula
        final_confidence = max_confidence
        if boost_confidence and occurrence_count > 1:
            final_confidence = min(
                1.0,
                max_confidence * (1 + CONFIDENCE_BOOST_FACTOR * (occurrence_count - 1)),
            )

        results.append(
            ConsolidatedTriplet(
                subject=subj,
                predicate=pred,
                object=obj,
                source_chunk_ids=sorted(set(all_chunks)),
                occurrence_count=occurrence_count,
                confidence=round(final_confidence, 4),
                original_forms=sorted(original_forms),
                qualifiers=_merge_qualifiers(all_qualifiers),
                temporal_scope=_pick_temporal_scope(all_scopes),
                context_sentences=list(dict.fromkeys(context_sentences)),  # dedup preserve order
            )
        )

    logger.info(
        "Triplet consolidation: %d raw -> %d consolidated (dedup ratio %.2f)",
        len(triplets),
        len(results),
        1 - len(results) / max(len(triplets), 1),
    )

    return results
