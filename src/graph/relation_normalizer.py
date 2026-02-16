# src/graph/relation_normalizer.py — v1
"""Relation normalization — Pass 2 of the triplet consolidation pipeline.

Maps raw predicates to canonical relations using the base taxonomy.
Unknown relations fall back to 'related_to'.

See spec §13.4 for full documentation.
"""

from __future__ import annotations

import logging

from ayextractor.core.models import QualifiedTriplet, RelationTaxonomyEntry
from ayextractor.graph.taxonomy import (
    DEFAULT_RELATION_TAXONOMY,
    find_canonical,
)

logger = logging.getLogger(__name__)


def extract_unique_predicates(triplets: list[QualifiedTriplet]) -> list[str]:
    """Extract all unique predicate forms from triplets."""
    return sorted({t.predicate.strip() for t in triplets if t.predicate.strip()})


def normalize_relations(
    triplets: list[QualifiedTriplet],
    extensible: bool = True,
) -> list[RelationTaxonomyEntry]:
    """Run Pass 2: map raw predicates to canonical taxonomy.

    Uses exact-match against the base taxonomy. Unmatched predicates
    are either added as new entries (if extensible=True) or mapped
    to 'related_to'.

    Args:
        triplets: Raw triplets (predicates not yet normalized).
        extensible: Allow creation of new taxonomy entries for unknown predicates.

    Returns:
        Extended taxonomy entries (base + any new ones).
    """
    unique_preds = extract_unique_predicates(triplets)
    if not unique_preds:
        return list(DEFAULT_RELATION_TAXONOMY)

    # Try matching each predicate against taxonomy
    matched: dict[str, str] = {}  # raw_form -> canonical
    unmatched: list[str] = []

    for pred in unique_preds:
        canonical = find_canonical(pred)
        if canonical:
            matched[pred] = canonical
        else:
            unmatched.append(pred)

    # Build extended taxonomy
    result = list(DEFAULT_RELATION_TAXONOMY)

    # Group unmatched by normalized lowercase form
    if unmatched:
        if extensible:
            # Create new entries for genuinely new predicates
            new_entries: dict[str, list[str]] = {}
            for pred in unmatched:
                key = pred.lower().strip().replace(" ", "_")
                new_entries.setdefault(key, []).append(pred)
                matched[pred] = key

            for canonical_key, forms in new_entries.items():
                entry = RelationTaxonomyEntry(
                    canonical_relation=canonical_key,
                    original_forms=forms,
                    category="extended",
                    is_directional=True,
                )
                result.append(entry)

            logger.info(
                "Relation normalization: %d matched, %d new entries created",
                len(unique_preds) - len(unmatched),
                len(new_entries),
            )
        else:
            # Map all unmatched to related_to
            for pred in unmatched:
                matched[pred] = "related_to"
            logger.info(
                "Relation normalization: %d matched, %d mapped to related_to",
                len(unique_preds) - len(unmatched),
                len(unmatched),
            )

    return result


def build_relation_mapping(
    triplets: list[QualifiedTriplet],
    extensible: bool = True,
) -> dict[str, str]:
    """Build raw_predicate -> canonical_relation mapping.

    Args:
        triplets: Raw triplets.
        extensible: Allow new taxonomy entries.

    Returns:
        Mapping from every raw predicate form to its canonical relation.
    """
    unique_preds = extract_unique_predicates(triplets)
    mapping: dict[str, str] = {}

    for pred in unique_preds:
        canonical = find_canonical(pred)
        if canonical:
            mapping[pred] = canonical
        elif extensible:
            mapping[pred] = pred.lower().strip().replace(" ", "_")
        else:
            mapping[pred] = "related_to"

    return mapping
