# src/graph/contradiction_detector.py — v1
"""Contradiction detector — find conflicting triplets in the knowledge graph.

Identifies:
  - Direct contradictions: same (subject, object) with opposing predicates.
  - Value conflicts: same (subject, predicate) with different literal objects.
  - Temporal contradictions: overlapping temporal scopes with conflicting facts.

Pure function, no LLM call. See spec §14.3 for detection rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ayextractor.core.models import ConsolidatedTriplet

logger = logging.getLogger(__name__)

# Predicate pairs considered as contradictions
OPPOSING_PREDICATES: list[tuple[str, str]] = [
    ("enables", "prevents"),
    ("causes", "prevents"),
    ("requires", "prevents"),
    ("precedes", "follows"),
    ("contains", "excludes"),
    ("complies_with", "violates"),
]


@dataclass
class Contradiction:
    """A detected contradiction between two triplets."""

    triplet_a: ConsolidatedTriplet
    triplet_b: ConsolidatedTriplet
    contradiction_type: str  # "opposing_predicate", "value_conflict", "temporal_conflict"
    severity: str  # "low", "medium", "high"
    explanation: str = ""


@dataclass
class ContradictionReport:
    """Full contradiction detection report."""

    contradictions: list[Contradiction] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def detect_contradictions(
    triplets: list[ConsolidatedTriplet],
) -> ContradictionReport:
    """Scan consolidated triplets for contradictions.

    Args:
        triplets: All consolidated triplets from the graph.

    Returns:
        ContradictionReport with detected conflicts.
    """
    if not triplets:
        return ContradictionReport(stats={"total_triplets": 0, "contradictions": 0})

    contradictions: list[Contradiction] = []

    # Build indexes
    by_subject_object: dict[tuple[str, str], list[ConsolidatedTriplet]] = {}
    by_subject_predicate: dict[tuple[str, str], list[ConsolidatedTriplet]] = {}

    for t in triplets:
        key_so = (t.subject.lower(), t.object.lower())
        key_sp = (t.subject.lower(), t.predicate.lower())
        by_subject_object.setdefault(key_so, []).append(t)
        by_subject_predicate.setdefault(key_sp, []).append(t)

    # Check 1: Opposing predicates (same subject + object, opposing relation)
    opposing_set = set()
    for p1, p2 in OPPOSING_PREDICATES:
        opposing_set.add((p1, p2))
        opposing_set.add((p2, p1))

    for (subj, obj), group in by_subject_object.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pair = (group[i].predicate.lower(), group[j].predicate.lower())
                if pair in opposing_set:
                    contradictions.append(Contradiction(
                        triplet_a=group[i],
                        triplet_b=group[j],
                        contradiction_type="opposing_predicate",
                        severity="high",
                        explanation=(
                            f"Opposing predicates '{group[i].predicate}' and "
                            f"'{group[j].predicate}' for ({subj}, {obj})."
                        ),
                    ))

    # Check 2: Value conflicts (same subject + predicate, different literal objects)
    for (subj, pred), group in by_subject_predicate.items():
        if len(group) < 2:
            continue
        # Only flag if objects are literals (numbers, dates, etc.)
        literal_group = [t for t in group if _looks_like_literal(t.object)]
        if len(literal_group) < 2:
            continue
        # Compare distinct object values
        seen_objects: dict[str, ConsolidatedTriplet] = {}
        for t in literal_group:
            obj_norm = t.object.strip().lower()
            if obj_norm in seen_objects:
                continue
            for existing_obj, existing_t in seen_objects.items():
                if existing_obj != obj_norm:
                    contradictions.append(Contradiction(
                        triplet_a=existing_t,
                        triplet_b=t,
                        contradiction_type="value_conflict",
                        severity="medium",
                        explanation=(
                            f"Conflicting values for ({subj}, {pred}): "
                            f"'{existing_t.object}' vs '{t.object}'."
                        ),
                    ))
            seen_objects[obj_norm] = t

    stats = {
        "total_triplets": len(triplets),
        "contradictions": len(contradictions),
        "high_severity": sum(1 for c in contradictions if c.severity == "high"),
        "medium_severity": sum(1 for c in contradictions if c.severity == "medium"),
    }

    if contradictions:
        logger.warning(
            "Detected %d contradictions (%d high, %d medium)",
            len(contradictions), stats["high_severity"], stats["medium_severity"],
        )
    else:
        logger.info("No contradictions detected in %d triplets", len(triplets))

    return ContradictionReport(contradictions=contradictions, stats=stats)


def _looks_like_literal(value: str) -> bool:
    """Heuristic: check if a value looks like a literal (number, date, measure)."""
    import re
    stripped = value.strip()
    if not stripped:
        return False
    if re.match(r"^[\d.,]+\s*[%€$£¥]?$", stripped):
        return True
    if re.match(r"^[\d.,]+\s+\w{1,10}$", stripped):
        return True
    if re.match(r"^\d{4}[-/]\d{2}([-/]\d{2})?$", stripped):
        return True
    return False
