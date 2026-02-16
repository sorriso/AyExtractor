# src/graph/decay_manager.py — v1
"""Decay manager — temporal confidence decay for aging facts.

Applies exponential decay to triplet/node confidence based on age.
Facts without temporal scope or recent corroboration lose confidence
over time, signaling staleness.

See spec §13.9.8 for decay formula and parameters.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import networkx as nx

logger = logging.getLogger(__name__)

# Default half-life in days (confidence halves after this many days)
DEFAULT_HALF_LIFE_DAYS = 365.0
# Minimum confidence floor (never decays below this)
MIN_CONFIDENCE = 0.05


@dataclass
class DecayStats:
    """Statistics from a decay pass."""

    nodes_decayed: int = 0
    edges_decayed: int = 0
    nodes_below_threshold: int = 0
    edges_below_threshold: int = 0


def compute_decay_factor(
    age_days: float,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Compute exponential decay factor for a given age.

    Formula: factor = 2^(-age_days / half_life_days)

    Args:
        age_days: Age of the fact in days.
        half_life_days: Days until confidence halves.

    Returns:
        Decay multiplier in [0, 1].
    """
    if age_days <= 0:
        return 1.0
    if half_life_days <= 0:
        return MIN_CONFIDENCE
    return max(MIN_CONFIDENCE, math.pow(2, -age_days / half_life_days))


def apply_decay(
    graph: nx.Graph,
    reference_date: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    prune_threshold: float = 0.0,
) -> DecayStats:
    """Apply confidence decay to all nodes and edges in the graph.

    Reads 'last_updated_at' (ISO string) from node/edge attributes.
    Multiplies existing 'confidence' by decay factor.

    Args:
        graph: NetworkX graph to update (mutated in place).
        reference_date: Current date for age computation (default: now UTC).
        half_life_days: Confidence half-life in days.
        prune_threshold: Remove elements with confidence below this (0 = no pruning).

    Returns:
        DecayStats with counts.
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    stats = DecayStats()

    # Decay nodes
    nodes_to_remove: list[str] = []
    for node, data in graph.nodes(data=True):
        last_updated = data.get("last_updated_at")
        confidence = data.get("confidence", 1.0)
        if last_updated is None:
            continue

        age_days = _compute_age_days(last_updated, reference_date)
        factor = compute_decay_factor(age_days, half_life_days)
        new_conf = max(MIN_CONFIDENCE, confidence * factor)
        data["confidence"] = round(new_conf, 6)
        data["decay_factor"] = round(factor, 6)
        stats.nodes_decayed += 1

        if new_conf < prune_threshold:
            nodes_to_remove.append(node)
            stats.nodes_below_threshold += 1

    # Decay edges
    edges_to_remove: list[tuple[str, str]] = []
    for u, v, data in graph.edges(data=True):
        last_updated = data.get("last_updated_at")
        confidence = data.get("confidence", 1.0)
        if last_updated is None:
            continue

        age_days = _compute_age_days(last_updated, reference_date)
        factor = compute_decay_factor(age_days, half_life_days)
        new_conf = max(MIN_CONFIDENCE, confidence * factor)
        data["confidence"] = round(new_conf, 6)
        data["decay_factor"] = round(factor, 6)
        stats.edges_decayed += 1

        if new_conf < prune_threshold:
            edges_to_remove.append((u, v))
            stats.edges_below_threshold += 1

    # Prune if threshold set
    if prune_threshold > 0:
        for u, v in edges_to_remove:
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)
        for node in nodes_to_remove:
            if graph.has_node(node):
                graph.remove_node(node)

    logger.info(
        "Decay applied: %d nodes, %d edges decayed. %d nodes, %d edges below threshold.",
        stats.nodes_decayed, stats.edges_decayed,
        stats.nodes_below_threshold, stats.edges_below_threshold,
    )
    return stats


def _compute_age_days(iso_timestamp: str, reference: datetime) -> float:
    """Parse ISO timestamp and compute age in days."""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = reference - dt
        return max(0.0, delta.total_seconds() / 86400)
    except (ValueError, TypeError):
        return 0.0
