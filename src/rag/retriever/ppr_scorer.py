# src/rag/retriever/ppr_scorer.py — v2
"""PPR scorer — Personalized PageRank on knowledge graph.

Computes structural relevance scores using PPR from seed entities.
Uses nx.pagerank when scipy is available, falls back to a pure-Python
power iteration implementation otherwise.

See spec §26.6.3 for full documentation.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.15
DEFAULT_MAX_ITER = 100
DEFAULT_TOL = 1e-6


def ppr_score(
    graph: nx.Graph,
    seed_entities: list[str],
    alpha: float = DEFAULT_ALPHA,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict[str, float]:
    """Personalized PageRank from seed entities.

    Tries nx.pagerank (scipy-backed) first for performance.
    Falls back to pure-Python power iteration if scipy is absent.

    Args:
        graph: NetworkX graph (Document or Corpus Graph).
        seed_entities: Entity node IDs to use as teleport targets.
        alpha: Teleport probability (damping = 1 - alpha).
        max_iter: Maximum iterations for convergence.

    Returns:
        Dict mapping node_id -> PPR score (0.0-1.0 range, normalized).
    """
    if graph.number_of_nodes() == 0 or not seed_entities:
        return {}

    # Build personalization vector: uniform over valid seed entities
    valid_seeds = [s for s in seed_entities if s in graph.nodes()]
    if not valid_seeds:
        logger.debug("No valid seed entities found in graph")
        return {}

    personalization = {
        node: (1.0 / len(valid_seeds) if node in valid_seeds else 0.0)
        for node in graph.nodes()
    }

    scores = _compute_ppr(graph, personalization, alpha, max_iter)

    # Normalize to 0-1 range
    if scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

    return scores


def _compute_ppr(
    graph: nx.Graph,
    personalization: dict,
    alpha: float,
    max_iter: int,
) -> dict:
    """Dispatch to scipy-backed or pure-Python PPR."""
    try:
        return nx.pagerank(
            graph,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iter,
        )
    except (ModuleNotFoundError, ImportError):
        logger.debug("scipy not available, using pure-Python PPR fallback")
        return _pagerank_python(graph, personalization, alpha, max_iter)
    except nx.PowerIterationFailedConvergence:
        logger.warning("PPR failed to converge in %d iterations", max_iter)
        try:
            return nx.pagerank(
                graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iter * 2,
                tol=1e-4,
            )
        except (ModuleNotFoundError, ImportError):
            return _pagerank_python(
                graph, personalization, alpha, max_iter * 2,
            )


def _pagerank_python(
    graph: nx.Graph,
    personalization: dict,
    alpha: float = DEFAULT_ALPHA,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> dict:
    """Pure-Python Personalized PageRank via power iteration.

    Implements the standard PPR algorithm without scipy dependency:
      PR(v) = alpha * personalization(v)
            + (1 - alpha) * sum(PR(u) / out_degree(u) for u in in_neighbors(v))

    For undirected graphs, each edge counts in both directions.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    # Normalize personalization vector
    p_sum = sum(personalization.get(v, 0.0) for v in nodes)
    if p_sum == 0:
        p = {v: 1.0 / n for v in nodes}
    else:
        p = {v: personalization.get(v, 0.0) / p_sum for v in nodes}

    # Build incoming-neighbor map with transition weights
    # For each node v, collect (u, 1/degree(u)) for each neighbor u
    incoming: dict[Any, list[tuple[Any, float]]] = {v: [] for v in nodes}
    for v in nodes:
        deg = graph.degree(v)
        if deg > 0:
            w = 1.0 / deg
            for u in graph.neighbors(v):
                incoming[u].append((v, w))

    # Initialize scores uniformly
    scores = {v: 1.0 / n for v in nodes}

    # Identify dangling nodes (no outgoing edges / degree 0)
    dangling = [v for v in nodes if graph.degree(v) == 0]

    damping = 1.0 - alpha

    for _iteration in range(max_iter):
        new_scores: dict[Any, float] = {}

        # Dangling node contribution (redistributed uniformly)
        dangling_sum = sum(scores[v] for v in dangling)

        for v in nodes:
            # Teleport component
            rank = alpha * p[v]

            # Dangling redistribution
            rank += damping * dangling_sum / n

            # Incoming neighbor contributions
            for u, w in incoming[v]:
                rank += damping * scores[u] * w

            new_scores[v] = rank

        # Check convergence (L1 norm)
        delta = sum(abs(new_scores[v] - scores[v]) for v in nodes)
        scores = new_scores

        if delta < tol:
            break

    return scores


def combined_score(
    vector_score: float,
    ppr: float,
    composite_weight: float = 0.3,
) -> float:
    """Compute blended score: alpha x composite + (1-alpha) x PPR.

    Args:
        vector_score: Composite/vector similarity score.
        ppr: PPR structural relevance score.
        composite_weight: Weight alpha for composite score.

    Returns:
        Blended score.
    """
    return composite_weight * vector_score + (1 - composite_weight) * ppr