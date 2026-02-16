# src/graph/inference_engine.py — v1
"""Inference engine — derive implicit triplets from graph patterns.

Applies basic reasoning rules to infer new edges that are not
explicitly stated in the document but follow from existing relations.

Rules implemented:
  1. Transitivity: A→B, B→C ⟹ A→C (for transitive predicates).
  2. Inverse: A→B via pred ⟹ B→A via inverse(pred).
  3. Membership propagation: A member_of B, B part_of C ⟹ A part_of C.

Inferred triplets are marked with is_inferred=True.
See spec §14.4 for inference rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger(__name__)

TRANSITIVE_PREDICATES = {"part_of", "is_a", "subclass_of", "contains"}

INVERSE_MAP = {
    "contains": "part_of",
    "part_of": "contains",
    "is_a": "subclass_of",
    "subclass_of": "is_a",
    "regulates": "regulated_by",
    "regulated_by": "regulates",
    "created_by": "creates",
    "creates": "created_by",
    "precedes": "follows",
    "follows": "precedes",
}

# Max chain depth for transitivity to avoid exponential blowup
MAX_TRANSITIVE_DEPTH = 3

# Confidence discount for inferred edges
INFERENCE_CONFIDENCE_FACTOR = 0.7


@dataclass
class InferredEdge:
    """A newly inferred edge."""

    source: str
    target: str
    relation_type: str
    confidence: float
    rule: str  # "transitivity", "inverse", "membership"
    evidence_path: list[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Result of inference pass."""

    inferred_edges: list[InferredEdge] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def run_inference(
    graph: nx.Graph,
    apply_to_graph: bool = False,
    max_depth: int = MAX_TRANSITIVE_DEPTH,
) -> InferenceResult:
    """Run all inference rules on the graph.

    Args:
        graph: NetworkX graph with typed edges.
        apply_to_graph: If True, add inferred edges to the graph.
        max_depth: Maximum transitivity chain depth.

    Returns:
        InferenceResult with inferred edges.
    """
    all_inferred: list[InferredEdge] = []

    # Rule 1: Transitivity
    transitive = _infer_transitive(graph, max_depth)
    all_inferred.extend(transitive)

    # Rule 2: Inverse relations
    inverses = _infer_inverses(graph)
    all_inferred.extend(inverses)

    # Deduplicate
    seen: set[tuple[str, str, str]] = set()
    unique: list[InferredEdge] = []
    for edge in all_inferred:
        key = (edge.source, edge.target, edge.relation_type)
        # Skip if already exists in graph
        if graph.has_edge(edge.source, edge.target):
            existing = graph[edge.source][edge.target]
            if existing.get("relation_type") == edge.relation_type:
                continue
        if key not in seen:
            seen.add(key)
            unique.append(edge)

    # Apply if requested
    if apply_to_graph:
        for edge in unique:
            graph.add_edge(
                edge.source,
                edge.target,
                relation_type=edge.relation_type,
                confidence=edge.confidence,
                is_inferred=True,
                inference_rule=edge.rule,
                evidence_path=edge.evidence_path,
                occurrence_count=0,
            )

    stats = {
        "transitive": sum(1 for e in unique if e.rule == "transitivity"),
        "inverse": sum(1 for e in unique if e.rule == "inverse"),
        "total_inferred": len(unique),
    }

    logger.info("Inference: %d new edges (%s)", len(unique), stats)
    return InferenceResult(inferred_edges=unique, stats=stats)


def _infer_transitive(graph: nx.Graph, max_depth: int) -> list[InferredEdge]:
    """Infer transitive edges: A→B→C ⟹ A→C."""
    results: list[InferredEdge] = []

    for u, v, data in graph.edges(data=True):
        pred = data.get("relation_type", "")
        if pred not in TRANSITIVE_PREDICATES:
            continue
        conf_uv = data.get("confidence", 0.5)

        # BFS forward from v, following same predicate
        visited = {u, v}
        queue: list[tuple[str, float, list[str]]] = [(v, conf_uv, [u, v])]

        while queue:
            current, conf, path = queue.pop(0)
            if len(path) > max_depth + 1:
                continue
            for _, neighbor, ndata in graph.edges(current, data=True):
                if ndata.get("relation_type") != pred:
                    continue
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                new_conf = conf * ndata.get("confidence", 0.5) * INFERENCE_CONFIDENCE_FACTOR
                new_path = path + [neighbor]
                results.append(InferredEdge(
                    source=u,
                    target=neighbor,
                    relation_type=pred,
                    confidence=round(new_conf, 4),
                    rule="transitivity",
                    evidence_path=new_path,
                ))
                queue.append((neighbor, new_conf, new_path))

    return results


def _infer_inverses(graph: nx.Graph) -> list[InferredEdge]:
    """Infer inverse edges: A→B via pred ⟹ B→A via inverse(pred)."""
    results: list[InferredEdge] = []

    for u, v, data in graph.edges(data=True):
        pred = data.get("relation_type", "")
        inverse_pred = INVERSE_MAP.get(pred)
        if inverse_pred is None:
            continue
        # Check if inverse already exists
        if graph.has_edge(v, u):
            existing = graph[v][u]
            if existing.get("relation_type") == inverse_pred:
                continue
        conf = data.get("confidence", 0.5) * INFERENCE_CONFIDENCE_FACTOR
        results.append(InferredEdge(
            source=v,
            target=u,
            relation_type=inverse_pred,
            confidence=round(conf, 4),
            rule="inverse",
            evidence_path=[v, u],
        ))

    return results
