# src/graph/layers/layer_classifier.py — v1
"""Layer classifier — assigns L2 (ACTORS) / L3 (EVIDENCE) to graph nodes.

Pure function: reads graph, returns mapping, does NOT modify the graph.
L1 (TOPICS) assignment is done separately by community_detector.py.

See spec §13.9.2 for classification logic.
"""

from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger(__name__)


def classify_layers(graph: nx.Graph) -> dict[str, int]:
    """Assign layer to each node based on measurable criteria.

    L2 is default for named entities. L3 for literal values (numbers,
    dates, percentages, measures). L1 is assigned later by community_detector.

    Args:
        graph: NetworkX graph with nodes having 'is_literal' attribute.

    Returns:
        Mapping {node_name: layer} where layer is 2 or 3.
    """
    layers: dict[str, int] = {}
    l2_count = 0
    l3_count = 0

    for node, data in graph.nodes(data=True):
        if data.get("is_literal", False):
            layers[node] = 3
            l3_count += 1
        else:
            layers[node] = 2
            l2_count += 1

    logger.info(
        "Layer classification: %d L2 (ACTORS), %d L3 (EVIDENCE)",
        l2_count,
        l3_count,
    )
    return layers


def apply_layers(graph: nx.Graph, layers: dict[str, int]) -> None:
    """Apply layer assignments to graph node attributes.

    Mutates graph nodes in place, setting the 'layer' attribute.

    Args:
        graph: NetworkX graph to update.
        layers: Mapping from classify_layers().
    """
    for node, layer in layers.items():
        if graph.has_node(node):
            graph.nodes[node]["layer"] = layer


def get_l2_subgraph(graph: nx.Graph) -> nx.Graph:
    """Extract the L2-only subgraph for community detection.

    Returns a new graph containing only L2 (ACTORS) nodes and
    their inter-connections. L3 nodes and their edges are excluded.

    Args:
        graph: Full document graph.

    Returns:
        Subgraph with L2 nodes only.
    """
    l2_nodes = [
        node for node, data in graph.nodes(data=True)
        if data.get("layer", 2) == 2
    ]
    return graph.subgraph(l2_nodes).copy()
