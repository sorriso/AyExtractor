# src/graph/layers/community_detector.py — v1
"""Hierarchical community detection via Leiden algorithm.

Pure function: takes a NetworkX L2 subgraph, returns CommunityHierarchy.
Does NOT modify the input graph. L1 injection is done by community_integrator.

Falls back to connected components if leidenalg/igraph not installed.
See spec §13.10.2 for full documentation.
"""

from __future__ import annotations

import logging

import networkx as nx

from ayextractor.graph.layers.models import Community, CommunityHierarchy

logger = logging.getLogger(__name__)


def detect_communities(
    graph: nx.Graph,
    resolution: float = 1.0,
    min_community_size: int = 3,
    seed: int | None = 42,
) -> CommunityHierarchy:
    """Apply hierarchical Leiden on L2 subgraph.

    Args:
        graph: NetworkX graph containing ONLY L2 nodes.
        resolution: Leiden resolution parameter (higher = more communities).
        min_community_size: Minimum members per community.
        seed: Random seed for reproducibility (None = non-deterministic).

    Returns:
        CommunityHierarchy with detected communities.
    """
    if graph.number_of_nodes() == 0:
        return CommunityHierarchy(
            num_levels=0, resolution=resolution, seed=seed,
            total_communities=0, modularity=0.0,
        )

    try:
        return _leiden_detect(graph, resolution, min_community_size, seed)
    except ImportError:
        logger.info("leidenalg/igraph not available, using connected components fallback")
        return _fallback_detect(graph, min_community_size, resolution, seed)


def _leiden_detect(
    graph: nx.Graph,
    resolution: float,
    min_community_size: int,
    seed: int | None,
) -> CommunityHierarchy:
    """Leiden-based detection via leidenalg + igraph."""
    import igraph as ig
    import leidenalg

    # Convert NetworkX -> igraph
    node_list = list(graph.nodes)
    node_index = {n: i for i, n in enumerate(node_list)}
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(node_list))
    ig_graph.vs["name"] = node_list

    for u, v in graph.edges:
        ig_graph.add_edge(node_index[u], node_index[v])

    # Run Leiden
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed if seed is not None else 0,
    )

    modularity = partition.modularity

    # Build communities
    communities: list[Community] = []
    comm_id = 0
    for members_idx in partition:
        members = [node_list[i] for i in members_idx]
        if len(members) < min_community_size:
            continue
        chunk_cov = _collect_chunk_coverage(graph, members)
        communities.append(Community(
            community_id=f"comm_{comm_id:03d}",
            level=0,
            members=sorted(members),
            modularity_score=modularity,
            chunk_coverage=sorted(chunk_cov),
        ))
        comm_id += 1

    return CommunityHierarchy(
        communities=communities,
        num_levels=1,
        resolution=resolution,
        seed=seed,
        total_communities=len(communities),
        modularity=modularity,
    )


def _fallback_detect(
    graph: nx.Graph,
    min_community_size: int,
    resolution: float,
    seed: int | None,
) -> CommunityHierarchy:
    """Fallback: use connected components as pseudo-communities."""
    communities: list[Community] = []
    comm_id = 0

    for component in nx.connected_components(graph):
        members = sorted(component)
        if len(members) < min_community_size:
            continue
        chunk_cov = _collect_chunk_coverage(graph, members)
        communities.append(Community(
            community_id=f"comm_{comm_id:03d}",
            level=0,
            members=members,
            modularity_score=0.0,
            chunk_coverage=sorted(chunk_cov),
        ))
        comm_id += 1

    # Compute modularity estimate for connected components
    modularity = 0.0
    if graph.number_of_edges() > 0 and communities:
        modularity = nx.algorithms.community.modularity(
            graph,
            [set(c.members) for c in communities],
        )

    return CommunityHierarchy(
        communities=communities,
        num_levels=1,
        resolution=resolution,
        seed=seed,
        total_communities=len(communities),
        modularity=modularity,
    )


def _collect_chunk_coverage(graph: nx.Graph, members: list[str]) -> set[str]:
    """Collect all chunk_ids covered by community members."""
    chunks: set[str] = set()
    for m in members:
        data = graph.nodes.get(m, {})
        for cid in data.get("source_chunk_ids", []):
            chunks.add(cid)
    return chunks
