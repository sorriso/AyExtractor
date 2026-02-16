# src/graph/layers/community_integrator.py — v1
"""Community integrator — inject L1 nodes into the document graph.

ONLY module that modifies the graph after builder.py.
Creates L1 community nodes, 'encompasses' edges (L1→L2),
and 'related_to' edges (L1↔L1) for co-occurring communities.

See spec §13.10.2 (integration section) for full documentation.
"""

from __future__ import annotations

import logging

import networkx as nx

from ayextractor.graph.layers.models import CommunityHierarchy

logger = logging.getLogger(__name__)

# Minimum shared chunks for L1↔L1 'related_to' edge
MIN_SHARED_CHUNKS = 3


def integrate_communities(
    graph: nx.Graph,
    hierarchy: CommunityHierarchy,
) -> nx.Graph:
    """Inject L1 community nodes and edges into the graph.

    1. Create L1 nodes with entity_type='community_topic' for each community.
    2. Create 'encompasses' edges (L1 → L2 member).
    3. Create 'related_to' edges (L1 ↔ L1) for communities sharing ≥3 chunks.
    4. Set community_id attribute on L2 member nodes.

    Args:
        graph: NetworkX graph (L2/L3 nodes from builder).
        hierarchy: Community detection result.

    Returns:
        Modified graph with L1 nodes and edges added.
    """
    if not hierarchy.communities:
        return graph

    # Step 1+2: Create L1 nodes and encompasses edges
    for community in hierarchy.communities:
        cid = community.community_id

        graph.add_node(
            cid,
            canonical_name=cid,
            layer=1,
            entity_type="community_topic",
            is_literal=False,
            aliases=[],
            members=community.members,
            modularity_score=community.modularity_score,
            chunk_coverage=community.chunk_coverage,
            occurrence_count=len(community.members),
            source_chunk_ids=community.chunk_coverage,
            confidence=community.modularity_score,
        )

        for member in community.members:
            if graph.has_node(member):
                graph.add_edge(
                    cid,
                    member,
                    relation_type="encompasses",
                    confidence=1.0,
                    occurrence_count=1,
                )
                # Step 4: set community_id on member
                graph.nodes[member]["community_id"] = cid

    # Step 3: L1↔L1 related_to edges based on shared chunks
    comm_list = hierarchy.communities
    for i in range(len(comm_list)):
        for j in range(i + 1, len(comm_list)):
            shared = set(comm_list[i].chunk_coverage) & set(comm_list[j].chunk_coverage)
            if len(shared) >= MIN_SHARED_CHUNKS:
                graph.add_edge(
                    comm_list[i].community_id,
                    comm_list[j].community_id,
                    relation_type="related_to",
                    confidence=len(shared) / max(
                        len(comm_list[i].chunk_coverage),
                        len(comm_list[j].chunk_coverage),
                        1,
                    ),
                    shared_chunks=sorted(shared),
                    occurrence_count=len(shared),
                )

    logger.info(
        "Integrated %d L1 community nodes into graph (now %d nodes, %d edges)",
        len(hierarchy.communities),
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph
