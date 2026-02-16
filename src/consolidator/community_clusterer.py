# src/consolidator/community_clusterer.py — v1
"""Community clusterer — Pass 2 of Corpus Graph consolidation.

Applies Leiden community detection on Corpus Graph C-nodes to
propose T-nodes (taxonomy categories). Clusters of size >= min_cluster_size
become candidate taxonomy entries.

See spec §13.15.4 for full documentation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ayextractor.consolidator.models import ClusteringReport, TNode

if TYPE_CHECKING:
    from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore

logger = logging.getLogger(__name__)


def cluster_corpus(
    corpus_store: BaseGraphStore,
    min_cluster_size: int = 3,
    seed: int | None = 42,
    resolution: float = 1.0,
) -> ClusteringReport:
    """Apply Leiden clustering on Corpus Graph C-nodes → propose T-nodes.

    Args:
        corpus_store: Graph store with Corpus Graph data.
        min_cluster_size: Minimum cluster size to form a T-node.
        seed: Random seed for reproducibility (None for non-deterministic).
        resolution: Leiden resolution parameter.

    Returns:
        ClusteringReport with new/updated T-node counts.
    """
    # Extract C-nodes and edges from corpus store
    graph = _build_cnode_graph(corpus_store)

    if graph.number_of_nodes() < min_cluster_size:
        logger.info(
            "Corpus has only %d C-nodes, below min_cluster_size=%d — skipping",
            graph.number_of_nodes(),
            min_cluster_size,
        )
        return ClusteringReport(new_tnodes=0, updated_tnodes=0, clusters_found=0)

    # Run Leiden community detection
    communities = _run_leiden(graph, resolution=resolution, seed=seed)

    if not communities:
        logger.info("No communities detected")
        return ClusteringReport(new_tnodes=0, updated_tnodes=0, clusters_found=0)

    # Filter by minimum size
    valid_clusters = {
        cid: members
        for cid, members in communities.items()
        if len(members) >= min_cluster_size
    }
    logger.info(
        "Leiden found %d clusters, %d meet min_size=%d",
        len(communities),
        len(valid_clusters),
        min_cluster_size,
    )

    # Reconcile with existing T-nodes
    existing_tnodes = _load_existing_tnodes(corpus_store)
    new_count, updated_count = _reconcile_tnodes(
        valid_clusters, existing_tnodes, corpus_store
    )

    return ClusteringReport(
        new_tnodes=new_count,
        updated_tnodes=updated_count,
        clusters_found=len(valid_clusters),
    )


def _build_cnode_graph(corpus_store: BaseGraphStore) -> Any:
    """Build a NetworkX graph from C-nodes and X-edges in the corpus store."""
    import networkx as nx

    graph = nx.Graph()

    if hasattr(corpus_store, "to_networkx"):
        full = corpus_store.to_networkx()
        # Filter to C-nodes only (entity-level)
        cnode_ids = [
            n for n, d in full.nodes(data=True) if d.get("node_type") == "cnode"
        ]
        if cnode_ids:
            return full.subgraph(cnode_ids).copy()
        # Fallback: use all nodes if no type annotation
        return full

    logger.warning("Corpus store has no to_networkx(); returning empty graph")
    return graph


def _run_leiden(
    graph: Any,
    resolution: float = 1.0,
    seed: int | None = 42,
) -> dict[int, list[str]]:
    """Run Leiden community detection, return {community_id: [node_ids]}.

    Falls back to Louvain if leidenalg is not available.
    """
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {}

    # Attempt leidenalg (preferred)
    try:
        import leidenalg
        import igraph as ig

        ig_graph = ig.Graph.from_networkx(graph)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=seed if seed is not None else 0,
        )
        node_names = list(graph.nodes())
        communities: dict[int, list[str]] = {}
        for cid, members in enumerate(partition):
            communities[cid] = [node_names[i] for i in members]
        return communities
    except ImportError:
        logger.debug("leidenalg not available, falling back to Louvain")

    # Fallback: NetworkX Louvain
    try:
        louvain = nx.community.louvain_communities(
            graph, resolution=resolution, seed=seed
        )
        communities = {}
        for cid, members in enumerate(louvain):
            communities[cid] = list(members)
        return communities
    except Exception:
        logger.warning("Louvain failed, falling back to connected components")
        components = list(nx.connected_components(graph))
        return {i: list(c) for i, c in enumerate(components)}


def _load_existing_tnodes(corpus_store: BaseGraphStore) -> dict[str, TNode]:
    """Load existing T-nodes from corpus store, keyed by canonical_name."""
    tnodes: dict[str, TNode] = {}
    if hasattr(corpus_store, "get_tnodes"):
        for t in corpus_store.get_tnodes():
            if isinstance(t, TNode):
                tnodes[t.canonical_name] = t
            elif isinstance(t, dict):
                try:
                    tnode = TNode(**t)
                    tnodes[tnode.canonical_name] = tnode
                except Exception:
                    pass
    return tnodes


def _reconcile_tnodes(
    clusters: dict[int, list[str]],
    existing: dict[str, TNode],
    corpus_store: BaseGraphStore,
) -> tuple[int, int]:
    """Reconcile detected clusters with existing T-nodes.

    Returns:
        (new_count, updated_count)
    """
    new_count = 0
    updated_count = 0
    now = datetime.now(timezone.utc)

    # Build reverse index: cnode → existing tnode
    cnode_to_tnode: dict[str, str] = {}
    for tname, tnode in existing.items():
        for cn in tnode.classified_cnodes:
            cnode_to_tnode[cn] = tname

    for _cid, members in clusters.items():
        # Check if this cluster maps to an existing T-node
        existing_match = _find_matching_tnode(members, cnode_to_tnode, existing)

        if existing_match is not None:
            # Update existing T-node with new members
            tnode = existing[existing_match]
            old_set = set(tnode.classified_cnodes)
            new_set = old_set | set(members)
            if new_set != old_set:
                tnode.classified_cnodes = sorted(new_set)
                _persist_tnode(corpus_store, tnode)
                updated_count += 1
        else:
            # Create new T-node
            name = _generate_cluster_name(members)
            tnode = TNode(
                canonical_name=name,
                level="concept",
                parent=None,
                children=[],
                classified_cnodes=sorted(members),
                created_by="consolidator_clustering",
                created_at=now,
            )
            _persist_tnode(corpus_store, tnode)
            new_count += 1

    return new_count, updated_count


def _find_matching_tnode(
    members: list[str],
    cnode_to_tnode: dict[str, str],
    existing: dict[str, TNode],
) -> str | None:
    """Find existing T-node that best matches a cluster's members."""
    tnode_votes: dict[str, int] = {}
    for m in members:
        if m in cnode_to_tnode:
            tn = cnode_to_tnode[m]
            tnode_votes[tn] = tnode_votes.get(tn, 0) + 1

    if not tnode_votes:
        return None

    # Best match = most overlapping members
    best = max(tnode_votes, key=tnode_votes.get)  # type: ignore[arg-type]
    overlap = tnode_votes[best]
    if overlap >= len(members) // 2:
        return best
    return None


def _generate_cluster_name(members: list[str]) -> str:
    """Generate a default name for a new T-node from its members."""
    if len(members) <= 3:
        return "cluster_" + "_".join(sorted(members)[:3])
    return f"cluster_{len(members)}_{sorted(members)[0]}"


def _persist_tnode(corpus_store: BaseGraphStore, tnode: TNode) -> None:
    """Persist a T-node to the corpus store."""
    if hasattr(corpus_store, "upsert_tnode"):
        corpus_store.upsert_tnode(tnode.model_dump())
    elif hasattr(corpus_store, "upsert_node"):
        corpus_store.upsert_node(
            tnode.canonical_name,
            {"node_type": "tnode", **tnode.model_dump()},
        )
    else:
        logger.warning("Corpus store has no upsert method; T-node not persisted")
