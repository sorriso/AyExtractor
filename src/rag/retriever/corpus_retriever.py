# src/rag/retriever/corpus_retriever.py — v1
"""Corpus retriever — cross-document knowledge from Corpus Graph.

Queries C-nodes, T-nodes, and X-edges from the graph store when
CONSOLIDATOR_ENABLED=true. Provides cross-document context for
enriching agent prompts.

See spec §26.6.1 for architecture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ayextractor.consolidator.models import CNode, TNode, XEdge
from ayextractor.rag.models import CorpusContext, SearchResult

if TYPE_CHECKING:
    from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore

logger = logging.getLogger(__name__)


async def retrieve_corpus_context(
    query: str,
    graph_store: BaseGraphStore,
    seed_entities: list[str] | None = None,
    max_cnodes: int = 20,
    max_xedges: int = 50,
) -> CorpusContext:
    """Retrieve cross-document knowledge from Corpus Graph.

    Args:
        query: Search query text (used for keyword matching).
        graph_store: Graph store containing the Corpus Graph.
        seed_entities: Entity names to seed graph traversal.
        max_cnodes: Maximum C-nodes to return.
        max_xedges: Maximum X-edges to return.

    Returns:
        CorpusContext with C-nodes, T-nodes, and X-edges.
    """
    seeds = seed_entities or []

    # Retrieve C-nodes by name matching or traversal
    cnodes = await _fetch_cnodes(graph_store, seeds, query, max_cnodes)

    # Retrieve X-edges connected to found C-nodes
    cnode_names = {c.canonical_name for c in cnodes}
    xedges = await _fetch_xedges(graph_store, cnode_names, max_xedges)

    # Retrieve T-nodes that classify found C-nodes
    tnodes = await _fetch_tnodes(graph_store, cnode_names)

    # Count distinct source documents
    doc_ids: set[str] = set()
    for c in cnodes:
        for sp in c.source_documents:
            if hasattr(sp, "document_id"):
                doc_ids.add(sp.document_id)

    logger.info(
        "Corpus retrieval: %d C-nodes, %d T-nodes, %d X-edges from %d documents",
        len(cnodes), len(tnodes), len(xedges), len(doc_ids),
    )

    return CorpusContext(
        cnodes=cnodes,
        tnodes=tnodes,
        xedges=xedges,
        source_document_count=len(doc_ids),
    )


def corpus_context_to_search_results(
    ctx: CorpusContext,
) -> list[SearchResult]:
    """Convert CorpusContext to SearchResult list for unified ranking."""
    results: list[SearchResult] = []
    for c in ctx.cnodes:
        content = f"{c.canonical_name} ({c.entity_type})"
        if c.consolidated_attributes:
            content += f" — {c.consolidated_attributes}"
        results.append(
            SearchResult(
                source_type="entity_profile",
                source_id=c.canonical_name,
                content=content,
                score=c.confidence,
                metadata={"source": "corpus_graph", "corroboration": c.corroboration},
            )
        )
    return results


# ------------------------------------------------------------------
# Internal fetch helpers
# ------------------------------------------------------------------


async def _fetch_cnodes(
    store: BaseGraphStore,
    seeds: list[str],
    query: str,
    max_count: int,
) -> list[CNode]:
    """Fetch C-nodes by seed names or keyword search."""
    cnodes: list[CNode] = []

    # Direct lookup by seed entity names
    if seeds and hasattr(store, "get_nodes_by_names"):
        raw = store.get_nodes_by_names(seeds)
        cnodes.extend(_parse_cnodes(raw))

    # Keyword search if store supports it
    if hasattr(store, "search_nodes"):
        keywords = query.split()[:5]  # top 5 keywords
        for kw in keywords:
            if len(cnodes) >= max_count:
                break
            raw = store.search_nodes(kw, limit=max_count - len(cnodes))
            cnodes.extend(_parse_cnodes(raw))

    # Deduplicate by canonical_name
    seen: set[str] = set()
    unique: list[CNode] = []
    for c in cnodes:
        if c.canonical_name not in seen:
            seen.add(c.canonical_name)
            unique.append(c)

    return unique[:max_count]


async def _fetch_xedges(
    store: BaseGraphStore,
    cnode_names: set[str],
    max_count: int,
) -> list[XEdge]:
    """Fetch X-edges connected to given C-nodes."""
    if not cnode_names or not hasattr(store, "get_edges_for_nodes"):
        return []

    raw = store.get_edges_for_nodes(list(cnode_names), limit=max_count)
    return _parse_xedges(raw)


async def _fetch_tnodes(
    store: BaseGraphStore,
    cnode_names: set[str],
) -> list[TNode]:
    """Fetch T-nodes that classify given C-nodes."""
    if not hasattr(store, "get_tnodes"):
        return []

    all_tnodes = store.get_tnodes()
    relevant: list[TNode] = []
    for t in all_tnodes:
        tnode = t if isinstance(t, TNode) else TNode(**t) if isinstance(t, dict) else None
        if tnode is None:
            continue
        if set(tnode.classified_cnodes) & cnode_names:
            relevant.append(tnode)

    return relevant


def _parse_cnodes(raw: Any) -> list[CNode]:
    """Parse raw store results into CNode objects."""
    if not raw:
        return []
    results = []
    for item in raw:
        try:
            if isinstance(item, CNode):
                results.append(item)
            elif isinstance(item, dict):
                results.append(CNode(**item))
        except Exception:
            pass
    return results


def _parse_xedges(raw: Any) -> list[XEdge]:
    """Parse raw store results into XEdge objects."""
    if not raw:
        return []
    results = []
    for item in raw:
        try:
            if isinstance(item, XEdge):
                results.append(item)
            elif isinstance(item, dict):
                results.append(XEdge(**item))
        except Exception:
            pass
    return results
