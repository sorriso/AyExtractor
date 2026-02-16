# src/rag/retriever/entity_retriever.py — v1
"""Entity retriever — Level 2 of hierarchical retrieval pipeline.

Vector search on entity_profiles + relation_profiles collections,
combined with PPR scoring on the knowledge graph for structural relevance.

See spec §26.6.1 for architecture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ayextractor.rag.models import SearchResult

if TYPE_CHECKING:
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder
    from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 20
DEFAULT_MIN_SCORE = 0.25
DEFAULT_COMPOSITE_WEIGHT = 0.3


async def retrieve_entities(
    query: str,
    vector_store: BaseVectorStore,
    embedder: BaseEmbedder,
    knowledge_graph: Any | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    composite_weight: float = DEFAULT_COMPOSITE_WEIGHT,
) -> list[SearchResult]:
    """Retrieve entity and relation profiles by vector + PPR scoring.

    Final score = α × composite_score + (1-α) × ppr_score
    where α = composite_weight (default 0.3).

    Args:
        query: Search query text.
        vector_store: Vector store with entity/relation profiles.
        embedder: Embedder to vectorize the query.
        knowledge_graph: Optional nx.Graph for PPR scoring.
        top_k: Maximum results to return.
        min_score: Minimum score threshold for pruning.
        composite_weight: Weight α for composite vs PPR score.

    Returns:
        Ranked list of SearchResult for entity/relation profiles.
    """
    query_embedding = await embedder.embed(query)

    # Search entity profiles
    entity_results = await vector_store.search(
        collection="entity_profiles",
        query_vector=query_embedding,
        top_k=top_k,
    )

    # Search relation profiles
    relation_results = await vector_store.search(
        collection="relation_profiles",
        query_vector=query_embedding,
        top_k=top_k // 2,
    )

    # Combine and convert
    combined = _convert_results(entity_results, "entity_profile")
    combined.extend(_convert_results(relation_results, "relation_profile"))

    # Apply PPR scoring if graph available
    if knowledge_graph is not None and knowledge_graph.number_of_nodes() > 0:
        combined = _apply_ppr_scoring(
            combined, knowledge_graph, composite_weight
        )

    # Prune by min_score and sort
    combined = [r for r in combined if r.score >= min_score]
    combined.sort(key=lambda r: r.score, reverse=True)

    logger.info(
        "Entity retrieval: %d results (top_k=%d)",
        len(combined[:top_k]), top_k,
    )
    return combined[:top_k]


def _convert_results(
    raw_results: list,
    source_type: str,
) -> list[SearchResult]:
    """Convert raw store results to SearchResult objects."""
    results = []
    for item in raw_results:
        if isinstance(item, dict):
            score = item.get("score", 0.0)
            content = item.get("content", "")
            source_id = item.get("id", "")
            metadata = item.get("metadata", {})
        else:
            score = getattr(item, "score", 0.0)
            content = getattr(item, "content", "")
            source_id = getattr(item, "id", "")
            metadata = getattr(item, "metadata", {})

        results.append(
            SearchResult(
                source_type=source_type,
                source_id=str(source_id),
                content=str(content),
                score=float(score),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return results


def _apply_ppr_scoring(
    results: list[SearchResult],
    graph: Any,
    composite_weight: float,
) -> list[SearchResult]:
    """Blend vector similarity scores with PPR graph scores."""
    from ayextractor.rag.retriever.ppr_scorer import ppr_score

    # Extract seed entities from top results
    seed_entities = [r.source_id for r in results[:5] if r.score > 0.3]
    if not seed_entities:
        return results

    # Filter seeds to existing graph nodes
    valid_seeds = [s for s in seed_entities if s in graph.nodes()]
    if not valid_seeds:
        return results

    ppr_scores = ppr_score(graph, valid_seeds)

    # Blend scores
    blended = []
    for r in results:
        ppr = ppr_scores.get(r.source_id, 0.0)
        combined = composite_weight * r.score + (1 - composite_weight) * ppr
        blended.append(
            SearchResult(
                source_type=r.source_type,
                source_id=r.source_id,
                content=r.content,
                score=combined,
                metadata={**r.metadata, "vector_score": r.score, "ppr_score": ppr},
            )
        )
    return blended
