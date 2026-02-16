# src/rag/retriever/community_retriever.py — v1
"""Community retriever — Level 1 of hierarchical retrieval pipeline.

Vector search on community_summaries collection. Returns top-K community
summaries ranked by relevance, pruned by score threshold.

See spec §26.6.1 for architecture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ayextractor.rag.models import SearchResult

if TYPE_CHECKING:
    from ayextractor.rag.embeddings.base_embedder import BaseEmbedder
    from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.3


async def retrieve_communities(
    query: str,
    vector_store: BaseVectorStore,
    embedder: BaseEmbedder,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[SearchResult]:
    """Retrieve top-K community summaries by vector similarity.

    Args:
        query: Search query text.
        vector_store: Vector store with community_summaries collection.
        embedder: Embedder to vectorize the query.
        top_k: Maximum results to return.
        min_score: Minimum similarity score threshold for pruning.

    Returns:
        Ranked list of SearchResult with source_type="community_summary".
    """
    # Embed query
    query_embedding = await embedder.embed(query)

    # Search community_summaries collection
    raw_results = await vector_store.search(
        collection="community_summaries",
        query_vector=query_embedding,
        top_k=top_k,
    )

    # Convert and prune
    results: list[SearchResult] = []
    for item in raw_results:
        score = item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)
        if score < min_score:
            continue

        content = item.get("content", "") if isinstance(item, dict) else getattr(item, "content", "")
        source_id = item.get("id", "") if isinstance(item, dict) else getattr(item, "id", "")
        metadata = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})

        results.append(
            SearchResult(
                source_type="community_summary",
                source_id=str(source_id),
                content=str(content),
                score=float(score),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    logger.info(
        "Community retrieval: %d results (top_k=%d, min_score=%.2f)",
        len(results), top_k, min_score,
    )
    return results[:top_k]
