# src/rag/vector_store/qdrant_store.py — v3
"""Qdrant vector store adapter.

Uses the qdrant-client SDK for local or cloud vector storage.
Requires: pip install qdrant-client.
See spec §30.6.

Changelog:
    v3: Score safety — handle None score from query_points.
    v2: Fix for qdrant-client>=1.13 — use query_points() instead of removed search().
"""

from __future__ import annotations

import logging
import uuid

from ayextractor.rag.models import SearchResult
from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantStore(BaseVectorStore):
    """Vector store backed by Qdrant."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError(
                "qdrant-client package required: pip install qdrant-client"
            ) from e

        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(":memory:")

    async def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Insert or update vectors."""
        from qdrant_client.models import PointStruct

        points = []
        for i, vec_id in enumerate(ids):
            payload = {"document": documents[i]}
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])
            # Qdrant requires UUID or int IDs; use deterministic UUID from string
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, vec_id))
            points.append(
                PointStruct(id=point_id, vector=embeddings[i], payload=payload)
            )
        self._client.upsert(collection_name=collection, points=points)

    async def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Query by embedding similarity.

        Uses query_points() (qdrant-client>=1.7) instead of the removed search().
        """
        search_kwargs: dict = {
            "collection_name": collection,
            "query": query_embedding,
            "limit": top_k,
        }
        if filter:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter.items()
            ]
            search_kwargs["query_filter"] = Filter(must=conditions)

        response = self._client.query_points(**search_kwargs)
        hits = response.points

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    source_type=payload.get("source_type", "chunk"),
                    source_id=str(hit.id),
                    content=payload.get("document", ""),
                    score=float(hit.score) if hit.score is not None else 0.0,
                    metadata=payload,
                )
            )
        return results

    async def delete(self, collection: str, ids: list[str]) -> None:
        """Delete vectors by ID."""
        from qdrant_client.models import PointIdsList

        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, vid)) for vid in ids]
        self._client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=point_ids),
        )

    async def create_collection(self, collection: str, dimensions: int) -> None:
        """Create a named collection with specified dimensions."""
        from qdrant_client.models import Distance, VectorParams

        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
            )

    async def collection_exists(self, collection: str) -> bool:
        """Check if a collection exists."""
        return self._client.collection_exists(collection)

    async def count(self, collection: str) -> int:
        """Return number of vectors in a collection."""
        info = self._client.get_collection(collection)
        return info.points_count or 0

    @property
    def provider_name(self) -> str:
        return "qdrant"