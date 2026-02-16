# src/rag/vector_store/chromadb_store.py — v1
"""ChromaDB vector store adapter.

Uses the chromadb SDK for local or remote vector storage.
Requires: pip install chromadb.
See spec §30.6.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ayextractor.rag.models import SearchResult
from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaDBStore(BaseVectorStore):
    """Vector store backed by ChromaDB."""

    def __init__(
        self,
        persist_path: str | Path | None = None,
        host: str | None = None,
        port: int = 8000,
    ) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "chromadb package required: pip install chromadb"
            ) from e

        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        elif persist_path:
            self._client = chromadb.PersistentClient(path=str(persist_path))
        else:
            self._client = chromadb.Client()

    async def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Insert or update vectors."""
        col = self._client.get_or_create_collection(collection)
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Query by embedding similarity."""
        col = self._client.get_collection(collection)
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter:
            kwargs["where"] = filter

        results = col.query(**kwargs)

        search_results: list[SearchResult] = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                score = 1.0 - (results["distances"][0][i] if results["distances"] else 0)
                doc = results["documents"][0][i] if results["documents"] else ""
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                search_results.append(
                    SearchResult(
                        source_type=meta.get("source_type", "chunk"),
                        source_id=doc_id,
                        content=doc,
                        score=score,
                        metadata=meta,
                    )
                )
        return search_results

    async def delete(self, collection: str, ids: list[str]) -> None:
        """Delete vectors by ID."""
        col = self._client.get_collection(collection)
        col.delete(ids=ids)

    async def create_collection(self, collection: str, dimensions: int) -> None:
        """Create a named collection."""
        self._client.get_or_create_collection(
            collection, metadata={"dimensions": dimensions}
        )

    async def collection_exists(self, collection: str) -> bool:
        """Check if a collection exists."""
        try:
            self._client.get_collection(collection)
            return True
        except Exception:
            return False

    async def count(self, collection: str) -> int:
        """Return number of vectors in a collection."""
        col = self._client.get_collection(collection)
        return col.count()

    @property
    def provider_name(self) -> str:
        return "chromadb"
