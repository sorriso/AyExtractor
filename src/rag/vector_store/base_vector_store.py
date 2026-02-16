# src/rag/vector_store/base_vector_store.py — v1
"""Abstract vector store interface.

See spec §30.6 for full documentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ayextractor.rag.models import SearchResult


class BaseVectorStore(ABC):
    """Unified interface for vector store backends."""

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Insert or update vectors with associated documents and metadata."""

    @abstractmethod
    async def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Query vectors by similarity."""

    @abstractmethod
    async def delete(self, collection: str, ids: list[str]) -> None:
        """Delete vectors by ID."""

    @abstractmethod
    async def create_collection(self, collection: str, dimensions: int) -> None:
        """Create a named collection with specified vector dimensions."""

    @abstractmethod
    async def collection_exists(self, collection: str) -> bool:
        """Check if a collection exists."""

    @abstractmethod
    async def count(self, collection: str) -> int:
        """Return number of vectors in a collection."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (chromadb, qdrant, arangodb)."""
