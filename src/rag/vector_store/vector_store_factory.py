# src/rag/vector_store/vector_store_factory.py — v1
"""Factory: instantiate vector store from configuration.

See spec §30.6.
"""

from __future__ import annotations

import logging

from ayextractor.config.settings import Settings
from ayextractor.rag.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class UnsupportedVectorStoreError(ValueError):
    """Raised when a vector store type is not supported."""


def create_vector_store(settings: Settings) -> BaseVectorStore:
    """Instantiate the configured vector store.

    Args:
        settings: Application settings (VECTOR_DB_TYPE).

    Returns:
        Configured BaseVectorStore instance.

    Raises:
        UnsupportedVectorStoreError: If type is not supported.
        ValueError: If type is 'none'.
    """
    db_type = settings.vector_db_type

    if db_type == "none":
        raise ValueError(
            "VECTOR_DB_TYPE is 'none'. Enable a vector DB to use this factory."
        )

    if db_type == "chromadb":
        from ayextractor.rag.vector_store.chromadb_store import ChromaDBStore
        url = settings.vector_db_url
        if url:
            # Remote ChromaDB: parse host:port
            host = url.split("://")[-1].split(":")[0] if "://" in url else url.split(":")[0]
            port = int(url.rsplit(":", 1)[-1]) if ":" in url.rsplit("/", 1)[-1] else 8000
            return ChromaDBStore(host=host, port=port)
        return ChromaDBStore(persist_path=str(settings.vector_db_path))

    if db_type == "qdrant":
        from ayextractor.rag.vector_store.qdrant_store import QdrantStore
        url = settings.vector_db_url
        if url:
            return QdrantStore(url=url, api_key=settings.vector_db_api_key or None)
        return QdrantStore(path=str(settings.vector_db_path))

    raise UnsupportedVectorStoreError(
        f"Unsupported vector store type: {db_type!r}. "
        f"Available: chromadb, qdrant"
    )
