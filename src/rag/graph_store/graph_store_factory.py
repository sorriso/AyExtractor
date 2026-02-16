# src/rag/graph_store/graph_store_factory.py — v1
"""Factory: instantiate graph store from configuration.

See spec §30.7.
"""

from __future__ import annotations

import logging

from ayextractor.config.settings import Settings
from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore

logger = logging.getLogger(__name__)


class UnsupportedGraphStoreError(ValueError):
    """Raised when a graph store type is not supported."""


def create_graph_store(settings: Settings) -> BaseGraphStore:
    """Instantiate the configured graph store.

    Args:
        settings: Application settings (GRAPH_DB_TYPE).

    Returns:
        Configured BaseGraphStore instance.

    Raises:
        UnsupportedGraphStoreError: If type is not supported.
        ValueError: If type is 'none'.
    """
    db_type = settings.graph_db_type

    if db_type == "none":
        raise ValueError(
            "GRAPH_DB_TYPE is 'none'. Enable a graph DB to use this factory."
        )

    if db_type == "neo4j":
        from ayextractor.rag.graph_store.neo4j_store import Neo4jStore
        return Neo4jStore(
            uri=settings.graph_db_uri,
            user=settings.graph_db_user,
            password=settings.graph_db_password,
            database=settings.graph_db_database,
        )

    if db_type == "arangodb":
        from ayextractor.rag.graph_store.arangodb_store import ArangoDBStore
        return ArangoDBStore(
            url=settings.graph_db_uri,
            database=settings.graph_db_database,
            user=settings.graph_db_user,
            password=settings.graph_db_password,
        )

    raise UnsupportedGraphStoreError(
        f"Unsupported graph store type: {db_type!r}. "
        f"Available: neo4j, arangodb"
    )
