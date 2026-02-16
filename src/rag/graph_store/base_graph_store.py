# src/rag/graph_store/base_graph_store.py — v1
"""Abstract graph store interface.

See spec §30.7 for full documentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal


class BaseGraphStore(ABC):
    """Unified interface for graph store backends."""

    # --- Node CRUD ---

    @abstractmethod
    async def upsert_node(self, node_id: str, properties: dict) -> None:
        """Insert or update a node (merge by node_id)."""

    @abstractmethod
    async def get_node(self, node_id: str) -> dict | None:
        """Retrieve a node by ID. Returns properties dict or None."""

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges."""

    # --- Edge CRUD ---

    @abstractmethod
    async def upsert_edge(
        self, source_id: str, relation_type: str, target_id: str, properties: dict
    ) -> None:
        """Insert or update an edge (merge by source+relation+target)."""

    @abstractmethod
    async def get_edges(
        self,
        node_id: str,
        direction: Literal["in", "out", "both"] = "both",
        relation_type: str | None = None,
    ) -> list[dict]:
        """Get edges connected to a node."""

    @abstractmethod
    async def delete_edges(
        self,
        source_id: str,
        relation_type: str | None = None,
        target_id: str | None = None,
    ) -> int:
        """Delete edges matching criteria. Returns count deleted."""

    # --- Query ---

    @abstractmethod
    async def query_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> dict:
        """Traverse graph from node_id up to depth hops."""

    @abstractmethod
    async def query_by_properties(
        self,
        label: str | None = None,
        filters: dict | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Find nodes matching property filters."""

    # --- Bulk operations ---

    @abstractmethod
    async def import_graph(self, nodes: list[dict], edges: list[dict]) -> None:
        """Bulk import nodes and edges."""

    @abstractmethod
    async def node_count(self) -> int:
        """Return total number of nodes."""

    @abstractmethod
    async def edge_count(self) -> int:
        """Return total number of edges."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (neo4j, arangodb)."""
