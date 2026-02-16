# src/rag/graph_store/arangodb_store.py — v1
"""ArangoDB graph store adapter.

Uses the python-arango SDK for graph storage.
Requires: pip install python-arango.
See spec §30.7.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore

logger = logging.getLogger(__name__)


class ArangoDBStore(BaseGraphStore):
    """Graph store backed by ArangoDB."""

    def __init__(
        self,
        url: str = "http://localhost:8529",
        database: str = "ayextractor",
        user: str = "root",
        password: str = "",
        graph_name: str = "knowledge_graph",
        node_collection: str = "nodes",
        edge_collection: str = "edges",
    ) -> None:
        try:
            from arango import ArangoClient
        except ImportError as e:
            raise ImportError(
                "python-arango package required: pip install python-arango"
            ) from e

        client = ArangoClient(hosts=url)
        self._db = client.db(database, username=user, password=password)
        self._graph_name = graph_name
        self._node_col = node_collection
        self._edge_col = edge_collection
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        """Create collections and graph definition if they don't exist."""
        if not self._db.has_collection(self._node_col):
            self._db.create_collection(self._node_col)
        if not self._db.has_collection(self._edge_col):
            self._db.create_collection(self._edge_col, edge=True)
        if not self._db.has_graph(self._graph_name):
            self._db.create_graph(
                self._graph_name,
                edge_definitions=[{
                    "edge_collection": self._edge_col,
                    "from_vertex_collections": [self._node_col],
                    "to_vertex_collections": [self._node_col],
                }],
            )

    def _node_key(self, node_id: str) -> str:
        """Build document _key from node_id."""
        return node_id.replace("/", "_").replace(" ", "_")

    def _node_doc_id(self, node_id: str) -> str:
        """Build full document _id."""
        return f"{self._node_col}/{self._node_key(node_id)}"

    # --- Node CRUD ---

    async def upsert_node(self, node_id: str, properties: dict) -> None:
        """Insert or update a node."""
        col = self._db.collection(self._node_col)
        doc = dict(properties)
        doc["_key"] = self._node_key(node_id)
        doc["node_id"] = node_id
        if col.has(doc["_key"]):
            col.update(doc)
        else:
            col.insert(doc)

    async def get_node(self, node_id: str) -> dict | None:
        """Retrieve a node by ID."""
        col = self._db.collection(self._node_col)
        key = self._node_key(node_id)
        if col.has(key):
            doc = col.get(key)
            return {k: v for k, v in doc.items() if not k.startswith("_")}
        return None

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges."""
        key = self._node_key(node_id)
        doc_id = self._node_doc_id(node_id)
        # Delete connected edges first
        edge_col = self._db.collection(self._edge_col)
        aql = "FOR e IN @@col FILTER e._from == @did OR e._to == @did RETURN e._key"
        cursor = self._db.aql.execute(aql, bind_vars={"@col": self._edge_col, "did": doc_id})
        for edge_key in cursor:
            edge_col.delete(edge_key)
        # Delete node
        node_col = self._db.collection(self._node_col)
        if node_col.has(key):
            node_col.delete(key)

    # --- Edge CRUD ---

    async def upsert_edge(
        self, source_id: str, relation_type: str, target_id: str, properties: dict
    ) -> None:
        """Insert or update an edge."""
        edge_col = self._db.collection(self._edge_col)
        edge_key = f"{self._node_key(source_id)}__{relation_type}__{self._node_key(target_id)}"
        doc = dict(properties)
        doc["_key"] = edge_key
        doc["_from"] = self._node_doc_id(source_id)
        doc["_to"] = self._node_doc_id(target_id)
        doc["relation_type"] = relation_type
        if edge_col.has(edge_key):
            edge_col.update(doc)
        else:
            edge_col.insert(doc)

    async def get_edges(
        self,
        node_id: str,
        direction: Literal["in", "out", "both"] = "both",
        relation_type: str | None = None,
    ) -> list[dict]:
        """Get edges connected to a node."""
        doc_id = self._node_doc_id(node_id)
        filters = []
        if direction == "out":
            filters.append("e._from == @did")
        elif direction == "in":
            filters.append("e._to == @did")
        else:
            filters.append("(e._from == @did OR e._to == @did)")
        if relation_type:
            filters.append("e.relation_type == @rel")

        where = " AND ".join(filters)
        bind: dict[str, Any] = {"@col": self._edge_col, "did": doc_id}
        if relation_type:
            bind["rel"] = relation_type

        aql = f"FOR e IN @@col FILTER {where} RETURN e"
        cursor = self._db.aql.execute(aql, bind_vars=bind)
        return [
            {k: v for k, v in doc.items() if not k.startswith("_")}
            for doc in cursor
        ]

    async def delete_edges(
        self,
        source_id: str,
        relation_type: str | None = None,
        target_id: str | None = None,
    ) -> int:
        """Delete edges matching criteria."""
        doc_from = self._node_doc_id(source_id)
        filters = ["e._from == @src"]
        bind: dict[str, Any] = {"@col": self._edge_col, "src": doc_from}
        if relation_type:
            filters.append("e.relation_type == @rel")
            bind["rel"] = relation_type
        if target_id:
            filters.append("e._to == @tgt")
            bind["tgt"] = self._node_doc_id(target_id)

        where = " AND ".join(filters)
        aql = f"FOR e IN @@col FILTER {where} REMOVE e IN @@col RETURN 1"
        cursor = self._db.aql.execute(aql, bind_vars=bind)
        return sum(1 for _ in cursor)

    # --- Query ---

    async def query_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> dict:
        """Traverse graph from node_id up to depth hops."""
        start = self._node_doc_id(node_id)
        edge_filter = ""
        if relation_types:
            types_str = ", ".join(f'"{r}"' for r in relation_types)
            edge_filter = f"FILTER e.relation_type IN [{types_str}]"

        aql = (
            f"FOR v, e, p IN 1..{depth} ANY @start "
            f"GRAPH @graph {edge_filter} "
            f"RETURN {{node: v, edge: e}}"
        )
        cursor = self._db.aql.execute(
            aql, bind_vars={"start": start, "graph": self._graph_name}
        )
        return {"results": list(cursor)}

    async def query_by_properties(
        self,
        label: str | None = None,
        filters: dict | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Find nodes matching property filters."""
        filter_parts = []
        bind: dict[str, Any] = {"@col": self._node_col, "limit": limit}
        if label:
            filter_parts.append("n.label == @label")
            bind["label"] = label
        if filters:
            for i, (k, v) in enumerate(filters.items()):
                filter_parts.append(f"n.{k} == @v{i}")
                bind[f"v{i}"] = v

        where = f"FILTER {' AND '.join(filter_parts)}" if filter_parts else ""
        aql = f"FOR n IN @@col {where} LIMIT @limit RETURN n"
        cursor = self._db.aql.execute(aql, bind_vars=bind)
        return [
            {k: v for k, v in doc.items() if not k.startswith("_")}
            for doc in cursor
        ]

    # --- Bulk ---

    async def import_graph(self, nodes: list[dict], edges: list[dict]) -> None:
        """Bulk import nodes and edges."""
        for node in nodes:
            nid = node.get("node_id", node.get("id", ""))
            await self.upsert_node(nid, dict(node))
        for edge in edges:
            await self.upsert_edge(
                edge["source"], edge["relation"], edge["target"],
                {k: v for k, v in edge.items() if k not in ("source", "relation", "target")},
            )

    async def node_count(self) -> int:
        """Return total number of nodes."""
        return self._db.collection(self._node_col).count()

    async def edge_count(self) -> int:
        """Return total number of edges."""
        return self._db.collection(self._edge_col).count()

    @property
    def provider_name(self) -> str:
        return "arangodb"
