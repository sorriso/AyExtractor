# src/rag/graph_store/neo4j_store.py — v1
"""Neo4j graph store adapter.

Uses the neo4j Python driver for graph storage.
Requires: pip install neo4j.
See spec §30.7.
"""

from __future__ import annotations

import logging
from typing import Literal

from ayextractor.rag.graph_store.base_graph_store import BaseGraphStore

logger = logging.getLogger(__name__)


class Neo4jStore(BaseGraphStore):
    """Graph store backed by Neo4j."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "",
        password: str = "",
        database: str = "neo4j",
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError(
                "neo4j package required: pip install neo4j"
            ) from e

        auth = (user, password) if user else None
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._database = database

    def _run(self, query: str, **params) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        with self._driver.session(database=self._database) as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    # --- Node CRUD ---

    async def upsert_node(self, node_id: str, properties: dict) -> None:
        """Insert or update a node (MERGE by node_id)."""
        label = properties.pop("label", "Entity")
        props_str = ", ".join(f"n.{k} = ${k}" for k in properties)
        query = f"MERGE (n:{label} {{node_id: $node_id}}) SET {props_str}" if props_str else \
                f"MERGE (n:{label} {{node_id: $node_id}})"
        self._run(query, node_id=node_id, **properties)

    async def get_node(self, node_id: str) -> dict | None:
        """Retrieve a node by ID."""
        results = self._run(
            "MATCH (n {node_id: $node_id}) RETURN properties(n) AS props",
            node_id=node_id,
        )
        return results[0]["props"] if results else None

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges."""
        self._run("MATCH (n {node_id: $node_id}) DETACH DELETE n", node_id=node_id)

    # --- Edge CRUD ---

    async def upsert_edge(
        self, source_id: str, relation_type: str, target_id: str, properties: dict
    ) -> None:
        """Insert or update an edge."""
        safe_rel = relation_type.replace(" ", "_").replace("-", "_").upper()
        props_str = ", ".join(f"r.{k} = ${k}" for k in properties)
        set_clause = f"SET {props_str}" if props_str else ""
        query = (
            f"MATCH (a {{node_id: $src}}), (b {{node_id: $tgt}}) "
            f"MERGE (a)-[r:{safe_rel}]->(b) {set_clause}"
        )
        self._run(query, src=source_id, tgt=target_id, **properties)

    async def get_edges(
        self,
        node_id: str,
        direction: Literal["in", "out", "both"] = "both",
        relation_type: str | None = None,
    ) -> list[dict]:
        """Get edges connected to a node."""
        rel_filter = f":{relation_type}" if relation_type else ""
        if direction == "out":
            pattern = f"(n {{node_id: $nid}})-[r{rel_filter}]->(m)"
        elif direction == "in":
            pattern = f"(m)-[r{rel_filter}]->(n {{node_id: $nid}})"
        else:
            pattern = f"(n {{node_id: $nid}})-[r{rel_filter}]-(m)"

        results = self._run(
            f"MATCH {pattern} RETURN type(r) AS rel, properties(r) AS props, "
            f"m.node_id AS target_id",
            nid=node_id,
        )
        return results

    async def delete_edges(
        self,
        source_id: str,
        relation_type: str | None = None,
        target_id: str | None = None,
    ) -> int:
        """Delete edges matching criteria."""
        rel_filter = f":{relation_type}" if relation_type else ""
        if target_id:
            query = (
                f"MATCH (a {{node_id: $src}})-[r{rel_filter}]->(b {{node_id: $tgt}}) "
                f"DELETE r RETURN count(r) AS cnt"
            )
            results = self._run(query, src=source_id, tgt=target_id)
        else:
            query = (
                f"MATCH (a {{node_id: $src}})-[r{rel_filter}]->() "
                f"DELETE r RETURN count(r) AS cnt"
            )
            results = self._run(query, src=source_id)
        return results[0]["cnt"] if results else 0

    # --- Query ---

    async def query_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        relation_types: list[str] | None = None,
    ) -> dict:
        """Traverse graph from node_id up to depth hops."""
        rel_filter = "|".join(relation_types) if relation_types else ""
        rel_clause = f":{rel_filter}" if rel_filter else ""
        query = (
            f"MATCH path = (n {{node_id: $nid}})-[{rel_clause}*1..{depth}]-(m) "
            f"RETURN [node IN nodes(path) | properties(node)] AS nodes, "
            f"[rel IN relationships(path) | {{type: type(rel), props: properties(rel)}}] AS rels"
        )
        results = self._run(query, nid=node_id)
        return {"paths": results}

    async def query_by_properties(
        self,
        label: str | None = None,
        filters: dict | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Find nodes matching property filters."""
        label_clause = f":{label}" if label else ""
        where_parts = []
        params: dict = {"limit": limit}
        if filters:
            for k, v in filters.items():
                where_parts.append(f"n.{k} = ${k}")
                params[k] = v
        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        query = f"MATCH (n{label_clause}) {where_clause} RETURN properties(n) AS props LIMIT $limit"
        return self._run(query, **params)

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
        results = self._run("MATCH (n) RETURN count(n) AS cnt")
        return results[0]["cnt"] if results else 0

    async def edge_count(self) -> int:
        """Return total number of edges."""
        results = self._run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return results[0]["cnt"] if results else 0

    @property
    def provider_name(self) -> str:
        return "neo4j"

    def close(self) -> None:
        """Close the driver connection."""
        self._driver.close()
