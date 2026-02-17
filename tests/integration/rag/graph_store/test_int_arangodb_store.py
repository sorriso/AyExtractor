# tests/integration/rag/graph_store/test_int_arangodb_store.py — v2
"""Integration tests for ArangoDB graph store via testcontainers.

Tests every public method of ArangoDBStore against a real ArangoDB instance.
Coverage target: arangodb_store.py 16% → 90%+

API notes (from source review):
- get_node / get_edges / query_by_properties strip all keys starting with '_'
- query_neighbors returns RAW ArangoDB docs (including _key, _id, _rev)
- upsert_edge key = f"{key(source)}__{relation}__{key(target)}"
- _node_key replaces '/' and ' ' with '_'

Container: session-scoped via conftest.arangodb_container
Isolation: per-test unique graph_name + collections via conftest.arangodb_store
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.arangodb]


# =====================================================================
#  NODE CRUD
# =====================================================================


class TestNodeUpsert:
    """upsert_node(node_id: str, properties: dict) -> None"""

    @pytest.mark.asyncio
    async def test_insert_new_node(self, arangodb_store):
        await arangodb_store.upsert_node("eu", {
            "label": "European Union",
            "entity_type": "organization",
            "score": 0.95,
        })
        node = await arangodb_store.get_node("eu")
        assert node is not None
        assert node["node_id"] == "eu"
        assert node["label"] == "European Union"
        assert node["entity_type"] == "organization"
        assert node["score"] == 0.95

    @pytest.mark.asyncio
    async def test_update_existing_node(self, arangodb_store):
        await arangodb_store.upsert_node("e1", {"label": "old", "v": 1})
        await arangodb_store.upsert_node("e1", {"label": "new", "v": 2})
        node = await arangodb_store.get_node("e1")
        assert node["label"] == "new"
        assert node["v"] == 2

    @pytest.mark.asyncio
    async def test_upsert_preserves_node_id(self, arangodb_store):
        """node_id field is always set on the document."""
        await arangodb_store.upsert_node("my_id", {"label": "X"})
        node = await arangodb_store.get_node("my_id")
        assert node["node_id"] == "my_id"


class TestNodeGet:
    """get_node(node_id: str) -> dict | None"""

    @pytest.mark.asyncio
    async def test_get_existing(self, arangodb_store):
        await arangodb_store.upsert_node("a", {"label": "A"})
        node = await arangodb_store.get_node("a")
        assert isinstance(node, dict)
        assert node["label"] == "A"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, arangodb_store):
        result = await arangodb_store.get_node("nonexistent_xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_strips_underscore_keys(self, arangodb_store):
        """Returned dict should NOT contain _key, _id, _rev."""
        await arangodb_store.upsert_node("n1", {"label": "test"})
        node = await arangodb_store.get_node("n1")
        for key in node:
            assert not key.startswith("_"), f"Unexpected key: {key}"

    @pytest.mark.asyncio
    async def test_get_with_special_chars_in_id(self, arangodb_store):
        """_node_key replaces '/' and ' ' with '_'."""
        await arangodb_store.upsert_node("org/EU HQ", {"label": "EU HQ"})
        node = await arangodb_store.get_node("org/EU HQ")
        assert node is not None
        assert node["label"] == "EU HQ"
        assert node["node_id"] == "org/EU HQ"


class TestNodeDelete:
    """delete_node(node_id: str) -> None — cascades edges"""

    @pytest.mark.asyncio
    async def test_delete_removes_node(self, arangodb_store):
        await arangodb_store.upsert_node("x", {"label": "X"})
        assert await arangodb_store.get_node("x") is not None
        await arangodb_store.delete_node("x")
        assert await arangodb_store.get_node("x") is None

    @pytest.mark.asyncio
    async def test_delete_cascades_edges(self, arangodb_store):
        await arangodb_store.upsert_node("a", {"label": "A"})
        await arangodb_store.upsert_node("b", {"label": "B"})
        await arangodb_store.upsert_edge("a", "rel", "b", {"w": 1})
        assert await arangodb_store.edge_count() == 1

        await arangodb_store.delete_node("a")
        assert await arangodb_store.get_node("a") is None
        # Edge should be gone too
        edges = await arangodb_store.get_edges("b", direction="both")
        assert len(edges) == 0
        assert await arangodb_store.edge_count() == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_no_error(self, arangodb_store):
        """Deleting a missing node should not raise."""
        await arangodb_store.delete_node("does_not_exist")


class TestNodeCount:
    """node_count() -> int"""

    @pytest.mark.asyncio
    async def test_count_empty(self, arangodb_store):
        assert await arangodb_store.node_count() == 0

    @pytest.mark.asyncio
    async def test_count_after_inserts(self, arangodb_store):
        await arangodb_store.upsert_node("n1", {"label": "X"})
        await arangodb_store.upsert_node("n2", {"label": "Y"})
        assert await arangodb_store.node_count() == 2

    @pytest.mark.asyncio
    async def test_count_after_upsert_same_id(self, arangodb_store):
        """Upsert same ID twice → still 1 node."""
        await arangodb_store.upsert_node("n1", {"label": "v1"})
        await arangodb_store.upsert_node("n1", {"label": "v2"})
        assert await arangodb_store.node_count() == 1


# =====================================================================
#  EDGE CRUD
# =====================================================================


class TestEdgeUpsert:
    """upsert_edge(source_id, relation_type, target_id, properties) -> None"""

    @pytest.mark.asyncio
    async def test_insert_edge(self, arangodb_store):
        await arangodb_store.upsert_node("eu", {"label": "EU"})
        await arangodb_store.upsert_node("nis2", {"label": "NIS2"})
        await arangodb_store.upsert_edge("eu", "regulates", "nis2", {
            "confidence": 0.9,
            "source_chunk": "chunk_001",
        })
        assert await arangodb_store.edge_count() == 1

    @pytest.mark.asyncio
    async def test_update_existing_edge(self, arangodb_store):
        """Same (source, relation, target) → update, not duplicate."""
        await arangodb_store.upsert_node("a", {"label": "A"})
        await arangodb_store.upsert_node("b", {"label": "B"})
        await arangodb_store.upsert_edge("a", "r", "b", {"weight": 1.0})
        await arangodb_store.upsert_edge("a", "r", "b", {"weight": 2.0})
        assert await arangodb_store.edge_count() == 1
        edges = await arangodb_store.get_edges("a", direction="out")
        assert edges[0]["weight"] == 2.0

    @pytest.mark.asyncio
    async def test_edge_stores_relation_type(self, arangodb_store):
        await arangodb_store.upsert_node("a", {"label": "A"})
        await arangodb_store.upsert_node("b", {"label": "B"})
        await arangodb_store.upsert_edge("a", "enacted", "b", {})
        edges = await arangodb_store.get_edges("a", direction="out")
        assert edges[0]["relation_type"] == "enacted"


class TestEdgeGet:
    """get_edges(node_id, direction, relation_type) -> list[dict]"""

    @pytest.mark.asyncio
    async def test_get_outgoing(self, arangodb_store):
        await arangodb_store.upsert_node("a", {"label": "A"})
        await arangodb_store.upsert_node("b", {"label": "B"})
        await arangodb_store.upsert_edge("a", "r1", "b", {"v": 1})
        edges = await arangodb_store.get_edges("a", direction="out")
        assert len(edges) == 1
        assert edges[0]["relation_type"] == "r1"
        assert edges[0]["v"] == 1

    @pytest.mark.asyncio
    async def test_get_incoming(self, arangodb_store):
        await arangodb_store.upsert_node("a", {"label": "A"})
        await arangodb_store.upsert_node("b", {"label": "B"})
        await arangodb_store.upsert_edge("a", "r1", "b", {})
        edges_in = await arangodb_store.get_edges("b", direction="in")
        assert len(edges_in) == 1
        edges_out = await arangodb_store.get_edges("b", direction="out")
        assert len(edges_out) == 0

    @pytest.mark.asyncio
    async def test_get_both_directions(self, arangodb_store):
        await arangodb_store.upsert_node("a", {"label": "A"})
        await arangodb_store.upsert_node("b", {"label": "B"})
        await arangodb_store.upsert_node("c", {"label": "C"})
        await arangodb_store.upsert_edge("a", "r1", "b", {})
        await arangodb_store.upsert_edge("c", "r2", "b", {})
        # b has 1 incoming from a, 1 incoming from c
        edges = await arangodb_store.get_edges("b", direction="both")
        assert len(edges) == 2

    @pytest.mark.asyncio
    async def test_filter_by_relation_type(self, arangodb_store):
        await arangodb_store.upsert_node("x", {"label": "X"})
        await arangodb_store.upsert_node("y", {"label": "Y"})
        await arangodb_store.upsert_edge("x", "type_a", "y", {})
        await arangodb_store.upsert_edge("x", "type_b", "y", {})
        filtered = await arangodb_store.get_edges("x", direction="out", relation_type="type_a")
        assert len(filtered) == 1
        assert filtered[0]["relation_type"] == "type_a"

    @pytest.mark.asyncio
    async def test_get_strips_underscore_keys(self, arangodb_store):
        """Returned edge dicts should NOT contain _key, _id, _rev, _from, _to."""
        await arangodb_store.upsert_node("a", {"label": "A"})
        await arangodb_store.upsert_node("b", {"label": "B"})
        await arangodb_store.upsert_edge("a", "r", "b", {"score": 0.5})
        edges = await arangodb_store.get_edges("a", direction="out")
        for key in edges[0]:
            assert not key.startswith("_"), f"Unexpected key: {key}"

    @pytest.mark.asyncio
    async def test_get_no_edges_returns_empty(self, arangodb_store):
        await arangodb_store.upsert_node("lonely", {"label": "L"})
        edges = await arangodb_store.get_edges("lonely", direction="both")
        assert edges == []


class TestEdgeDelete:
    """delete_edges(source_id, relation_type, target_id) -> int"""

    @pytest.mark.asyncio
    async def test_delete_all_from_source(self, arangodb_store):
        await arangodb_store.upsert_node("s", {"label": "S"})
        await arangodb_store.upsert_node("t1", {"label": "T1"})
        await arangodb_store.upsert_node("t2", {"label": "T2"})
        await arangodb_store.upsert_edge("s", "r", "t1", {})
        await arangodb_store.upsert_edge("s", "r", "t2", {})
        deleted = await arangodb_store.delete_edges("s")
        assert deleted == 2
        assert await arangodb_store.edge_count() == 0

    @pytest.mark.asyncio
    async def test_delete_by_relation_type(self, arangodb_store):
        await arangodb_store.upsert_node("s", {"label": "S"})
        await arangodb_store.upsert_node("t1", {"label": "T1"})
        await arangodb_store.upsert_node("t2", {"label": "T2"})
        await arangodb_store.upsert_edge("s", "rel", "t1", {})
        await arangodb_store.upsert_edge("s", "rel", "t2", {})
        await arangodb_store.upsert_edge("s", "other", "t1", {})
        deleted = await arangodb_store.delete_edges("s", relation_type="rel")
        assert deleted == 2
        remaining = await arangodb_store.get_edges("s", direction="out")
        assert len(remaining) == 1
        assert remaining[0]["relation_type"] == "other"

    @pytest.mark.asyncio
    async def test_delete_by_relation_and_target(self, arangodb_store):
        await arangodb_store.upsert_node("s", {"label": "S"})
        await arangodb_store.upsert_node("t1", {"label": "T1"})
        await arangodb_store.upsert_node("t2", {"label": "T2"})
        await arangodb_store.upsert_edge("s", "rel", "t1", {})
        await arangodb_store.upsert_edge("s", "rel", "t2", {})
        deleted = await arangodb_store.delete_edges("s", relation_type="rel", target_id="t1")
        assert deleted == 1
        assert await arangodb_store.edge_count() == 1

    @pytest.mark.asyncio
    async def test_delete_returns_zero_if_none_match(self, arangodb_store):
        await arangodb_store.upsert_node("s", {"label": "S"})
        deleted = await arangodb_store.delete_edges("s", relation_type="nope")
        assert deleted == 0


class TestEdgeCount:
    """edge_count() -> int"""

    @pytest.mark.asyncio
    async def test_count_empty(self, arangodb_store):
        assert await arangodb_store.edge_count() == 0

    @pytest.mark.asyncio
    async def test_count_after_inserts(self, arangodb_store):
        await arangodb_store.upsert_node("a", {"label": "A"})
        await arangodb_store.upsert_node("b", {"label": "B"})
        await arangodb_store.upsert_edge("a", "r", "b", {})
        assert await arangodb_store.edge_count() == 1


# =====================================================================
#  QUERY / TRAVERSAL
# =====================================================================


class TestQueryNeighbors:
    """query_neighbors(node_id, depth, relation_types) -> {"results": [...]}"""

    @pytest.mark.asyncio
    async def test_depth_1(self, arangodb_store):
        await arangodb_store.upsert_node("center", {"label": "C"})
        for i in range(3):
            await arangodb_store.upsert_node(f"leaf_{i}", {"label": f"L{i}"})
            await arangodb_store.upsert_edge("center", "conn", f"leaf_{i}", {})
        result = await arangodb_store.query_neighbors("center", depth=1)
        assert "results" in result
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_depth_2(self, arangodb_store):
        await arangodb_store.upsert_node("a", {"label": "A"})
        await arangodb_store.upsert_node("b", {"label": "B"})
        await arangodb_store.upsert_node("c", {"label": "C"})
        await arangodb_store.upsert_edge("a", "r", "b", {})
        await arangodb_store.upsert_edge("b", "r", "c", {})

        d1 = await arangodb_store.query_neighbors("a", depth=1)
        d1_ids = {r["node"]["node_id"] for r in d1["results"]}
        assert d1_ids == {"b"}

        d2 = await arangodb_store.query_neighbors("a", depth=2)
        d2_ids = {r["node"]["node_id"] for r in d2["results"]}
        assert "b" in d2_ids
        assert "c" in d2_ids

    @pytest.mark.asyncio
    async def test_results_contain_raw_docs(self, arangodb_store):
        """query_neighbors returns raw ArangoDB docs (with _key, _id)."""
        await arangodb_store.upsert_node("a", {"label": "A"})
        await arangodb_store.upsert_node("b", {"label": "B"})
        await arangodb_store.upsert_edge("a", "r", "b", {})
        result = await arangodb_store.query_neighbors("a", depth=1)
        entry = result["results"][0]
        # Raw docs have _key and _id
        assert "_key" in entry["node"] or "_id" in entry["node"]
        assert "node_id" in entry["node"]

    @pytest.mark.asyncio
    async def test_filter_by_relation_types(self, arangodb_store):
        await arangodb_store.upsert_node("a", {"label": "A"})
        await arangodb_store.upsert_node("b", {"label": "B"})
        await arangodb_store.upsert_node("c", {"label": "C"})
        await arangodb_store.upsert_edge("a", "type_x", "b", {})
        await arangodb_store.upsert_edge("a", "type_y", "c", {})
        result = await arangodb_store.query_neighbors("a", depth=1, relation_types=["type_x"])
        ids = {r["node"]["node_id"] for r in result["results"]}
        assert "b" in ids
        assert "c" not in ids

    @pytest.mark.asyncio
    async def test_no_neighbors_returns_empty(self, arangodb_store):
        await arangodb_store.upsert_node("isolated", {"label": "I"})
        result = await arangodb_store.query_neighbors("isolated", depth=1)
        assert result["results"] == []


class TestQueryByProperties:
    """query_by_properties(label, filters, limit) -> list[dict]"""

    @pytest.mark.asyncio
    async def test_filter_by_entity_type(self, arangodb_store):
        await arangodb_store.upsert_node("o1", {"label": "EU", "entity_type": "org"})
        await arangodb_store.upsert_node("o2", {"label": "UN", "entity_type": "org"})
        await arangodb_store.upsert_node("r1", {"label": "NIS2", "entity_type": "reg"})
        orgs = await arangodb_store.query_by_properties(filters={"entity_type": "org"})
        assert len(orgs) == 2
        assert {n["label"] for n in orgs} == {"EU", "UN"}

    @pytest.mark.asyncio
    async def test_filter_by_label(self, arangodb_store):
        await arangodb_store.upsert_node("n1", {"label": "EU", "t": "org"})
        await arangodb_store.upsert_node("n2", {"label": "UN", "t": "org"})
        result = await arangodb_store.query_by_properties(label="EU")
        assert len(result) == 1
        assert result[0]["label"] == "EU"

    @pytest.mark.asyncio
    async def test_limit(self, arangodb_store):
        for i in range(5):
            await arangodb_store.upsert_node(f"n{i}", {"label": f"N{i}", "t": "x"})
        results = await arangodb_store.query_by_properties(filters={"t": "x"}, limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self, arangodb_store):
        await arangodb_store.upsert_node("n1", {"label": "X", "t": "a"})
        results = await arangodb_store.query_by_properties(filters={"t": "zzz"})
        assert results == []

    @pytest.mark.asyncio
    async def test_strips_underscore_keys(self, arangodb_store):
        await arangodb_store.upsert_node("n1", {"label": "X"})
        results = await arangodb_store.query_by_properties(label="X")
        for key in results[0]:
            assert not key.startswith("_"), f"Unexpected key: {key}"

    @pytest.mark.asyncio
    async def test_combined_label_and_filters(self, arangodb_store):
        await arangodb_store.upsert_node("a", {"label": "NIS2", "layer": "L2"})
        await arangodb_store.upsert_node("b", {"label": "NIS2", "layer": "L3"})
        await arangodb_store.upsert_node("c", {"label": "CRA", "layer": "L2"})
        results = await arangodb_store.query_by_properties(label="NIS2", filters={"layer": "L2"})
        assert len(results) == 1
        assert results[0]["node_id"] == "a"


# =====================================================================
#  BULK IMPORT
# =====================================================================


class TestImportGraph:
    """import_graph(nodes, edges) -> None"""

    @pytest.mark.asyncio
    async def test_basic_import(self, arangodb_store):
        nodes = [
            {"node_id": "EU", "label": "European Union", "entity_type": "org"},
            {"node_id": "NIS2", "label": "NIS2 Directive", "entity_type": "reg"},
            {"node_id": "cyber", "label": "cybersecurity", "entity_type": "concept"},
        ]
        edges = [
            {"source": "EU", "relation": "enacted", "target": "NIS2", "confidence": 0.95},
            {"source": "NIS2", "relation": "regulates", "target": "cyber", "confidence": 0.9},
        ]
        await arangodb_store.import_graph(nodes, edges)
        assert await arangodb_store.node_count() == 3
        assert await arangodb_store.edge_count() == 2

    @pytest.mark.asyncio
    async def test_import_traversal(self, arangodb_store):
        """After import, traversal should reach depth-2 nodes."""
        nodes = [
            {"node_id": "A", "label": "A"},
            {"node_id": "B", "label": "B"},
            {"node_id": "C", "label": "C"},
        ]
        edges = [
            {"source": "A", "relation": "r", "target": "B"},
            {"source": "B", "relation": "r", "target": "C"},
        ]
        await arangodb_store.import_graph(nodes, edges)
        result = await arangodb_store.query_neighbors("A", depth=2)
        reached = {r["node"]["node_id"] for r in result["results"]}
        assert "B" in reached
        assert "C" in reached

    @pytest.mark.asyncio
    async def test_import_idempotent(self, arangodb_store):
        """Importing same data twice should not duplicate."""
        nodes = [{"node_id": "X", "label": "X"}]
        edges = []
        await arangodb_store.import_graph(nodes, edges)
        await arangodb_store.import_graph(nodes, edges)
        assert await arangodb_store.node_count() == 1

    @pytest.mark.asyncio
    async def test_import_incremental(self, arangodb_store):
        """Second import adds new nodes/edges alongside existing ones."""
        await arangodb_store.import_graph(
            nodes=[
                {"node_id": "EU", "label": "EU", "entity_type": "org"},
                {"node_id": "NIS2", "label": "NIS2", "entity_type": "reg"},
            ],
            edges=[{"source": "EU", "relation": "enacted", "target": "NIS2"}],
        )
        # Second document's graph
        await arangodb_store.import_graph(
            nodes=[
                {"node_id": "CRA", "label": "CRA", "entity_type": "reg"},
                {"node_id": "EU", "label": "EU", "entity_type": "org", "updated": True},
            ],
            edges=[{"source": "EU", "relation": "enacted", "target": "CRA"}],
        )
        assert await arangodb_store.node_count() == 3  # EU, NIS2, CRA
        assert await arangodb_store.edge_count() == 2
        # EU should have updated property
        eu = await arangodb_store.get_node("EU")
        assert eu.get("updated") is True

    @pytest.mark.asyncio
    async def test_import_with_extra_edge_properties(self, arangodb_store):
        """Extra properties in edge dicts are preserved."""
        nodes = [
            {"node_id": "s", "label": "S"},
            {"node_id": "t", "label": "T"},
        ]
        edges = [
            {"source": "s", "relation": "r", "target": "t",
             "confidence": 0.85, "source_chunk": "c_042"},
        ]
        await arangodb_store.import_graph(nodes, edges)
        edge_list = await arangodb_store.get_edges("s", direction="out")
        assert edge_list[0]["confidence"] == 0.85
        assert edge_list[0]["source_chunk"] == "c_042"

    @pytest.mark.asyncio
    async def test_import_node_with_id_key(self, arangodb_store):
        """import_graph accepts 'id' as fallback for 'node_id'."""
        nodes = [{"id": "fallback_id", "label": "Fallback"}]
        await arangodb_store.import_graph(nodes, [])
        # The source does: nid = node.get("node_id", node.get("id", ""))
        # So the stored node_id will be "fallback_id"
        node = await arangodb_store.get_node("fallback_id")
        assert node is not None
        assert node["label"] == "Fallback"


# =====================================================================
#  PROVIDER PROPERTY
# =====================================================================


class TestProvider:

    @pytest.mark.asyncio
    async def test_provider_name(self, arangodb_store):
        assert arangodb_store.provider_name == "arangodb"


# =====================================================================
#  INTERNAL HELPERS (indirect tests)
# =====================================================================


class TestNodeKeyEscaping:
    """_node_key escaping behaviour tested via public API."""

    @pytest.mark.asyncio
    async def test_slash_in_id(self, arangodb_store):
        await arangodb_store.upsert_node("a/b", {"label": "AB"})
        assert await arangodb_store.get_node("a/b") is not None

    @pytest.mark.asyncio
    async def test_space_in_id(self, arangodb_store):
        await arangodb_store.upsert_node("hello world", {"label": "HW"})
        assert await arangodb_store.get_node("hello world") is not None

    @pytest.mark.asyncio
    async def test_slash_and_space_edge(self, arangodb_store):
        """Edge between nodes with special chars in IDs."""
        await arangodb_store.upsert_node("org/EU", {"label": "EU"})
        await arangodb_store.upsert_node("reg NIS2", {"label": "NIS2"})
        await arangodb_store.upsert_edge("org/EU", "enacted", "reg NIS2", {})
        edges = await arangodb_store.get_edges("org/EU", direction="out")
        assert len(edges) == 1
        assert edges[0]["relation_type"] == "enacted"
