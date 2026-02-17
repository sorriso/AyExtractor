# tests/integration/rag/test_int_rag_pipeline.py — v2
"""End-to-end RAG pipeline integration: embed → index → retrieve.

Combines: Ollama embedder + Qdrant vector store + ArangoDB graph store.
Coverage targets: indexer.py, retriever pipeline, embedder_factory
"""

from __future__ import annotations

import uuid

import pytest


@pytest.mark.qdrant
@pytest.mark.ollama
@pytest.mark.slow
class TestVectorIndexAndRetrieve:
    """Embed + index into Qdrant + retrieve by similarity."""

    @pytest.mark.asyncio
    async def test_index_and_retrieve_chunks(self, ollama_embedder, qdrant_store):
        col = f"tc_{uuid.uuid4().hex[:8]}"
        chunks = [
            {"id": "c1", "text": "The NIS2 Directive strengthens EU cybersecurity."},
            {"id": "c2", "text": "ISO 21434 covers automotive cybersecurity."},
            {"id": "c3", "text": "Python is a popular programming language."},
            {"id": "c4", "text": "EU Cyber Resilience Act covers product security."},
        ]
        try:
            texts = [c["text"] for c in chunks]
            embeddings = await ollama_embedder.embed_texts(texts)
            await qdrant_store.create_collection(col, dimensions=768)
            await qdrant_store.upsert(
                col, ids=[c["id"] for c in chunks],
                embeddings=embeddings, documents=texts,
                metadatas=[{"chunk_id": c["id"]} for c in chunks],
            )
            assert await qdrant_store.count(col) == 4

            # EU cybersecurity query → should match c1 or c4
            q = await ollama_embedder.embed_query("EU cybersecurity regulation")
            results = await qdrant_store.query(col, query_embedding=q, top_k=2)
            assert len(results) == 2
            result_texts = {r.content for r in results}
            assert result_texts & {chunks[0]["text"], chunks[3]["text"]}

            # Python query → should match c3
            q2 = await ollama_embedder.embed_query("Python programming language")
            results2 = await qdrant_store.query(col, query_embedding=q2, top_k=1)
            assert results2[0].content == chunks[2]["text"]
        finally:
            try:
                qdrant_store._client.delete_collection(col)
            except Exception:
                pass


@pytest.mark.arangodb
class TestGraphIndexAndTraverse:
    """Index knowledge graph into ArangoDB and traverse."""

    @pytest.mark.asyncio
    async def test_index_and_traverse_document_graph(self, arangodb_store):
        nodes = [
            {"node_id": "EU", "label": "European Union", "entity_type": "org", "layer": "L2"},
            {"node_id": "NIS2", "label": "NIS2 Directive", "entity_type": "reg", "layer": "L2"},
            {"node_id": "cyber", "label": "cybersecurity", "entity_type": "concept", "layer": "L2"},
            {"node_id": "2022", "label": "2022", "entity_type": "date", "layer": "L3"},
        ]
        edges = [
            {"source": "EU", "relation": "enacted", "target": "NIS2", "confidence": 0.95},
            {"source": "NIS2", "relation": "regulates", "target": "cyber", "confidence": 0.9},
            {"source": "NIS2", "relation": "effective_date", "target": "2022", "confidence": 0.85},
        ]
        await arangodb_store.import_graph(nodes, edges)
        assert await arangodb_store.node_count() == 4
        assert await arangodb_store.edge_count() == 3

        result = await arangodb_store.query_neighbors("EU", depth=2)
        reached = {r["node"]["node_id"] for r in result["results"]}
        assert "NIS2" in reached
        assert "cyber" in reached

    @pytest.mark.asyncio
    async def test_incremental_update(self, arangodb_store):
        """Add second document's graph to existing store."""
        await arangodb_store.import_graph(
            nodes=[
                {"node_id": "EU", "label": "EU", "entity_type": "org"},
                {"node_id": "NIS2", "label": "NIS2", "entity_type": "reg"},
            ],
            edges=[{"source": "EU", "relation": "enacted", "target": "NIS2"}],
        )
        await arangodb_store.import_graph(
            nodes=[
                {"node_id": "CRA", "label": "CRA", "entity_type": "reg"},
                {"node_id": "EU", "label": "EU", "entity_type": "org", "updated": True},
            ],
            edges=[{"source": "EU", "relation": "enacted", "target": "CRA"}],
        )
        assert await arangodb_store.node_count() == 3
        assert await arangodb_store.edge_count() == 2
        eu = await arangodb_store.get_node("EU")
        assert eu.get("updated") is True


@pytest.mark.qdrant
@pytest.mark.arangodb
@pytest.mark.ollama
@pytest.mark.slow
class TestFullRAGCycle:
    """Complete RAG cycle: embed → index vectors + graph → query → assemble."""

    @pytest.mark.asyncio
    async def test_full_cycle(self, ollama_embedder, qdrant_store, arangodb_store):
        col = f"rag_{uuid.uuid4().hex[:8]}"
        try:
            entities = {
                "EU": "The European Union is a political and economic union of 27 states.",
                "NIS2": "NIS2 is the EU cybersecurity legislation for critical infrastructure.",
                "ISO21434": "ISO 21434 specifies cybersecurity for road vehicles.",
            }
            # Embed + index into Qdrant
            texts = list(entities.values())
            ids = list(entities.keys())
            embeddings = await ollama_embedder.embed_texts(texts)
            await qdrant_store.create_collection(col, dimensions=768)
            await qdrant_store.upsert(
                col, ids=ids, embeddings=embeddings, documents=texts,
                metadatas=[{"source_type": "entity_profile", "entity_id": eid} for eid in ids],
            )

            # Index graph into ArangoDB
            await arangodb_store.import_graph(
                nodes=[{"node_id": k, "label": k, "entity_type": "concept"} for k in entities],
                edges=[{"source": "EU", "relation": "enacted", "target": "NIS2"}],
            )

            # Query
            q_emb = await ollama_embedder.embed_query("EU cybersecurity regulation")
            results = await qdrant_store.query(col, q_emb, top_k=2)
            assert len(results) >= 1
            top_ids = {r.metadata.get("entity_id") for r in results}
            assert top_ids & {"EU", "NIS2"}

            # Traverse from top entity
            top_entity = results[0].metadata.get("entity_id")
            if top_entity:
                neighbors = await arangodb_store.query_neighbors(top_entity, depth=1)
                assert neighbors["results"] is not None

            # Assemble context
            context = "\n".join(r.content for r in results)
            assert "cybersecurity" in context.lower()
        finally:
            try:
                qdrant_store._client.delete_collection(col)
            except Exception:
                pass
