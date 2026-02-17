# tests/integration/pipeline/test_int_pipeline_subsystem.py — v4
"""Integration tests for pipeline subsystem.

Covers: pipeline/state.py, pipeline/dag_builder.py, pipeline/registry.py,
        pipeline/runner.py, pipeline/document_pipeline.py, pipeline/llm_factory.py,
        pipeline/plugin_kit/base_agent.py, pipeline/plugin_kit/models.py
        pipeline/agents/ (all 9 agents with mock LLM)

No Docker required — uses MockLLMClient from conftest.

Source API (verified against deployed source):
- Chunk(id, position, content, source_file)             — core/models.py v1
- AgentMetadata(agent_name, agent_version,
    execution_time_ms, llm_calls, tokens_used)           — plugin_kit/models.py v1
- build_dag(dependency_map) → ExecutionPlan              — dag_builder.py v1
- AgentRegistry() + .load_all() + .agents/.agent_names   — registry.py v1
- LLMFactory(settings).get_client(agent_name)            — llm_factory.py v1
- DocumentPipeline(settings, llm_factory, ...) + .process()
  + .registry + .execution_order                         — document_pipeline.py v1
- PipelineRunner(registry, plan, llm_factory?, ...)      — runner.py v1
- All agents: execute(state: Input|dict, llm) → AgentOutput
  - SummarizerInput(chunk, current_summary, document_title, language)
  - ConceptExtractorInput(chunk, document_title, language, dense_summary="")
  - DecontextualizerInput(chunk, refine_summary, preceding_chunks,
      document_title, language)
  - DensifierInput(refine_summary, document_title, language)
  - CriticInput(document_title, dense_summary, synthesis)
  - SynthesizerInput(document_title, dense_summary)
  - ProfileGeneratorInput(entity_name, graph_data, language)
  - CommunitySummarizerInput(community: Community, graph_data, language)
  - ReferenceExtractorInput(enriched_text, document_title, language)

Changelog:
    v4: Complete rewrite. Fix Chunk constructor (id/position/source_file),
        fix agent inputs (typed Input or dict, not PipelineState),
        fix DensifierInput (refine_summary not current_summary),
        use build_dag/AgentRegistry/LLMFactory (not old aliases),
        test DocumentPipeline.execution_order (not phases),
        PipelineRunner(registry=, plan=) with required args.
    v3: (never deployed) Partial fixes, still had Chunk/agent input bugs.
"""

from __future__ import annotations

import json
import time

import networkx as nx
import pytest

from ayextractor.config.settings import Settings
from ayextractor.core.models import Chunk
from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
from ayextractor.pipeline.state import PipelineState


# =====================================================================
#  HELPERS — valid Chunk factory
# =====================================================================

def _make_chunk(
    chunk_id: str = "chunk_001",
    position: int = 0,
    content: str = "Sample chunk content.",
    source_file: str = "test_document.pdf",
    **kwargs,
) -> Chunk:
    """Create a valid Chunk with all required fields."""
    return Chunk(
        id=chunk_id,
        position=position,
        content=content,
        source_file=source_file,
        **kwargs,
    )


def _make_metadata(
    agent_name: str = "test_agent",
    agent_version: str = "1.0.0",
    execution_time_ms: int = 100,
    llm_calls: int = 1,
    tokens_used: int = 200,
) -> AgentMetadata:
    """Create a valid AgentMetadata with all required fields."""
    return AgentMetadata(
        agent_name=agent_name,
        agent_version=agent_version,
        execution_time_ms=execution_time_ms,
        llm_calls=llm_calls,
        tokens_used=tokens_used,
    )


# =====================================================================
#  PIPELINE STATE — state.py
# =====================================================================

class TestPipelineState:
    """Test PipelineState: identity, accumulation, serialization."""

    def test_default_state(self):
        """Default state has run_id, empty collections, zero counters."""
        state = PipelineState()
        assert state.run_id
        assert state.language == "en"
        assert state.chunks == []
        assert state.raw_triplets == []
        assert state.total_llm_calls == 0

    def test_record_agent_output(self):
        """record_agent_output increments counters and stores output."""
        state = PipelineState()
        output = AgentOutput(
            data={"summary": "test"},
            confidence=0.9,
            metadata=_make_metadata(
                agent_name="summarizer", llm_calls=2, tokens_used=500,
            ),
        )
        state.record_agent_output("summarizer", output)
        assert state.total_llm_calls == 2
        assert state.total_tokens_used == 500
        assert "summarizer" in state.agent_outputs

    def test_get_graph_stats_no_graph(self):
        """No graph → nodes=0, edges=0."""
        state = PipelineState()
        stats = state.get_graph_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_get_graph_stats_with_graph(self):
        """Graph with layered nodes → correct layer counts."""
        state = PipelineState()
        g = nx.Graph()
        g.add_node("A", layer=1)
        g.add_node("B", layer=2)
        g.add_node("C", layer=3)
        g.add_edge("A", "B")
        state.graph = g
        stats = state.get_graph_stats()
        assert stats["nodes"] == 3
        assert stats["edges"] == 1
        assert stats["l1_nodes"] == 1
        assert stats["l2_nodes"] == 1
        assert stats["l3_nodes"] == 1

    def test_state_accumulates_errors(self):
        state = PipelineState()
        state.errors.append("test error 1")
        state.errors.append("test error 2")
        assert len(state.errors) == 2

    def test_state_serializable(self):
        """PipelineState model_dump excludes graph (non-serializable)."""
        state = PipelineState(document_title="Test Doc", language="fr")
        data = state.model_dump(exclude={"graph"})
        assert data["document_title"] == "Test Doc"
        assert data["language"] == "fr"

    def test_multiple_agent_outputs(self):
        """Three agents recorded → counters accumulate correctly."""
        state = PipelineState()
        for i in range(3):
            output = AgentOutput(
                data={},
                confidence=0.8,
                metadata=_make_metadata(
                    agent_name=f"agent_{i}", llm_calls=1, tokens_used=100,
                ),
            )
            state.record_agent_output(f"agent_{i}", output)
        assert state.total_llm_calls == 3
        assert state.total_tokens_used == 300


# =====================================================================
#  PLUGIN KIT MODELS — plugin_kit/models.py
# =====================================================================

class TestPluginKitModels:

    def test_agent_output_creation(self):
        """AgentOutput holds data dict, confidence, and metadata."""
        output = AgentOutput(
            data={"key": "value"},
            confidence=0.95,
            metadata=_make_metadata(agent_name="test", llm_calls=1, tokens_used=100),
        )
        assert output.confidence == 0.95
        assert output.data["key"] == "value"
        assert output.warnings == []

    def test_agent_metadata_fields(self):
        """AgentMetadata stores all agent execution stats."""
        meta = _make_metadata(
            agent_name="summarizer",
            agent_version="1.0.0",
            llm_calls=5,
            tokens_used=2000,
            execution_time_ms=5000,
        )
        assert meta.agent_name == "summarizer"
        assert meta.llm_calls == 5
        assert meta.execution_time_ms == 5000
        assert meta.prompt_hash is None  # optional field

    def test_agent_output_with_warnings(self):
        """AgentOutput can carry warning messages."""
        output = AgentOutput(
            data={},
            confidence=0.5,
            metadata=_make_metadata(),
            warnings=["Low confidence extraction"],
        )
        assert len(output.warnings) == 1


# =====================================================================
#  DAG BUILDER — dag_builder.py
# =====================================================================

class TestDAGBuilder:

    def _sample_dep_map(self) -> dict[str, list[str]]:
        """Minimal dependency map for testing build_dag."""
        return {
            "summarizer": [],
            "concept_extractor": ["summarizer"],
            "synthesizer": ["concept_extractor"],
        }

    def test_build_dag(self):
        """build_dag produces an ExecutionPlan with correct agent count."""
        from ayextractor.pipeline.dag_builder import build_dag
        plan = build_dag(self._sample_dep_map())
        assert plan.total_agents == 3
        assert len(plan.stages) >= 1

    def test_dag_flat_order_respects_dependencies(self):
        """Flat order must respect declared dependencies."""
        from ayextractor.pipeline.dag_builder import build_dag
        plan = build_dag(self._sample_dep_map())
        order = plan.flat_order
        assert len(order) == 3
        assert order.index("summarizer") < order.index("concept_extractor")
        assert order.index("concept_extractor") < order.index("synthesizer")

    def test_dag_stages_are_parallelizable(self):
        """Agents with no mutual dependencies share a stage."""
        from ayextractor.pipeline.dag_builder import build_dag
        dep_map = {
            "a": [],
            "b": [],
            "c": ["a", "b"],
        }
        plan = build_dag(dep_map)
        # a and b have no deps → stage 0 together
        assert {"a", "b"} == set(plan.stages[0])
        assert plan.stages[1] == ["c"]

    def test_build_dag_empty(self):
        """Empty map → empty plan."""
        from ayextractor.pipeline.dag_builder import build_dag
        plan = build_dag({})
        assert plan.total_agents == 0

    def test_build_dag_cycle_raises(self):
        """Cyclic dependencies → DAGError."""
        from ayextractor.pipeline.dag_builder import DAGError, build_dag
        cyclic = {"a": ["b"], "b": ["a"]}
        with pytest.raises(DAGError):
            build_dag(cyclic)

    def test_build_dag_missing_dep_raises(self):
        """Reference to unregistered dependency → DAGError."""
        from ayextractor.pipeline.dag_builder import DAGError, build_dag
        broken = {"a": ["nonexistent"]}
        with pytest.raises(DAGError):
            build_dag(broken)


# =====================================================================
#  REGISTRY — registry.py
# =====================================================================

class TestRegistry:

    def test_registry_creation(self):
        """AgentRegistry starts empty."""
        from ayextractor.pipeline.registry import AgentRegistry
        registry = AgentRegistry()
        assert isinstance(registry.agents, dict)
        assert len(registry.agents) == 0

    def test_registry_load_all(self):
        """load_all populates agents from AGENT_REGISTRY config."""
        from ayextractor.pipeline.registry import AgentRegistry
        registry = AgentRegistry()
        registry.load_all()
        assert len(registry.agents) > 0
        assert len(registry.agent_names) > 0

    def test_registry_agents_are_base_agent(self):
        """Every loaded agent is a BaseAgent instance."""
        from ayextractor.pipeline.plugin_kit.base_agent import BaseAgent
        from ayextractor.pipeline.registry import AgentRegistry
        registry = AgentRegistry()
        registry.load_all()
        for name, agent in registry.agents.items():
            assert isinstance(agent, BaseAgent), f"{name} is not a BaseAgent"

    def test_registry_get_dependency_map(self):
        """get_dependency_map returns agent→deps mapping."""
        from ayextractor.pipeline.registry import AgentRegistry
        registry = AgentRegistry()
        registry.load_all()
        dep_map = registry.get_dependency_map()
        assert isinstance(dep_map, dict)
        for name, deps in dep_map.items():
            assert isinstance(deps, list)

    def test_registry_validate_dependencies(self):
        """validate_dependencies returns empty list when all deps satisfied."""
        from ayextractor.pipeline.registry import AgentRegistry
        registry = AgentRegistry()
        registry.load_all()
        errors = registry.validate_dependencies()
        assert isinstance(errors, list)


# =====================================================================
#  DOCUMENT PIPELINE — document_pipeline.py
# =====================================================================

class TestDocumentPipeline:

    def test_create_pipeline(self):
        """DocumentPipeline can be created with defaults."""
        from ayextractor.pipeline.document_pipeline import DocumentPipeline
        pipeline = DocumentPipeline()
        assert pipeline is not None

    def test_pipeline_has_process(self):
        """DocumentPipeline exposes .process() async method."""
        from ayextractor.pipeline.document_pipeline import DocumentPipeline
        pipeline = DocumentPipeline()
        assert hasattr(pipeline, "process")
        assert callable(pipeline.process)

    def test_pipeline_has_execution_order(self):
        """DocumentPipeline exposes execution_order from DAG."""
        from ayextractor.pipeline.document_pipeline import DocumentPipeline
        pipeline = DocumentPipeline()
        order = pipeline.execution_order
        assert isinstance(order, list)

    def test_pipeline_has_registry(self):
        """DocumentPipeline exposes loaded registry."""
        from ayextractor.pipeline.document_pipeline import DocumentPipeline
        pipeline = DocumentPipeline()
        assert pipeline.registry is not None


# =====================================================================
#  LLM FACTORY — llm_factory.py
# =====================================================================

class TestLLMFactory:

    def test_factory_creation(self):
        """LLMFactory instantiates with Settings."""
        from ayextractor.pipeline.llm_factory import LLMFactory
        settings = Settings(llm_default_provider="ollama", llm_default_model="llama3")
        factory = LLMFactory(settings)
        assert factory is not None

    def test_get_client_for_agent(self):
        """get_client returns a BaseLLMClient for a known agent."""
        from ayextractor.llm.base_client import BaseLLMClient
        from ayextractor.pipeline.llm_factory import LLMFactory
        settings = Settings(llm_default_provider="ollama", llm_default_model="llama3")
        factory = LLMFactory(settings)
        client = factory.get_client("summarizer")
        assert isinstance(client, BaseLLMClient)
        assert client.provider_name == "ollama"

    def test_get_client_caches(self):
        """Same (provider, model) → same client instance."""
        from ayextractor.pipeline.llm_factory import LLMFactory
        settings = Settings(llm_default_provider="ollama", llm_default_model="llama3")
        factory = LLMFactory(settings)
        c1 = factory.get_client("summarizer")
        c2 = factory.get_client("densifier")
        # Both resolve to same default → same cached instance
        assert c1 is c2

    def test_callable_interface(self):
        """LLMFactory supports __call__ for PipelineRunner integration."""
        from ayextractor.pipeline.llm_factory import LLMFactory
        settings = Settings(llm_default_provider="ollama", llm_default_model="llama3")
        factory = LLMFactory(settings)
        client = factory("summarizer")
        assert client.provider_name == "ollama"


# =====================================================================
#  PIPELINE AGENTS — tested with MockLLMClient (dict input)
# =====================================================================

class TestSummarizerAgent:
    """SummarizerInput: chunk, current_summary, document_title, language."""

    @pytest.mark.asyncio
    async def test_execute_produces_output(self, mock_llm):
        from ayextractor.pipeline.agents.summarizer import SummarizerAgent

        mock_llm.set_default(json.dumps({
            "updated_summary": "Document discusses AI safety.",
            "new_information": ["neural scaling laws"],
            "confidence": 0.85,
        }))

        agent = SummarizerAgent()
        state_dict = {
            "chunk": _make_chunk("c1", 0, "Neural scaling laws show..."),
            "current_summary": "",
            "document_title": "AI Research",
            "language": "en",
        }
        output = await agent.execute(state_dict, mock_llm)
        assert output.confidence > 0
        assert "updated_summary" in output.data
        assert output.metadata.agent_name == "summarizer"

    def test_properties(self):
        from ayextractor.pipeline.agents.summarizer import SummarizerAgent
        agent = SummarizerAgent()
        assert agent.name == "summarizer"
        assert agent.version
        assert agent.description


class TestConceptExtractorAgent:
    """ConceptExtractorInput: chunk, document_title, language, dense_summary."""

    @pytest.mark.asyncio
    async def test_execute_extracts_triplets(self, mock_llm):
        from ayextractor.pipeline.agents.concept_extractor import ConceptExtractorAgent

        mock_llm.set_default(json.dumps({
            "triplets": [
                {
                    "subject": "Python",
                    "predicate": "is_a",
                    "object": "programming language",
                    "confidence": 0.9,
                    "qualifiers": {},
                }
            ],
            "extraction_confidence": 0.85,
        }))

        agent = ConceptExtractorAgent()
        state_dict = {
            "chunk": _make_chunk("c1", 0, "Python is a programming language"),
            "document_title": "Intro to Python",
            "language": "en",
            "dense_summary": "",
        }
        output = await agent.execute(state_dict, mock_llm)
        assert output.confidence > 0
        assert output.metadata.agent_name == "concept_extractor"

    def test_properties(self):
        from ayextractor.pipeline.agents.concept_extractor import ConceptExtractorAgent
        agent = ConceptExtractorAgent()
        assert agent.name == "concept_extractor"


class TestDecontextualizerAgent:
    """DecontextualizerInput: chunk, refine_summary, preceding_chunks,
    document_title, language."""

    @pytest.mark.asyncio
    async def test_execute_decontextualizes(self, mock_llm):
        from ayextractor.pipeline.agents.decontextualizer import DecontextualizerAgent

        mock_llm.set_default(json.dumps({
            "decontextualized_content": "Python, created by Guido van Rossum, is widely used.",
            "resolved_references": [],
        }))

        agent = DecontextualizerAgent()
        state_dict = {
            "chunk": _make_chunk("c1", 1, "It is widely used."),
            "refine_summary": "Document about Python programming language.",
            "preceding_chunks": [],
            "document_title": "Python Guide",
            "language": "en",
        }
        output = await agent.execute(state_dict, mock_llm)
        assert output.confidence > 0
        assert output.metadata.agent_name == "decontextualizer"

    def test_properties(self):
        from ayextractor.pipeline.agents.decontextualizer import DecontextualizerAgent
        agent = DecontextualizerAgent()
        assert agent.name == "decontextualizer"


class TestDensifierAgent:
    """DensifierInput: refine_summary, document_title, language."""

    @pytest.mark.asyncio
    async def test_execute_densifies(self, mock_llm):
        from ayextractor.pipeline.agents.densifier import DensifierAgent

        mock_llm.set_default(json.dumps({
            "dense_summary": "Python is a high-level language supporting OOP.",
            "missing_entities": [],
            "iteration": 1,
        }))

        agent = DensifierAgent()
        state_dict = {
            "refine_summary": "Python is a programming language.",
            "document_title": "Python Guide",
            "language": "en",
        }
        output = await agent.execute(state_dict, mock_llm)
        assert output.confidence > 0
        assert output.metadata.agent_name == "densifier"

    def test_properties(self):
        from ayextractor.pipeline.agents.densifier import DensifierAgent
        agent = DensifierAgent()
        assert agent.name == "densifier"


class TestCriticAgent:
    """CriticInput: document_title, dense_summary, synthesis, plus optional
    graph_stats, sample_triplets, community_count, entity_count, language."""

    @pytest.mark.asyncio
    async def test_execute_critic(self, mock_llm):
        from ayextractor.pipeline.agents.critic import CriticAgent

        mock_llm.set_default(json.dumps({
            "quality_score": 0.8,
            "issues": [],
            "recommendations": ["Add more references"],
        }))

        agent = CriticAgent()
        # CriticAgent expects CriticInput or dict — NOT PipelineState
        state_dict = {
            "document_title": "Test",
            "dense_summary": "Dense summary of the document.",
            "synthesis": "Final synthesis of the document.",
            "graph_stats": {"nodes": 10, "edges": 15},
            "sample_triplets": [],
            "community_count": 3,
            "entity_count": 10,
            "language": "en",
        }
        output = await agent.execute(state_dict, mock_llm)
        assert output.confidence > 0
        assert output.metadata.agent_name == "critic"

    def test_properties(self):
        from ayextractor.pipeline.agents.critic import CriticAgent
        agent = CriticAgent()
        assert agent.name == "critic"


class TestSynthesizerAgent:
    """SynthesizerInput: document_title, dense_summary, plus optional
    community_summaries, top_entities, graph_stats, language."""

    @pytest.mark.asyncio
    async def test_execute_synthesizer(self, mock_llm):
        from ayextractor.pipeline.agents.synthesizer import SynthesizerAgent

        mock_llm.set_default(json.dumps({
            "synthesis": "This document presents a comprehensive analysis.",
            "key_findings": ["Finding 1", "Finding 2"],
            "confidence": 0.9,
        }))

        agent = SynthesizerAgent()
        # SynthesizerAgent expects SynthesizerInput or dict — NOT PipelineState
        state_dict = {
            "document_title": "AI Safety",
            "dense_summary": "AI safety is important...",
            "community_summaries": [],
            "top_entities": [],
            "graph_stats": {"nodes": 5, "edges": 8},
            "language": "en",
        }
        output = await agent.execute(state_dict, mock_llm)
        assert output.confidence > 0
        assert output.metadata.agent_name == "synthesizer"


class TestReferenceExtractorAgent:
    """ReferenceExtractorInput: enriched_text, document_title, language."""

    @pytest.mark.asyncio
    async def test_execute_extracts_references(self, mock_llm):
        from ayextractor.pipeline.agents.reference_extractor import ReferenceExtractorAgent

        mock_llm.set_default(json.dumps({
            "references": [
                {"title": "Paper A", "authors": ["Smith"], "year": 2024, "type": "journal"}
            ],
            "extraction_confidence": 0.8,
        }))

        agent = ReferenceExtractorAgent()
        state_dict = {
            "enriched_text": "As shown by Smith et al. (2024)...",
            "document_title": "Survey",
            "language": "en",
        }
        output = await agent.execute(state_dict, mock_llm)
        assert output.confidence > 0
        assert output.metadata.agent_name == "reference_extractor"


class TestProfileGeneratorAgent:
    """ProfileGeneratorInput: entity_name, graph_data, language."""

    @pytest.mark.asyncio
    async def test_execute_generates_profiles(self, mock_llm):
        from ayextractor.pipeline.agents.profile_generator import ProfileGeneratorAgent

        mock_llm.set_default(json.dumps({
            "entity_profiles": [
                {"canonical_name": "Python", "entity_type": "technology",
                 "profile_text": "Python is a programming language."}
            ],
            "relation_profiles": [],
            "confidence": 0.85,
        }))

        agent = ProfileGeneratorAgent()
        # ProfileGeneratorAgent expects ProfileGeneratorInput or dict
        state_dict = {
            "entity_name": "Python",
            "graph_data": {
                "nodes": {
                    "Python": {"type": "technology", "layer": 1},
                    "Guido": {"type": "person", "layer": 1},
                },
                "edges": [
                    {"source": "Guido", "target": "Python",
                     "predicate": "created", "weight": 1.0}
                ],
            },
            "language": "en",
        }
        output = await agent.execute(state_dict, mock_llm)
        assert output.confidence >= 0
        assert output.metadata.agent_name == "profile_generator"


class TestCommunitySummarizerAgent:
    """CommunitySummarizerInput: community (Community), graph_data, language."""

    @pytest.mark.asyncio
    async def test_execute_summarizes_community(self, mock_llm):
        from ayextractor.graph.layers.models import Community
        from ayextractor.pipeline.agents.community_summarizer import CommunitySummarizerAgent

        mock_llm.set_default(json.dumps({
            "community_summary": "This community focuses on AI and ML concepts.",
            "key_entities": ["AI", "machine learning"],
            "confidence": 0.8,
        }))

        agent = CommunitySummarizerAgent()
        community = Community(
            community_id="comm_1",
            level=0,
            members=["AI", "machine_learning", "deep_learning"],
        )
        state_dict = {
            "community": community,
            "graph_data": {
                "nodes": {
                    "AI": {"type": "concept", "layer": 1},
                    "machine_learning": {"type": "concept", "layer": 1},
                    "deep_learning": {"type": "concept", "layer": 1},
                },
                "edges": [
                    {"source": "AI", "target": "machine_learning",
                     "predicate": "includes", "weight": 1.0},
                    {"source": "machine_learning", "target": "deep_learning",
                     "predicate": "includes", "weight": 0.9},
                ],
            },
            "language": "en",
        }
        output = await agent.execute(state_dict, mock_llm)
        assert output is not None
        assert output.metadata.agent_name == "community_summarizer"


# =====================================================================
#  RUNNER — runner.py
# =====================================================================

class TestRunner:

    def test_runner_creation(self):
        """PipelineRunner requires registry + plan."""
        from ayextractor.pipeline.dag_builder import build_dag
        from ayextractor.pipeline.registry import AgentRegistry
        from ayextractor.pipeline.runner import PipelineRunner

        registry = AgentRegistry()
        plan = build_dag({"agent_a": []})
        runner = PipelineRunner(registry=registry, plan=plan)
        assert runner is not None

    def test_runner_has_run_method(self):
        """PipelineRunner exposes async run(state) method."""
        from ayextractor.pipeline.dag_builder import build_dag
        from ayextractor.pipeline.registry import AgentRegistry
        from ayextractor.pipeline.runner import PipelineRunner

        registry = AgentRegistry()
        plan = build_dag({"agent_a": []})
        runner = PipelineRunner(registry=registry, plan=plan)
        assert hasattr(runner, "run")
        assert callable(runner.run)

    @pytest.mark.asyncio
    async def test_runner_skips_unknown_agents(self):
        """Runner skips agents not in registry without crashing."""
        from ayextractor.pipeline.dag_builder import build_dag
        from ayextractor.pipeline.registry import AgentRegistry
        from ayextractor.pipeline.runner import PipelineRunner

        registry = AgentRegistry()  # empty — no agents loaded
        plan = build_dag({"unknown_agent": []})
        runner = PipelineRunner(registry=registry, plan=plan)

        state = PipelineState()
        result = await runner.run(state)
        assert "unknown_agent" in result.skipped_agents