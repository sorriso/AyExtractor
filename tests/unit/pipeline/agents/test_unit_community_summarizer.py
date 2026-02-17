# tests/unit/pipeline/agents/test_community_summarizer.py — v1
"""Tests for CommunitySummarizerAgent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from ayextractor.graph.layers.models import Community
from ayextractor.llm.models import LLMResponse
from ayextractor.pipeline.agents.community_summarizer import (
    CommunitySummarizerAgent,
    CommunitySummarizerInput,
)


@pytest.fixture
def agent():
    return CommunitySummarizerAgent()


@pytest.fixture
def sample_community():
    return Community(
        community_id="comm_000", level=0,
        members=["EU", "AI Act", "ISO 21434"],
        chunk_coverage=["c1", "c2"],
        modularity_score=0.65,
    )


@pytest.fixture
def sample_graph_data():
    return {
        "nodes": {
            "EU": {"entity_type": "organization", "aliases": []},
            "AI Act": {"entity_type": "document", "aliases": []},
            "ISO 21434": {"entity_type": "document", "aliases": []},
        },
        "edges": [
            {"source": "EU", "target": "AI Act", "relation_type": "regulates"},
            {"source": "ISO 21434", "target": "EU", "relation_type": "requires"},
        ],
    }


def _mock_llm(response_data: dict) -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content=json.dumps(response_data),
        input_tokens=200, output_tokens=150,
        model="test", provider="test", latency_ms=50,
    ))
    return llm


class TestCommunitySummarizerProperties:
    def test_name(self, agent):
        assert agent.name == "community_summarizer"

    def test_version(self, agent):
        assert agent.version == "1.0.0"

    def test_prompt_file_exists(self, agent):
        import os
        assert agent.prompt_file is not None
        assert os.path.isfile(agent.prompt_file)


class TestCommunitySummarizerExecution:
    @pytest.mark.asyncio
    async def test_successful_summarization(self, agent, sample_community, sample_graph_data):
        llm = _mock_llm({
            "title": "AI Governance Framework",
            "summary": "This community covers EU regulation of AI.",
            "key_entities": ["EU", "AI Act"],
            "confidence": 0.85,
        })
        inp = CommunitySummarizerInput(
            community=sample_community,
            graph_data=sample_graph_data,
            language="en",
        )
        output = await agent.execute(inp, llm)
        summary = output.data["summary"]
        assert summary["title"] == "AI Governance Framework"
        assert summary["community_id"] == "comm_000"
        assert output.confidence == 0.85

    @pytest.mark.asyncio
    async def test_json_parse_failure_fallback(self, agent, sample_community, sample_graph_data):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=LLMResponse(
            content="NOT JSON", input_tokens=10, output_tokens=5,
            model="test", provider="test", latency_ms=10,
        ))
        inp = CommunitySummarizerInput(
            community=sample_community,
            graph_data=sample_graph_data,
        )
        output = await agent.execute(inp, llm)
        assert output.data["summary"]["community_id"] == "comm_000"
        assert output.confidence == 0.3

    @pytest.mark.asyncio
    async def test_dict_state_input(self, agent, sample_community, sample_graph_data):
        llm = _mock_llm({"title": "T", "summary": "S", "key_entities": [], "confidence": 0.7})
        state = {
            "community": sample_community.model_dump(),
            "graph_data": sample_graph_data,
            "language": "en",
        }
        output = await agent.execute(state, llm)
        assert output.data["summary"]["title"] == "T"

    @pytest.mark.asyncio
    async def test_invalid_state_type(self, agent):
        with pytest.raises(TypeError):
            await agent.execute("invalid", AsyncMock())

    @pytest.mark.asyncio
    async def test_metadata_populated(self, agent, sample_community, sample_graph_data):
        llm = _mock_llm({"title": "T", "summary": "S", "key_entities": [], "confidence": 0.7})
        inp = CommunitySummarizerInput(
            community=sample_community, graph_data=sample_graph_data,
        )
        output = await agent.execute(inp, llm)
        assert output.metadata.agent_name == "community_summarizer"
        assert output.metadata.llm_calls == 1


class TestCommunitySummarizerValidation:
    def test_validate_with_content(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"summary": {"summary": "Some text"}, "confidence": 0.8},
            confidence=0.8,
            metadata=AgentMetadata(
                agent_name="community_summarizer", agent_version="1.0.0",
                execution_time_ms=50, llm_calls=1, tokens_used=300,
            ),
        )
        assert agent.validate_output(output) == 0.8

    def test_validate_empty_summary(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"summary": {"summary": ""}, "confidence": 0.5},
            confidence=0.5,
            metadata=AgentMetadata(
                agent_name="community_summarizer", agent_version="1.0.0",
                execution_time_ms=10, llm_calls=1, tokens_used=50,
            ),
        )
        assert agent.validate_output(output) == 0.0
