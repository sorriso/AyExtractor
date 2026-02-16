# tests/unit/pipeline/agents/test_profile_generator.py — v1
"""Tests for ProfileGeneratorAgent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from ayextractor.llm.models import LLMResponse
from ayextractor.pipeline.agents.profile_generator import (
    ProfileGeneratorAgent,
    ProfileGeneratorInput,
)


@pytest.fixture
def agent():
    return ProfileGeneratorAgent(min_relations=2)


@pytest.fixture
def graph_data_rich():
    """Graph data where EU has enough relations."""
    return {
        "nodes": {
            "EU": {"entity_type": "organization", "aliases": ["European Union"], "community_id": "comm_000"},
            "AI Act": {"entity_type": "document", "aliases": []},
            "ISO 21434": {"entity_type": "document", "aliases": []},
        },
        "edges": [
            {"source": "EU", "target": "AI Act", "relation_type": "regulates"},
            {"source": "EU", "target": "ISO 21434", "relation_type": "enforces"},
            {"source": "AI Act", "target": "ISO 21434", "relation_type": "references"},
        ],
    }


@pytest.fixture
def graph_data_sparse():
    """Graph data where entity has too few relations."""
    return {
        "nodes": {"Lonely": {"entity_type": "concept", "aliases": []}},
        "edges": [
            {"source": "Lonely", "target": "X", "relation_type": "related_to"},
        ],
    }


def _mock_llm(response_data: dict) -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content=json.dumps(response_data),
        input_tokens=150, output_tokens=100,
        model="test", provider="test", latency_ms=40,
    ))
    return llm


class TestProfileGeneratorProperties:
    def test_name(self, agent):
        assert agent.name == "profile_generator"

    def test_version(self, agent):
        assert agent.version == "1.0.0"

    def test_prompt_file_exists(self, agent):
        import os
        assert agent.prompt_file is not None
        assert os.path.isfile(agent.prompt_file)


class TestProfileGeneratorExecution:
    @pytest.mark.asyncio
    async def test_successful_profile(self, agent, graph_data_rich):
        llm = _mock_llm({
            "profile_text": "The EU is an organization that regulates AI through the AI Act.",
            "key_relations": ["regulates AI Act", "enforces ISO 21434"],
            "confidence": 0.85,
        })
        inp = ProfileGeneratorInput(
            entity_name="EU", graph_data=graph_data_rich, language="en",
        )
        output = await agent.execute(inp, llm)
        profile = output.data["entity_profile"]
        assert profile is not None
        assert profile["canonical_name"] == "EU"
        assert "regulates" in profile["profile_text"].lower() or len(profile["profile_text"]) > 0
        assert output.confidence == 0.85

    @pytest.mark.asyncio
    async def test_skips_sparse_entity(self, agent, graph_data_sparse):
        llm = AsyncMock()
        inp = ProfileGeneratorInput(
            entity_name="Lonely", graph_data=graph_data_sparse, language="en",
        )
        output = await agent.execute(inp, llm)
        assert output.data["skipped"] is True
        assert output.data["reason"] == "too_few_relations"
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_json_parse_failure_fallback(self, agent, graph_data_rich):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=LLMResponse(
            content="BROKEN", input_tokens=10, output_tokens=5,
            model="test", provider="test", latency_ms=10,
        ))
        inp = ProfileGeneratorInput(
            entity_name="EU", graph_data=graph_data_rich, language="en",
        )
        output = await agent.execute(inp, llm)
        profile = output.data["entity_profile"]
        assert profile is not None
        assert "EU" in profile["profile_text"]

    @pytest.mark.asyncio
    async def test_dict_state_input(self, agent, graph_data_rich):
        llm = _mock_llm({
            "profile_text": "Test profile.", "key_relations": [], "confidence": 0.7,
        })
        state = {"entity_name": "EU", "graph_data": graph_data_rich, "language": "en"}
        output = await agent.execute(state, llm)
        assert output.data["entity_profile"]["canonical_name"] == "EU"

    @pytest.mark.asyncio
    async def test_invalid_state_type(self, agent):
        with pytest.raises(TypeError):
            await agent.execute(42, AsyncMock())

    @pytest.mark.asyncio
    async def test_metadata_populated(self, agent, graph_data_rich):
        llm = _mock_llm({"profile_text": "T", "key_relations": [], "confidence": 0.7})
        inp = ProfileGeneratorInput(entity_name="EU", graph_data=graph_data_rich)
        output = await agent.execute(inp, llm)
        assert output.metadata.agent_name == "profile_generator"
        assert output.metadata.llm_calls == 1


class TestProfileGeneratorValidation:
    def test_validate_with_content(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"entity_profile": {"profile_text": "Some content"}},
            confidence=0.8,
            metadata=AgentMetadata(
                agent_name="profile_generator", agent_version="1.0.0",
                execution_time_ms=50, llm_calls=1, tokens_used=200,
            ),
        )
        assert agent.validate_output(output) == 0.8

    def test_validate_skipped(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"skipped": True, "reason": "too_few_relations"},
            confidence=1.0,
            metadata=AgentMetadata(
                agent_name="profile_generator", agent_version="1.0.0",
                execution_time_ms=5, llm_calls=0, tokens_used=0,
            ),
        )
        assert agent.validate_output(output) == 1.0

    def test_validate_empty_profile(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"entity_profile": {"profile_text": ""}},
            confidence=0.5,
            metadata=AgentMetadata(
                agent_name="profile_generator", agent_version="1.0.0",
                execution_time_ms=10, llm_calls=1, tokens_used=50,
            ),
        )
        assert agent.validate_output(output) == 0.0
