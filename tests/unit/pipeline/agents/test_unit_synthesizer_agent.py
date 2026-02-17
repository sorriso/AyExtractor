# tests/unit/pipeline/agents/test_synthesizer.py — v1
"""Tests for SynthesizerAgent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from ayextractor.llm.models import LLMResponse
from ayextractor.pipeline.agents.synthesizer import (
    SynthesizerAgent,
    SynthesizerInput,
)


@pytest.fixture
def agent():
    return SynthesizerAgent()


def _mock_llm(response_data: dict) -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content=json.dumps(response_data),
        input_tokens=500, output_tokens=400,
        model="test", provider="test", latency_ms=200,
    ))
    return llm


@pytest.fixture
def sample_input():
    return SynthesizerInput(
        document_title="Cybersecurity Regulation Report",
        dense_summary="This document covers automotive cybersecurity regulations.",
        community_summaries=[
            {"title": "EU Regulations", "summary": "EU regulatory framework for AI."},
            {"title": "Standards", "summary": "ISO standards for cybersecurity."},
        ],
        top_entities=[
            {"canonical_name": "EU", "profile_text": "The EU regulates AI."},
            {"canonical_name": "ISO 21434", "profile_text": "Standard for automotive cyber."},
        ],
        graph_stats={"nodes": 50, "edges": 120, "communities": 3},
        language="en",
    )


class TestSynthesizerProperties:
    def test_name(self, agent):
        assert agent.name == "synthesizer"

    def test_version(self, agent):
        assert agent.version == "1.0.0"

    def test_dependencies(self, agent):
        assert "densifier" in agent.dependencies
        assert "community_summarizer" in agent.dependencies

    def test_prompt_file_exists(self, agent):
        import os
        assert agent.prompt_file is not None
        assert os.path.isfile(agent.prompt_file)


class TestSynthesizerExecution:
    @pytest.mark.asyncio
    async def test_successful_synthesis(self, agent, sample_input):
        llm = _mock_llm({
            "synthesis": "The document presents a comprehensive cybersecurity framework "
                        "spanning EU regulation and international standards." * 5,
            "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
            "confidence": 0.88,
        })
        output = await agent.execute(sample_input, llm)
        assert "cybersecurity" in output.data["synthesis"].lower()
        assert len(output.data["key_findings"]) == 3
        assert output.confidence == 0.88

    @pytest.mark.asyncio
    async def test_json_parse_failure_fallback(self, agent, sample_input):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=LLMResponse(
            content="NOT JSON", input_tokens=10, output_tokens=5,
            model="test", provider="test", latency_ms=10,
        ))
        output = await agent.execute(sample_input, llm)
        assert "Cybersecurity Regulation Report" in output.data["synthesis"]
        assert output.confidence == 0.3

    @pytest.mark.asyncio
    async def test_dict_state_input(self, agent, sample_input):
        llm = _mock_llm({
            "synthesis": "Test synthesis.", "key_findings": [], "confidence": 0.7,
        })
        state = sample_input.model_dump()
        output = await agent.execute(state, llm)
        assert output.data["synthesis"] == "Test synthesis."

    @pytest.mark.asyncio
    async def test_invalid_state_type(self, agent):
        with pytest.raises(TypeError):
            await agent.execute("invalid", AsyncMock())

    @pytest.mark.asyncio
    async def test_metadata_populated(self, agent, sample_input):
        llm = _mock_llm({"synthesis": "S", "key_findings": [], "confidence": 0.7})
        output = await agent.execute(sample_input, llm)
        assert output.metadata.agent_name == "synthesizer"
        assert output.metadata.llm_calls == 1
        assert output.metadata.tokens_used == 900

    @pytest.mark.asyncio
    async def test_empty_communities_and_entities(self, agent):
        inp = SynthesizerInput(
            document_title="Empty", dense_summary="Short.", language="en",
        )
        llm = _mock_llm({"synthesis": "Minimal.", "key_findings": [], "confidence": 0.5})
        output = await agent.execute(inp, llm)
        assert output.data["synthesis"] == "Minimal."


class TestSynthesizerValidation:
    def test_validate_with_long_synthesis(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"synthesis": "A" * 600, "key_findings": []},
            confidence=0.9,
            metadata=AgentMetadata(
                agent_name="synthesizer", agent_version="1.0.0",
                execution_time_ms=200, llm_calls=1, tokens_used=900,
            ),
        )
        score = agent.validate_output(output)
        assert score > 0.5

    def test_validate_empty_synthesis(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"synthesis": "", "key_findings": []},
            confidence=0.5,
            metadata=AgentMetadata(
                agent_name="synthesizer", agent_version="1.0.0",
                execution_time_ms=10, llm_calls=1, tokens_used=50,
            ),
        )
        assert agent.validate_output(output) == 0.0

    def test_validate_short_synthesis(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"synthesis": "Brief.", "key_findings": []},
            confidence=0.8,
            metadata=AgentMetadata(
                agent_name="synthesizer", agent_version="1.0.0",
                execution_time_ms=50, llm_calls=1, tokens_used=200,
            ),
        )
        score = agent.validate_output(output)
        assert score < 0.8  # penalized for shortness
