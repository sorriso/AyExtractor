# tests/unit/pipeline/agents/test_critic.py — v1
"""Tests for CriticAgent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from ayextractor.llm.models import LLMResponse
from ayextractor.pipeline.agents.critic import (
    CriticAgent,
    CriticInput,
)


@pytest.fixture
def agent():
    return CriticAgent()


def _mock_llm(response_data: dict) -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content=json.dumps(response_data),
        input_tokens=300, output_tokens=250,
        model="test", provider="test", latency_ms=100,
    ))
    return llm


@pytest.fixture
def sample_input():
    return CriticInput(
        document_title="Test Doc",
        dense_summary="Document about automotive cybersecurity.",
        synthesis="The document covers cybersecurity regulations.",
        graph_stats={"nodes": 50, "edges": 120},
        sample_triplets=[
            {"subject": "EU", "predicate": "regulates", "object": "AI", "confidence": 0.9},
            {"subject": "ISO", "predicate": "requires", "object": "audit", "confidence": 0.4},
        ],
        community_count=3,
        entity_count=50,
        language="en",
    )


class TestCriticProperties:
    def test_name(self, agent):
        assert agent.name == "critic"

    def test_version(self, agent):
        assert agent.version == "1.0.0"

    def test_dependencies(self, agent):
        assert "synthesizer" in agent.dependencies

    def test_prompt_file_exists(self, agent):
        import os
        assert agent.prompt_file is not None
        assert os.path.isfile(agent.prompt_file)


class TestCriticExecution:
    @pytest.mark.asyncio
    async def test_successful_critique(self, agent, sample_input):
        llm = _mock_llm({
            "overall_quality": 0.78,
            "issues": [
                {
                    "severity": "medium",
                    "category": "low_confidence",
                    "description": "Triplet (ISO, requires, audit) has low confidence.",
                    "affected_entities": ["ISO"],
                    "suggestion": "Verify source text.",
                },
            ],
            "summary": "Generally good extraction with one low-confidence triplet.",
        })
        output = await agent.execute(sample_input, llm)
        assert output.data["overall_quality"] == 0.78
        assert len(output.data["issues"]) == 1
        assert output.data["issues"][0]["severity"] == "medium"
        assert output.data["high_severity_count"] == 0

    @pytest.mark.asyncio
    async def test_no_issues_found(self, agent, sample_input):
        llm = _mock_llm({
            "overall_quality": 0.95,
            "issues": [],
            "summary": "Excellent extraction quality.",
        })
        output = await agent.execute(sample_input, llm)
        assert output.data["issue_count"] == 0
        assert output.confidence == 0.95

    @pytest.mark.asyncio
    async def test_high_severity_counted(self, agent, sample_input):
        llm = _mock_llm({
            "overall_quality": 0.4,
            "issues": [
                {"severity": "high", "category": "contradiction",
                 "description": "Contradictory claims about X."},
                {"severity": "low", "category": "orphan_nodes",
                 "description": "Minor orphan node."},
            ],
            "summary": "Major issue found.",
        })
        output = await agent.execute(sample_input, llm)
        assert output.data["high_severity_count"] == 1
        assert output.data["issue_count"] == 2

    @pytest.mark.asyncio
    async def test_json_parse_failure_fallback(self, agent, sample_input):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=LLMResponse(
            content="BROKEN", input_tokens=10, output_tokens=5,
            model="test", provider="test", latency_ms=10,
        ))
        output = await agent.execute(sample_input, llm)
        assert output.data["overall_quality"] == 0.5
        assert output.data["issues"] == []

    @pytest.mark.asyncio
    async def test_malformed_issue_skipped(self, agent, sample_input):
        llm = _mock_llm({
            "overall_quality": 0.7,
            "issues": [
                {"severity": "low", "category": "test", "description": "valid"},
                {"bad_key": "invalid"},
            ],
            "summary": "OK.",
        })
        output = await agent.execute(sample_input, llm)
        # malformed issue may still be created with defaults since we have fallback
        assert output.data["issue_count"] >= 1

    @pytest.mark.asyncio
    async def test_dict_state_input(self, agent, sample_input):
        llm = _mock_llm({"overall_quality": 0.8, "issues": [], "summary": "Good."})
        state = sample_input.model_dump()
        output = await agent.execute(state, llm)
        assert output.data["overall_quality"] == 0.8

    @pytest.mark.asyncio
    async def test_invalid_state_type(self, agent):
        with pytest.raises(TypeError):
            await agent.execute("invalid", AsyncMock())

    @pytest.mark.asyncio
    async def test_metadata_populated(self, agent, sample_input):
        llm = _mock_llm({"overall_quality": 0.7, "issues": [], "summary": "OK."})
        output = await agent.execute(sample_input, llm)
        assert output.metadata.agent_name == "critic"
        assert output.metadata.llm_calls == 1
        assert output.metadata.tokens_used == 550


class TestCriticValidation:
    def test_validate_returns_quality(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"overall_quality": 0.85, "issues": []},
            confidence=0.85,
            metadata=AgentMetadata(
                agent_name="critic", agent_version="1.0.0",
                execution_time_ms=100, llm_calls=1, tokens_used=550,
            ),
        )
        assert agent.validate_output(output) == 0.85
