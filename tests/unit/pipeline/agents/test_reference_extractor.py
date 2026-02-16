# tests/unit/pipeline/agents/test_reference_extractor.py — v1
"""Tests for ReferenceExtractorAgent.

Validates reference extraction from full document text with mocked LLM.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from ayextractor.core.models import DocumentStructure, Footnote, Section
from ayextractor.llm.models import LLMResponse
from ayextractor.pipeline.agents.reference_extractor import (
    ReferenceExtractorAgent,
    ReferenceExtractorInput,
)


@pytest.fixture
def agent():
    return ReferenceExtractorAgent()


def _mock_llm(response_data: dict) -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(response_data),
            input_tokens=200,
            output_tokens=150,
            model="test-model",
            provider="test",
            latency_ms=80,
        )
    )
    return llm


class TestReferenceExtractorProperties:
    def test_name(self, agent):
        assert agent.name == "reference_extractor"

    def test_version(self, agent):
        assert agent.version == "1.0.0"

    def test_input_schema(self, agent):
        assert agent.input_schema is ReferenceExtractorInput

    def test_prompt_file_exists(self, agent):
        import os
        assert agent.prompt_file is not None
        assert os.path.isfile(agent.prompt_file)


class TestReferenceExtractorExecution:
    @pytest.mark.asyncio
    async def test_successful_extraction(self, agent):
        llm = _mock_llm({
            "references": [
                {
                    "type": "citation",
                    "text": "[Smith et al., 2023]",
                    "target": "Smith2023",
                    "source_chunk_id": "document",
                },
                {
                    "type": "internal_ref",
                    "text": "see section 3.2",
                    "target": "section 3.2",
                    "source_chunk_id": "document",
                },
            ],
            "confidence": 0.9,
        })
        inp = ReferenceExtractorInput(
            enriched_text="Some text with [Smith et al., 2023] and see section 3.2.",
            document_title="Test Doc",
            language="en",
        )
        output = await agent.execute(inp, llm)
        assert len(output.data["references"]) == 2
        assert output.data["references"][0]["type"] == "citation"
        assert output.data["references"][1]["type"] == "internal_ref"
        assert output.confidence == 0.9

    @pytest.mark.asyncio
    async def test_no_references_found(self, agent):
        llm = _mock_llm({"references": [], "confidence": 1.0})
        inp = ReferenceExtractorInput(
            enriched_text="Plain text without references.",
            document_title="Simple",
            language="en",
        )
        output = await agent.execute(inp, llm)
        assert output.data["references"] == []
        assert output.confidence == 1.0

    @pytest.mark.asyncio
    async def test_json_parse_failure(self, agent):
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=LLMResponse(
                content="INVALID",
                input_tokens=10, output_tokens=5,
                model="test", provider="test", latency_ms=10,
            )
        )
        inp = ReferenceExtractorInput(
            enriched_text="text", document_title="Test", language="en",
        )
        output = await agent.execute(inp, llm)
        assert output.data["references"] == []
        assert output.confidence == 0.0

    @pytest.mark.asyncio
    async def test_malformed_reference_skipped(self, agent):
        llm = _mock_llm({
            "references": [
                {"type": "citation", "text": "valid ref", "source_chunk_id": "document"},
                {"bad_key": "no type"},
            ],
            "confidence": 0.7,
        })
        inp = ReferenceExtractorInput(
            enriched_text="text", document_title="Test", language="en",
        )
        output = await agent.execute(inp, llm)
        assert len(output.data["references"]) == 1

    @pytest.mark.asyncio
    async def test_dict_state_input(self, agent):
        llm = _mock_llm({"references": [], "confidence": 0.5})
        state = {
            "enriched_text": "some text",
            "document_title": "Test",
            "language": "en",
        }
        output = await agent.execute(state, llm)
        assert output.confidence == 0.5

    @pytest.mark.asyncio
    async def test_invalid_state_type(self, agent):
        llm = AsyncMock()
        with pytest.raises(TypeError):
            await agent.execute(42, llm)

    @pytest.mark.asyncio
    async def test_structure_hints_in_prompt(self, agent):
        """Structure metadata should appear in the formatted prompt."""
        structure = DocumentStructure(
            has_toc=True,
            sections=[Section(title="Introduction", level=1, start_position=0, end_position=500)],
            has_bibliography=True,
            bibliography_position=42,
            has_annexes=False,
            annexes=[],
            footnotes=[Footnote(id="1", content="A footnote", position=100)],
            has_index=False,
        )
        inp = ReferenceExtractorInput(
            enriched_text="text",
            document_title="Test",
            language="en",
            structure=structure,
        )
        prompt = agent._format_prompt(inp)
        assert "Bibliography detected" in prompt
        assert "Footnotes detected: 1" in prompt

    @pytest.mark.asyncio
    async def test_metadata_populated(self, agent):
        llm = _mock_llm({"references": [], "confidence": 0.5})
        inp = ReferenceExtractorInput(
            enriched_text="text", document_title="Test", language="en",
        )
        output = await agent.execute(inp, llm)
        assert output.metadata.agent_name == "reference_extractor"
        assert output.metadata.llm_calls == 1
        assert output.metadata.tokens_used == 350


class TestReferenceExtractorValidation:
    def test_validate_with_references(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"references": [{"type": "citation"}], "confidence": 0.8},
            confidence=0.8,
            metadata=AgentMetadata(
                agent_name="reference_extractor", agent_version="1.0.0",
                execution_time_ms=50, llm_calls=1, tokens_used=300,
            ),
        )
        assert agent.validate_output(output) == 0.8

    def test_validate_no_references(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"references": [], "confidence": 1.0},
            confidence=1.0,
            metadata=AgentMetadata(
                agent_name="reference_extractor", agent_version="1.0.0",
                execution_time_ms=10, llm_calls=1, tokens_used=50,
            ),
        )
        assert agent.validate_output(output) == 0.5
