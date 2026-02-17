# tests/unit/pipeline/agents/test_concept_extractor.py — v2
"""Tests for ConceptExtractorAgent.

Validates triplet extraction from chunks with mocked LLM responses.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ayextractor.core.models import Chunk
from ayextractor.llm.models import LLMResponse
from ayextractor.pipeline.agents.concept_extractor import (
    ConceptExtractorAgent,
    ConceptExtractorInput,
)


@pytest.fixture
def agent():
    return ConceptExtractorAgent()


@pytest.fixture
def sample_chunk():
    return Chunk(
        id="chunk_001",
        position=0,
        content="The European Union regulates AI systems through the AI Act since 2025.",
        source_file="doc.pdf",
        char_count=70,
        word_count=12,
    )


def _mock_llm(response_data: dict) -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(response_data),
            input_tokens=100,
            output_tokens=200,
            model="test-model",
            provider="test",
            latency_ms=50,
        )
    )
    return llm


class TestConceptExtractorProperties:
    def test_name(self, agent):
        assert agent.name == "concept_extractor"

    def test_version(self, agent):
        assert agent.version == "1.0.0"

    def test_dependencies(self, agent):
        assert agent.dependencies == []  # decontextualizer runs in Phase 2

    def test_input_schema(self, agent):
        assert agent.input_schema is ConceptExtractorInput

    def test_prompt_file_exists(self, agent):
        import os
        assert agent.prompt_file is not None
        assert os.path.isfile(agent.prompt_file)


class TestConceptExtractorExecution:
    @pytest.mark.asyncio
    async def test_successful_extraction(self, agent, sample_chunk):
        llm = _mock_llm({
            "triplets": [
                {
                    "subject": "European Union",
                    "predicate": "regulates",
                    "object": "AI systems",
                    "confidence": 0.92,
                    "context_sentence": "The EU regulates AI systems through the AI Act.",
                    "qualifiers": {"instrument": "AI Act"},
                    "temporal_scope": {
                        "type": "point",
                        "start": "2025",
                        "granularity": "year",
                        "raw_expression": "since 2025",
                    },
                }
            ],
            "extraction_confidence": 0.88,
        })

        inp = ConceptExtractorInput(
            chunk=sample_chunk,
            document_title="AI Regulation Report",
            language="en",
        )
        output = await agent.execute(inp, llm)

        assert output.confidence == 0.88
        assert len(output.data["triplets"]) == 1
        t = output.data["triplets"][0]
        assert t["subject"] == "European Union"
        assert t["predicate"] == "regulates"
        assert t["object"] == "AI systems"
        assert t["qualifiers"]["instrument"] == "AI Act"
        assert t["temporal_scope"]["type"] == "point"

    @pytest.mark.asyncio
    async def test_empty_extraction(self, agent, sample_chunk):
        llm = _mock_llm({"triplets": [], "extraction_confidence": 0.0})
        inp = ConceptExtractorInput(
            chunk=sample_chunk,
            document_title="Empty doc",
            language="en",
        )
        output = await agent.execute(inp, llm)
        assert output.data["triplets"] == []
        assert output.confidence == 0.0

    @pytest.mark.asyncio
    async def test_json_parse_failure_fallback(self, agent, sample_chunk):
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=LLMResponse(
                content="NOT JSON",
                input_tokens=10,
                output_tokens=5,
                model="test",
                provider="test",
                latency_ms=10,
            )
        )
        inp = ConceptExtractorInput(
            chunk=sample_chunk,
            document_title="Test",
            language="en",
        )
        output = await agent.execute(inp, llm)
        assert output.data["triplets"] == []
        assert output.confidence == 0.0

    @pytest.mark.asyncio
    async def test_malformed_triplet_skipped(self, agent, sample_chunk):
        llm = _mock_llm({
            "triplets": [
                {"subject": "A", "predicate": "rel", "object": "B", "confidence": 0.9,
                 "context_sentence": "test"},
                {"bad_key": "no subject"},  # malformed
            ],
            "extraction_confidence": 0.7,
        })
        inp = ConceptExtractorInput(
            chunk=sample_chunk,
            document_title="Test",
            language="en",
        )
        output = await agent.execute(inp, llm)
        assert len(output.data["triplets"]) == 1

    @pytest.mark.asyncio
    async def test_dict_state_input(self, agent, sample_chunk):
        llm = _mock_llm({"triplets": [], "extraction_confidence": 0.5})
        state = {
            "chunk": sample_chunk.model_dump(),
            "document_title": "Test",
            "language": "en",
        }
        output = await agent.execute(state, llm)
        assert output.data["chunk_id"] == "chunk_001"

    @pytest.mark.asyncio
    async def test_invalid_state_type(self, agent):
        llm = AsyncMock()
        with pytest.raises(TypeError):
            await agent.execute("invalid", llm)

    @pytest.mark.asyncio
    async def test_metadata_populated(self, agent, sample_chunk):
        llm = _mock_llm({"triplets": [], "extraction_confidence": 0.5})
        inp = ConceptExtractorInput(
            chunk=sample_chunk, document_title="Test", language="en",
        )
        output = await agent.execute(inp, llm)
        assert output.metadata.agent_name == "concept_extractor"
        assert output.metadata.llm_calls == 1
        assert output.metadata.tokens_used == 300
        assert output.metadata.prompt_hash is not None

    @pytest.mark.asyncio
    async def test_markdown_fenced_json(self, agent, sample_chunk):
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=LLMResponse(
                content='```json\n{"triplets": [], "extraction_confidence": 0.6}\n```',
                input_tokens=10,
                output_tokens=10,
                model="test",
                provider="test",
                latency_ms=5,
            )
        )
        inp = ConceptExtractorInput(
            chunk=sample_chunk, document_title="Test", language="en",
        )
        output = await agent.execute(inp, llm)
        assert output.confidence == 0.6


class TestConceptExtractorValidation:
    def test_validate_with_triplets(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"triplets": [{"confidence": 0.8}], "extraction_confidence": 0.85},
            confidence=0.85,
            metadata=AgentMetadata(
                agent_name="concept_extractor", agent_version="1.0.0",
                execution_time_ms=50, llm_calls=1, tokens_used=300,
            ),
        )
        assert agent.validate_output(output) == 0.8

    def test_validate_empty_triplets(self, agent):
        from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput
        output = AgentOutput(
            data={"triplets": [], "extraction_confidence": 0.0},
            confidence=0.0,
            metadata=AgentMetadata(
                agent_name="concept_extractor", agent_version="1.0.0",
                execution_time_ms=10, llm_calls=1, tokens_used=50,
            ),
        )
        assert agent.validate_output(output) == 0.3
