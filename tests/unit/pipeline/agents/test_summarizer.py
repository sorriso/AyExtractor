# tests/unit/pipeline/agents/test_summarizer.py — v1
"""Tests for pipeline/agents/summarizer.py."""

from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock

from ayextractor.core.models import Chunk
from ayextractor.llm.models import LLMResponse
from ayextractor.pipeline.agents.summarizer import (
    SummarizerAgent,
    SummarizerInput,
)
from ayextractor.pipeline.plugin_kit.models import AgentMetadata, AgentOutput


# --- Fixtures ---

@pytest.fixture
def agent():
    return SummarizerAgent()


@pytest.fixture
def chunk():
    return Chunk(
        id="chunk_002", position=1,
        content="The NIS2 Directive applies to essential and important entities across the EU.",
        source_file="report.pdf", char_count=77, word_count=13,
        token_count_est=18, fingerprint="sum1",
    )


@pytest.fixture
def summarizer_input(chunk):
    return SummarizerInput(
        chunk=chunk,
        current_summary="The document discusses EU cybersecurity regulation.",
        document_title="EU Cybersecurity Report",
        language="en",
    )


def _make_llm_response(content_dict: dict) -> LLMResponse:
    return LLMResponse(
        content=json.dumps(content_dict),
        input_tokens=300, output_tokens=150,
        model="claude-sonnet-4-20250514", provider="anthropic", latency_ms=400,
    )


# --- Properties ---

class TestSummarizerProperties:
    def test_name(self, agent):
        assert agent.name == "summarizer"

    def test_version(self, agent):
        assert agent.version == "1.0.0"

    def test_description_mentions_refine(self, agent):
        assert "refine" in agent.description.lower()

    def test_input_schema(self, agent):
        assert agent.input_schema is SummarizerInput

    def test_prompt_file_exists(self, agent):
        from pathlib import Path
        assert agent.prompt_file is not None
        assert Path(agent.prompt_file).exists()


# --- Prompt building ---

class TestSummarizerPrompt:
    def test_prompt_contains_chunk(self, agent, summarizer_input):
        prompt = agent._format_prompt(summarizer_input)
        assert "NIS2 Directive" in prompt

    def test_prompt_contains_current_summary(self, agent, summarizer_input):
        prompt = agent._format_prompt(summarizer_input)
        assert "cybersecurity regulation" in prompt

    def test_prompt_contains_title(self, agent, summarizer_input):
        prompt = agent._format_prompt(summarizer_input)
        assert "EU Cybersecurity Report" in prompt

    def test_prompt_first_chunk_marker(self, agent, chunk):
        inp = SummarizerInput(
            chunk=chunk, current_summary="",
            document_title="Test", language="en",
        )
        prompt = agent._format_prompt(inp)
        assert "first chunk" in prompt


# --- Execution ---

class TestSummarizerExecution:
    @pytest.mark.asyncio
    async def test_successful_refine(self, agent, summarizer_input):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response({
            "updated_summary": (
                "The document discusses EU cybersecurity regulation. "
                "The NIS2 Directive specifically targets essential and important "
                "entities across EU member states."
            ),
            "new_information": [
                "NIS2 targets essential and important entities",
                "Applies across EU member states",
            ],
            "confidence": 0.88,
        }))

        output = await agent.execute(summarizer_input, llm)

        assert output.confidence == pytest.approx(0.88)
        assert "NIS2" in output.data["updated_summary"]
        assert len(output.data["new_information"]) == 2
        assert output.metadata.agent_name == "summarizer"
        assert output.metadata.llm_calls == 1

    @pytest.mark.asyncio
    async def test_no_new_information(self, agent, chunk):
        """Chunk adds nothing new — summary should stay the same."""
        existing = "The document already covers everything in this chunk."
        inp = SummarizerInput(
            chunk=chunk, current_summary=existing,
            document_title="Test", language="en",
        )
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response({
            "updated_summary": existing,
            "new_information": [],
            "confidence": 0.95,
        }))

        output = await agent.execute(inp, llm)
        assert output.data["updated_summary"] == existing
        assert output.data["new_information"] == []

    @pytest.mark.asyncio
    async def test_first_chunk_empty_summary(self, agent, chunk):
        """First chunk with empty summary — should bootstrap."""
        inp = SummarizerInput(
            chunk=chunk, current_summary="",
            document_title="Test", language="en",
        )
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response({
            "updated_summary": "NIS2 Directive applies to essential entities in the EU.",
            "new_information": ["NIS2 Directive scope"],
            "confidence": 0.8,
        }))

        output = await agent.execute(inp, llm)
        assert "NIS2" in output.data["updated_summary"]

    @pytest.mark.asyncio
    async def test_json_parse_failure(self, agent, summarizer_input):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=LLMResponse(
            content="Broken response",
            input_tokens=100, output_tokens=30,
            model="test", provider="mock", latency_ms=50,
        ))

        output = await agent.execute(summarizer_input, llm)

        # Fallback: keeps existing summary
        assert output.data["updated_summary"] == summarizer_input.current_summary
        assert output.confidence == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_dict_state_input(self, agent, summarizer_input):
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_make_llm_response({
            "updated_summary": "Updated text",
            "new_information": [],
            "confidence": 0.9,
        }))

        output = await agent.execute(summarizer_input.model_dump(), llm)
        assert output.data["updated_summary"] == "Updated text"

    @pytest.mark.asyncio
    async def test_invalid_state_type(self, agent):
        with pytest.raises(TypeError, match="Expected SummarizerInput"):
            await agent.execute(42, AsyncMock())


# --- apply_to_chunk ---

class TestSummarizerApplyToChunk:
    def test_stores_context_summary(self, agent, chunk):
        summary = "Cumulative summary up to chunk 2."
        updated = agent.apply_to_chunk(chunk, summary)
        assert updated.context_summary == summary

    def test_does_not_overwrite_content(self, agent, chunk):
        original_content = chunk.content
        agent.apply_to_chunk(chunk, "Some summary")
        assert chunk.content == original_content


# --- Validation ---

class TestSummarizerValidation:
    def test_validate_good_output(self, agent):
        output = AgentOutput(
            data={"updated_summary": "A reasonable summary of moderate length with details."},
            confidence=0.85,
            metadata=AgentMetadata(
                agent_name="summarizer", agent_version="1.0.0",
                execution_time_ms=100, llm_calls=1, tokens_used=450,
            ),
        )
        assert agent.validate_output(output) == 0.85

    def test_validate_empty_summary(self, agent):
        output = AgentOutput(
            data={"updated_summary": ""},
            confidence=0.9,
            metadata=AgentMetadata(
                agent_name="summarizer", agent_version="1.0.0",
                execution_time_ms=100, llm_calls=1, tokens_used=450,
            ),
        )
        assert agent.validate_output(output) == 0.0

    def test_validate_short_summary_penalized(self, agent):
        output = AgentOutput(
            data={"updated_summary": "Too short."},
            confidence=0.9,
            metadata=AgentMetadata(
                agent_name="summarizer", agent_version="1.0.0",
                execution_time_ms=100, llm_calls=1, tokens_used=450,
            ),
        )
        assert agent.validate_output(output) <= 0.5
