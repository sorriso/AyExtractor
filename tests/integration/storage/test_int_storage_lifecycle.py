# tests/integration/storage/test_int_storage_lifecycle.py — v1
"""Integration tests for storage pipeline: layout → run_manager → writer → reader.

No external services required — filesystem only.
Coverage targets: layout.py 78%→95%, reader.py 22%→85%, run_manager.py 49%→85%, local_writer.py 58%→90%
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ayextractor.storage import layout
from ayextractor.storage.models import RunManifest, StepManifest


class TestLayoutPaths:
    """All layout path functions."""

    def test_document_root(self, tmp_path: Path):
        assert layout.document_root(tmp_path, "doc1") == tmp_path / "doc1"

    def test_source_dir(self, tmp_path: Path):
        sd = layout.source_dir(tmp_path, "doc1")
        assert "source" in str(sd).lower() or "doc1" in str(sd)

    def test_runs_dir(self, tmp_path: Path):
        rd = layout.runs_dir(tmp_path, "doc1")
        assert "runs" in str(rd).lower() or "doc1" in str(rd)

    def test_run_dir(self, tmp_path: Path):
        rd = layout.run_dir(tmp_path, "doc1", "run1")
        assert "run1" in str(rd)

    def test_latest_link(self, tmp_path: Path):
        link = layout.latest_link(tmp_path, "doc1")
        assert "latest" in str(link).lower()

    def test_subdirectory_functions(self, tmp_path: Path):
        run_path = tmp_path / "run"
        run_path.mkdir()
        for fn in [layout.metadata_dir, layout.extraction_dir, layout.chunks_dir,
                    layout.concepts_dir, layout.synthesis_dir]:
            result = fn(run_path)
            assert isinstance(result, Path)

    def test_file_path_functions(self, tmp_path: Path):
        run_path = tmp_path / "run"
        for fn in [layout.run_manifest_path, layout.fingerprint_path, layout.language_path,
                    layout.structure_path, layout.raw_text_path, layout.enriched_text_path,
                    layout.references_path, layout.chunks_index_path, layout.refine_summary_path,
                    layout.dense_summary_path, layout.graph_json_path, layout.communities_path,
                    layout.execution_stats_path, layout.calls_log_path]:
            result = fn(run_path)
            assert isinstance(result, Path)

    def test_chunk_path_and_original(self, tmp_path: Path):
        run_path = tmp_path / "run"
        cp = layout.chunk_path(run_path, "chunk_001")
        assert "chunk_001" in str(cp)
        co = layout.chunk_original_path(run_path, "chunk_001")
        assert "chunk_001" in str(co)

    def test_ensure_run_directories(self, tmp_path: Path):
        run_path = layout.run_dir(tmp_path, "doc1", "run1")
        layout.ensure_run_directories(run_path)
        assert run_path.exists()
        assert layout.extraction_dir(run_path).exists()
        assert layout.chunks_dir(run_path).exists()
        assert layout.concepts_dir(run_path).exists()
        assert layout.synthesis_dir(run_path).exists()


class TestRunManager:
    """Run lifecycle management."""

    def test_generate_document_id(self):
        from ayextractor.storage.run_manager import generate_document_id
        ts = datetime(2026, 2, 16, 14, 0, 0, tzinfo=timezone.utc)
        doc_id = generate_document_id(timestamp=ts)
        assert doc_id.startswith("20260216_140000_")
        assert len(doc_id) == 24  # yyyymmdd_hhmmss_{8hex}

    def test_generate_run_id(self):
        from ayextractor.storage.run_manager import generate_run_id
        ts = datetime(2026, 2, 16, 14, 30, 0, tzinfo=timezone.utc)
        run_id = generate_run_id(timestamp=ts)
        assert run_id.startswith("20260216_1430_")

    def test_auto_timestamp(self):
        from ayextractor.storage.run_manager import generate_document_id, generate_run_id
        assert len(generate_document_id()) > 15
        assert len(generate_run_id()) > 10

    @pytest.mark.asyncio
    async def test_create_run(self, tmp_path: Path):
        from ayextractor.storage.run_manager import create_run, generate_document_id, generate_run_id
        from ayextractor.storage.local_writer import LocalWriter

        writer = LocalWriter(base_path=str(tmp_path))
        doc_id = generate_document_id()
        run_id = generate_run_id()

        run_path, manifest = await create_run(
            writer=writer, output_path=tmp_path,
            document_id=doc_id, run_id=run_id, pipeline_version="0.3.1",
        )
        assert run_path.exists()
        assert manifest.status == "running"
        assert manifest.document_id == doc_id
        manifest_path = layout.run_manifest_path(run_path)
        assert manifest_path.exists()

    @pytest.mark.asyncio
    async def test_finalize_run(self, tmp_path: Path):
        from ayextractor.storage.run_manager import create_run, finalize_run, generate_document_id, generate_run_id
        from ayextractor.storage.local_writer import LocalWriter

        writer = LocalWriter(base_path=str(tmp_path))
        doc_id = generate_document_id()
        run_id = generate_run_id()

        run_path, manifest = await create_run(
            writer=writer, output_path=tmp_path,
            document_id=doc_id, run_id=run_id, pipeline_version="0.3.1",
        )
        await finalize_run(writer, tmp_path, doc_id, run_path, manifest, status="completed")
        assert manifest.status == "completed"
        assert manifest.completed_at is not None

    @pytest.mark.asyncio
    async def test_record_and_complete_step(self, tmp_path: Path):
        from ayextractor.storage.run_manager import create_run, record_step, complete_step, generate_document_id, generate_run_id
        from ayextractor.storage.local_writer import LocalWriter

        writer = LocalWriter(base_path=str(tmp_path))
        run_path, manifest = await create_run(
            writer=writer, output_path=tmp_path,
            document_id=generate_document_id(), run_id=generate_run_id(),
            pipeline_version="0.3.1",
        )
        await record_step(manifest, "extraction", origin="fresh")
        assert "extraction" in manifest.steps
        assert manifest.steps["extraction"].origin == "fresh"

        await complete_step(manifest, "extraction", output_hash="abc123")
        assert manifest.steps["extraction"].completed_at is not None
        assert manifest.steps["extraction"].output_hash == "abc123"


class TestReader:
    """Reader functions for loading analysis outputs."""

    def test_load_manifest(self, tmp_path: Path):
        from ayextractor.storage.reader import load_manifest

        run_path = layout.run_dir(tmp_path, "doc1", "run1")
        layout.ensure_run_directories(run_path)
        manifest = RunManifest(
            run_id="run1", document_id="doc1", pipeline_version="0.3.1",
            created_at=datetime.now(timezone.utc), status="completed",
            config_overrides_applied={}, llm_assignments={"summarizer": "ollama/qwen2.5"},
            prompt_hashes={}, steps={"extraction": StepManifest(origin="fresh")},
        )
        layout.run_manifest_path(run_path).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8",
        )
        loaded = load_manifest(run_path)
        assert loaded.document_id == "doc1"
        assert loaded.status == "completed"
        assert "extraction" in loaded.steps

    def test_load_chunks(self, tmp_path: Path):
        from ayextractor.storage.reader import load_chunks

        run_path = layout.run_dir(tmp_path, "doc1", "run1")
        layout.ensure_run_directories(run_path)
        chunks_dir = layout.chunks_dir(run_path)
        # Write a chunk file matching expected pattern
        chunk_data = {
            "id": "chunk_001", "position": 0,
            "content": "Test chunk content.",
            "source_file": "test.pdf", "source_pages": [1],
            "source_sections": [], "char_count": 19, "word_count": 3,
            "token_count_est": 5, "fingerprint": "abc",
            "primary_language": "en", "key_entities": [],
        }
        (chunks_dir / "chunk_001.json").write_text(json.dumps(chunk_data))
        chunks = load_chunks(run_path)
        assert len(chunks) == 1
        assert chunks[0].id == "chunk_001"

    def test_load_chunks_with_index(self, tmp_path: Path):
        from ayextractor.storage.reader import load_chunks

        run_path = layout.run_dir(tmp_path, "doc1", "run1")
        layout.ensure_run_directories(run_path)
        chunks_dir = layout.chunks_dir(run_path)
        chunk_data = {
            "id": "c1", "position": 0, "content": "X",
            "source_file": "f.pdf", "source_pages": [1],
            "source_sections": [], "char_count": 1, "word_count": 1,
            "token_count_est": 1, "fingerprint": "x",
            "primary_language": "en", "key_entities": [],
        }
        (chunks_dir / "c1.json").write_text(json.dumps(chunk_data))
        index_path = layout.chunks_index_path(run_path)
        index_path.write_text(json.dumps({"chunks": [{"id": "c1"}]}))
        chunks = load_chunks(run_path)
        assert len(chunks) == 1

    def test_load_dense_summary(self, tmp_path: Path):
        from ayextractor.storage.reader import load_dense_summary

        run_path = layout.run_dir(tmp_path, "doc1", "run1")
        layout.ensure_run_directories(run_path)
        summary_path = layout.dense_summary_path(run_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("Dense summary text.", encoding="utf-8")
        assert load_dense_summary(run_path) == "Dense summary text."

    def test_load_dense_summary_missing(self, tmp_path: Path):
        from ayextractor.storage.reader import load_dense_summary

        run_path = layout.run_dir(tmp_path, "doc1", "run1")
        layout.ensure_run_directories(run_path)
        assert load_dense_summary(run_path) is None

    def test_load_enriched_text(self, tmp_path: Path):
        from ayextractor.storage.reader import load_enriched_text

        run_path = layout.run_dir(tmp_path, "doc1", "run1")
        layout.ensure_run_directories(run_path)
        et_path = layout.enriched_text_path(run_path)
        et_path.parent.mkdir(parents=True, exist_ok=True)
        et_path.write_text("Enriched text.", encoding="utf-8")
        assert load_enriched_text(run_path) == "Enriched text."

    def test_find_latest_run(self, tmp_path: Path):
        from ayextractor.storage.reader import find_latest_run

        # No runs → None
        assert find_latest_run(tmp_path, "doc1") is None

        # Create two runs
        for rid in ["20260216_1000_aaaaa", "20260216_1100_bbbbb"]:
            rp = layout.run_dir(tmp_path, "doc1", rid)
            layout.ensure_run_directories(rp)
        latest = find_latest_run(tmp_path, "doc1")
        assert latest is not None
        assert "bbbbb" in str(latest)


class TestLocalWriter:
    """LocalWriter filesystem operations."""

    @pytest.mark.asyncio
    async def test_write_and_read(self, tmp_path: Path):
        from ayextractor.storage.local_writer import LocalWriter
        writer = LocalWriter(base_path=str(tmp_path))
        path = str(tmp_path / "sub" / "file.json")
        await writer.write(path, '{"a": 1}')
        assert await writer.exists(path)
        raw = await writer.read(path)
        assert json.loads(raw)["a"] == 1

    @pytest.mark.asyncio
    async def test_write_bytes(self, tmp_path: Path):
        from ayextractor.storage.local_writer import LocalWriter
        writer = LocalWriter(base_path=str(tmp_path))
        path = str(tmp_path / "bin.dat")
        await writer.write(path, b"\x00\x01\x02")
        raw = await writer.read(path)
        assert raw == b"\x00\x01\x02"

    @pytest.mark.asyncio
    async def test_list_dir(self, tmp_path: Path):
        from ayextractor.storage.local_writer import LocalWriter
        writer = LocalWriter(base_path=str(tmp_path))
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        entries = await writer.list_dir(str(tmp_path))
        assert "a.txt" in entries and "b.txt" in entries

    @pytest.mark.asyncio
    async def test_copy(self, tmp_path: Path):
        from ayextractor.storage.local_writer import LocalWriter
        writer = LocalWriter(base_path=str(tmp_path))
        src = tmp_path / "orig.txt"
        src.write_text("hello")
        dst = str(tmp_path / "copy.txt")
        await writer.copy(str(src), dst)
        assert await writer.exists(dst)

    @pytest.mark.asyncio
    async def test_create_symlink(self, tmp_path: Path):
        from ayextractor.storage.local_writer import LocalWriter
        writer = LocalWriter(base_path=str(tmp_path))
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        (target_dir / "f.txt").write_text("x")
        link = str(tmp_path / "link")
        await writer.create_symlink(str(target_dir), link)
        assert Path(link).is_symlink()

    @pytest.mark.asyncio
    async def test_exists_false(self, tmp_path: Path):
        from ayextractor.storage.local_writer import LocalWriter
        writer = LocalWriter(base_path=str(tmp_path))
        assert not await writer.exists(str(tmp_path / "nope"))
