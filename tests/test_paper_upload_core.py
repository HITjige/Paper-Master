from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pypdf import PdfWriter

from nanobot.agent.tools.paper import PaperIngestTool
from nanobot.api.server import (
    _FileSizeExceeded,
    _ingest_uploaded_paper,
    _PaperIngestJobManager,
    _PaperUploadError,
    _read_json_body_limited,
    _stream_paper_upload,
)


class _Part:
    filename = "paper.pdf"

    def __init__(self, data: bytes):
        self.data = data
        self.position = 0

    async def read_chunk(self, size: int) -> bytes:
        chunk = self.data[self.position:self.position + size]
        self.position += len(chunk)
        return chunk


def _pdf_bytes() -> bytes:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(output)
    return output.getvalue()


def _fake_ingest_dependencies(tmp_path: Path):
    kb = MagicMock()
    kb.base_dir = tmp_path / "kb"
    kb.chunks_file = kb.base_dir / "chunks.jsonl"
    kb.load_docs_meta.return_value = {}
    kb._read_jsonl.return_value = []
    agent = MagicMock()
    agent.kb_ingest_local = AsyncMock(return_value={
        "status": "ok",
        "chunk_count": 2,
        "storage_backend": "jsonl",
    })
    return agent, kb


def test_stream_upload_is_content_addressed_and_validated(tmp_path: Path):
    raw = _pdf_bytes()
    uploaded = asyncio.run(_stream_paper_upload(_Part(raw), tmp_path))

    assert uploaded.sha256 == hashlib.sha256(raw).hexdigest()
    assert uploaded.path.name == f"{uploaded.sha256}.pdf"
    assert uploaded.page_count == 1


def test_stream_upload_rejects_fake_pdf(tmp_path: Path):
    with pytest.raises(_PaperUploadError) as error:
        asyncio.run(_stream_paper_upload(_Part(b"not a pdf"), tmp_path))

    assert error.value.code == "PDF_SIGNATURE_INVALID"
    assert not list(tmp_path.glob("*.part"))


def test_uploaded_paper_uses_hash_identity_and_preserves_unknown_year(tmp_path: Path):
    raw = _pdf_bytes()
    uploaded = asyncio.run(_stream_paper_upload(_Part(raw), tmp_path))
    agent, kb = _fake_ingest_dependencies(tmp_path)

    result = asyncio.run(_ingest_uploaded_paper(agent, kb, uploaded, {}))

    assert result["paper_id"] == f"upload:{uploaded.sha256}"
    assert result["status"] == "ok"
    doc = agent.kb_ingest_local.await_args.kwargs["doc"]
    assert doc["year"] is None
    assert doc["content_sha256"] == uploaded.sha256


def test_background_ingest_job_reaches_completed_state(tmp_path: Path):
    async def _run():
        raw = _pdf_bytes()
        uploaded = await _stream_paper_upload(_Part(raw), tmp_path)
        agent, kb = _fake_ingest_dependencies(tmp_path)
        manager = _PaperIngestJobManager(concurrency=1)
        job = manager.submit(
            agent_loop=agent,
            kb=kb,
            uploads=[uploaded],
            ingest_locks={},
            validation_results=[],
        )
        for _ in range(20):
            current = manager.get(job["job_id"])
            if current and current["status"] == "completed":
                break
            await asyncio.sleep(0)
        await manager.close()
        return current

    completed = asyncio.run(_run())
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["succeeded"] == 1


def test_json_body_limit_is_enforced_for_chunked_requests():
    class _Content:
        async def iter_chunked(self, _size: int):
            yield b'{"value":"'
            yield b"too-large"
            yield b'"}'

    request = MagicMock()
    request.content = _Content()
    with pytest.raises(_FileSizeExceeded, match="JSON request exceeds"):
        asyncio.run(_read_json_body_limited(request, max_bytes=10))


def test_shared_local_pdf_pipeline_preserves_page_provenance(tmp_path: Path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "paper.pdf"
    document = fitz.open()
    page_one = document.new_page()
    page_one.insert_text(
        (72, 72),
        "# Test Paper\n# Introduction\n" + "Grounded introduction text. " * 20,
    )
    page_two = document.new_page()
    page_two.insert_text(
        (72, 72),
        "# Method\n" + "Grounded method and experiment details. " * 20,
    )
    document.save(pdf_path)
    document.close()

    kb = MagicMock()
    kb.base_dir = tmp_path / "kb"
    kb.config = SimpleNamespace(
        max_chunk_chars=1000,
        min_chunk_chars=30,
        num_hypothetical_questions=3,
        metadata_concurrency=2,
    )
    kb.upsert_semantic_chunks = AsyncMock(return_value={
        "paper_id": "upload:test",
        "chunk_count": 2,
        "question_count": 6,
        "storage_backend": "jsonl",
        "degraded": False,
    })
    async def _direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("nanobot.agent.tools.paper.asyncio.to_thread", _direct_to_thread)
    tool = PaperIngestTool(workspace=tmp_path, kb=kb)
    paper = {
        "paper_id": "upload:test",
        "title": "paper",
        "page_count": 2,
        "content_sha256": "test",
    }

    result = asyncio.run(tool.ingest_local_pdf(paper, pdf_path, summarize=False))

    assert result["status"] == "ok"
    assert result["parser_name"] == "pypdf"
    assert result["page_coverage_ratio"] == 1.0
    chunks = kb.upsert_semantic_chunks.await_args.kwargs["semantic_chunks"]
    assert any(chunk.get("page_start") == 1 for chunk in chunks)
    assert any(chunk.get("page_end") == 2 for chunk in chunks)
