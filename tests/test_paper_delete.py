"""Focused tests for complete paper deletion across KB stores."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nanobot.agent.paper_kb import PaperKbConfig, PaperKnowledgeBase


@pytest.fixture(autouse=True)
def _run_thread_calls_inline(monkeypatch):
    """Avoid the test environment's executor-shutdown hang around asyncio.run."""
    async def inline(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("nanobot.agent.paper_kb.asyncio.to_thread", inline)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _jsonl_only_kb(tmp_path: Path) -> PaperKnowledgeBase:
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(embedding_model="", embedding_fallback="hash"),
    )
    kb._chroma_client = None
    kb._chunk_collection = None
    kb._summary_collection = None
    kb._question_collection = None
    return kb


class _FakeCollection:
    def __init__(self, name: str, rows: dict[str, tuple[str, dict, list[float]]]):
        self.name = name
        self.rows = dict(rows)

    def get(self, *, where=None, include=None):
        selected = [
            (row_id, row)
            for row_id, row in self.rows.items()
            if not where or row[1].get("paper_id") == where.get("paper_id")
        ]
        return {
            "ids": [row_id for row_id, _row in selected],
            "documents": [row[0] for _row_id, row in selected],
            "metadatas": [row[1] for _row_id, row in selected],
            "embeddings": [row[2] for _row_id, row in selected],
        }

    def delete(self, *, ids):
        for row_id in ids:
            self.rows.pop(row_id, None)

    def upsert(self, *, ids, documents, metadatas, embeddings):
        for index, row_id in enumerate(ids):
            self.rows[row_id] = (
                documents[index],
                metadatas[index],
                embeddings[index],
            )


def test_delete_paper_removes_jsonl_fts_assets_and_managed_files(tmp_path) -> None:
    kb = _jsonl_only_kb(tmp_path)
    digest = "a" * 64
    paper_id = f"upload:{digest}"
    upload_dir = kb.base_dir / "uploads"
    pdf_path = upload_dir / f"{digest}.pdf"
    md_path = upload_dir / f"{digest}.md"
    upload_dir.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF test")
    md_path.write_text("target markdown", encoding="utf-8")
    unrelated_md = upload_dir / "unrelated.md"
    unrelated_md.write_text("keep", encoding="utf-8")
    external_md = tmp_path / "outside-managed-storage.md"
    external_md.write_text("never delete", encoding="utf-8")

    target_doc = {
        "paper_id": paper_id,
        "title": "Reinforcement Learning Paper",
        "url": str(external_md),
        "content_sha256": digest,
    }
    other_doc = {
        "paper_id": "other-paper",
        "title": "Vision Paper",
        "url": str(unrelated_md),
    }
    target_chunk = {
        "chunk_id": f"{paper_id}:0",
        "paper_id": paper_id,
        "text": "reinforcement policy optimization",
    }
    other_chunk = {
        "chunk_id": "other-paper:0",
        "paper_id": "other-paper",
        "text": "visual recognition transformer",
    }
    _jsonl(kb.docs_file, [target_doc, other_doc])
    _jsonl(kb.chunks_file, [target_chunk, other_chunk])
    _jsonl(
        kb.base_dir / "figures.jsonl",
        [
            {"key": f"{paper_id}_figure_1", "value": {"caption": "remove"}},
            {"key": "other-paper_figure_1", "value": {"caption": "keep"}},
        ],
    )

    try:
        asyncio.run(kb._replace_lexical_paper_strict(
            doc_row=target_doc,
            chunk_rows=[target_chunk],
        ))
        asyncio.run(kb._replace_lexical_paper_strict(
            doc_row=other_doc,
            chunk_rows=[other_chunk],
        ))

        result = asyncio.run(kb.delete_paper(paper_id))

        assert result["status"] == "deleted"
        assert result["deleted"] is True
        assert result["deleted_chunk_count"] == 1
        assert result["deleted_asset_count"] == 1
        assert not pdf_path.exists()
        assert not md_path.exists()
        assert unrelated_md.exists()
        assert external_md.exists()
        assert [row["paper_id"] for row in _jsonl_rows(kb.docs_file)] == ["other-paper"]
        assert [row["paper_id"] for row in _jsonl_rows(kb.chunks_file)] == ["other-paper"]
        assert [row["key"] for row in _jsonl_rows(kb.base_dir / "figures.jsonl")] == [
            "other-paper_figure_1"
        ]
        assert kb._lexical_index is not None
        assert kb._lexical_index.search("reinforcement policy") == []
        assert kb._lexical_index.search("visual transformer")
    finally:
        if kb._lexical_index is not None:
            kb._lexical_index.close()


def test_delete_paper_returns_not_found_without_mutation(tmp_path) -> None:
    kb = _jsonl_only_kb(tmp_path)
    _jsonl(kb.docs_file, [{"paper_id": "kept", "title": "Kept"}])
    try:
        result = asyncio.run(kb.delete_paper("missing"))
        assert result == {
            "status": "not_found",
            "deleted": False,
            "paper_id": "missing",
        }
        assert _jsonl_rows(kb.docs_file) == [{"paper_id": "kept", "title": "Kept"}]
    finally:
        if kb._lexical_index is not None:
            kb._lexical_index.close()


def test_delete_paper_removes_all_three_vector_representations(tmp_path) -> None:
    kb = _jsonl_only_kb(tmp_path)
    paper_id = "2511.14460v2"
    doc = {"paper_id": paper_id, "title": "Agent-R1"}
    chunk = {"chunk_id": f"{paper_id}:0", "paper_id": paper_id, "text": "body"}
    _jsonl(kb.docs_file, [doc])
    _jsonl(kb.chunks_file, [chunk])
    target = ("target", {"paper_id": paper_id}, [1.0, 0.0])
    other = ("other", {"paper_id": "kept"}, [0.0, 1.0])
    collections = [
        _FakeCollection("paper_chunks", {"target-chunk": target, "other-chunk": other}),
        _FakeCollection("paper_summaries", {"target-summary": target}),
        _FakeCollection("paper_questions", {"target-question": target}),
    ]
    kb._chroma_client = object()
    kb._chunk_collection, kb._summary_collection, kb._question_collection = collections

    try:
        result = asyncio.run(kb.delete_paper(paper_id, delete_files=False))
        assert result["deleted_vector_count"] == 3
        assert set(result["deleted_vectors"]) == {
            "paper_chunks",
            "paper_summaries",
            "paper_questions",
        }
        assert set(collections[0].rows) == {"other-chunk"}
        assert collections[1].rows == {}
        assert collections[2].rows == {}
    finally:
        if kb._lexical_index is not None:
            kb._lexical_index.close()


def test_delete_paper_restores_jsonl_and_assets_when_fts_commit_fails(
    tmp_path,
    monkeypatch,
) -> None:
    kb = _jsonl_only_kb(tmp_path)
    paper_id = "rollback-paper"
    doc = {"paper_id": paper_id, "title": "Rollback"}
    chunk = {"chunk_id": f"{paper_id}:0", "paper_id": paper_id, "text": "evidence"}
    asset = {"key": f"{paper_id}_table_1", "value": {"content": "result"}}
    _jsonl(kb.docs_file, [doc])
    _jsonl(kb.chunks_file, [chunk])
    _jsonl(kb.base_dir / "figures.jsonl", [asset])

    original = kb._replace_lexical_paper_strict
    calls = 0

    async def fail_once(*, doc_row, chunk_rows):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated FTS failure")
        await original(doc_row=doc_row, chunk_rows=chunk_rows)

    monkeypatch.setattr(kb, "_replace_lexical_paper_strict", fail_once)
    try:
        with pytest.raises(RuntimeError, match="failed to delete paper"):
            asyncio.run(kb.delete_paper(paper_id))
        assert _jsonl_rows(kb.docs_file) == [doc]
        assert _jsonl_rows(kb.chunks_file) == [chunk]
        assert _jsonl_rows(kb.base_dir / "figures.jsonl") == [asset]
    finally:
        if kb._lexical_index is not None:
            kb._lexical_index.close()
