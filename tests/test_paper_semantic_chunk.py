"""Tests for paper semantic chunking and hypothetical question retrieval."""

import asyncio
import json
import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.paper_kb import (
    PaperKbConfig,
    PaperKnowledgeBase,
    _hash_embedding,
    _select_diverse,
    _SQLiteBM25Index,
    _weighted_rrf_fuse,
    tokenize_text,
)
from nanobot.agent.tools.paper import (
    _decode_chunk_metadata_json,
    _generate_chunk_metadata,
    _remove_noisy_blocks,
    _split_markdown_semantic,
)

# Sample markdown text for testing
SAMPLE_MARKDOWN = """
# Abstract

This paper presents a novel approach to semantic chunking of scientific documents. We propose using Markdown headers to identify natural boundaries in the document structure, enabling more precise retrieval of relevant content.

## Introduction

The problem of document chunking is fundamental to many information retrieval systems. Traditional approaches use fixed-size chunks or simple paragraph boundaries, which often fail to capture the semantic structure of scientific papers.

### Motivation

Scientific papers have a well-defined structure with sections like Introduction, Methods, Results, and Conclusion. Each section serves a specific purpose and contains related information. By preserving this structure during chunking, we can improve retrieval accuracy.

### Related Work

Previous work on document chunking includes various approaches such as sliding windows, sentence boundaries, and topic modeling. However, these methods do not explicitly leverage the document structure.

## Method

Our approach uses Markdown header levels to split documents. We process each header level (#, ##, ###) as a potential boundary.

### Algorithm

1. Parse Markdown headers
2. Create chunks at each header boundary
3. Preserve header context in each chunk
4. Generate embeddings for retrieval

## Results

We evaluated our method on a corpus of 1000 scientific papers. The results show significant improvement over baseline methods.

| Method | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Fixed-size | 0.65 | 0.72 | 0.68 |
| Our method | 0.85 | 0.88 | 0.86 |

## Conclusion

We presented a novel Markdown-based chunking method that improves retrieval accuracy for scientific documents. Future work will explore dynamic chunk sizing and multi-document chunking.

## References

[1] Smith et al., 2023
[2] Johnson et al., 2024

## Acknowledgements

We thank the anonymous reviewers for their helpful comments.
"""


def test_select_diverse_respects_k_and_per_paper_limit():
    scored = [
        {"chunk_id": "p1:0", "paper_id": "p1", "score": 0.9, "embedding": [1.0, 0.0]},
        {"chunk_id": "p1:1", "paper_id": "p1", "score": 0.8, "embedding": [0.9, 0.1]},
        {"chunk_id": "p2:0", "paper_id": "p2", "score": 0.7, "embedding": [0.0, 1.0]},
    ]
    selected = _select_diverse(scored, k=3, per_paper_limit=1)
    assert len(selected) == 2
    assert {item["paper_id"] for item in selected} == {"p1", "p2"}


def test_select_diverse_handles_candidate_count_below_k():
    selected = _select_diverse(
        [{"chunk_id": "p1:0", "paper_id": "p1", "score": 1.0, "embedding": [1.0]}],
        k=5,
        per_paper_limit=2,
    )
    assert [item["chunk_id"] for item in selected] == ["p1:0"]


def test_mixed_tokenizer_supports_chinese_and_english():
    tokens = tokenize_text("多智能体论文检索 Agentic RAG")
    assert "多智" in tokens
    assert "论文" in tokens
    assert "检索" in tokens
    assert "agentic" in tokens
    assert "rag" in tokens


def test_chinese_tokens_work_in_hash_embedding():
    assert any(value != 0.0 for value in _hash_embedding("中文论文检索"))


def test_sqlite_fts5_indexes_one_weighted_row_per_parent_chunk(tmp_path: Path):
    index = _SQLiteBM25Index(tmp_path / "lexical.db")
    try:
        index.replace_paper(
            "paper-1",
            [{
                "chunk_id": "paper-1:0",
                "paper_id": "paper-1",
                "title": "多智能体论文检索",
                "keywords": ["RAG", "BM25"],
                "summary": "结合稀疏与稠密向量的混合检索",
                "questions": ["如何检索中文论文？"],
                "body": "系统使用查询重写和重排。",
            }],
            source_mtime_ns=1,
        )
        index.replace_paper(
            "paper-2",
            [{
                "chunk_id": "paper-2:0",
                "paper_id": "paper-2",
                "title": "图像分类",
                "body": "卷积神经网络视觉识别。",
            }],
            source_mtime_ns=2,
        )

        assert index.count() == 2
        assert index.search("中文论文检索", top_n=1)[0][0] == "paper-1:0"
        assert index.search("图像分类", top_n=1, paper_id="paper-2")[0][0] == "paper-2:0"
        assert index.search("中文论文", top_n=5, paper_id="paper-2") == []

        # Replacing a paper removes stale parent chunks instead of appending duplicates.
        index.replace_paper(
            "paper-1",
            [{
                "chunk_id": "paper-1:1",
                "paper_id": "paper-1",
                "title": "新的检索论文",
                "body": "持久化倒排索引。",
            }],
            source_mtime_ns=3,
        )
        assert index.count() == 2
        assert index.search("多智能体", top_n=5) == []
        assert index.search("持久化倒排索引", top_n=1)[0][0] == "paper-1:1"
    finally:
        index.close()


def test_weighted_rrf_normalizes_each_retrieval_family():
    fused = _weighted_rrf_fuse(
        [
            ([[('dense', 0.9)], [('dense', 0.8)]], 0.5),
            ([[('sparse', 8.0)]], 0.5),
        ],
        k=60,
    )

    assert fused["dense"] == pytest.approx(fused["sparse"])


def test_paper_kb_rebuilds_persistent_fts_from_jsonl(tmp_path: Path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True)
    (kb_dir / "documents.jsonl").write_text(
        json.dumps({"paper_id": "p1", "title": "中文论文检索"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (kb_dir / "chunks.jsonl").write_text(
        json.dumps({
            "chunk_id": "p1:0",
            "paper_id": "p1",
            "text": "持久化 SQLite FTS5 倒排索引",
            "summary": "中文混合检索",
            "hypothetical_questions": ["如何检索中文论文？"],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, enable_hypothetical_retrieval=False),
    )
    try:
        status = kb.get_lexical_status()
        assert status["backend"] == "sqlite_fts5"
        assert status["document_count"] == 1
        assert kb._lexical_index.search("中文论文", top_n=1)[0][0] == "p1:0"
    finally:
        kb._lexical_index.close()


def test_vector_stamp_does_not_resubmit_immutable_hnsw_metadata(
    tmp_path: Path,
) -> None:
    pytest.importorskip("chromadb")
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(
            enabled=True,
            embedding_model="",
            embedding_fallback="hash",
            chroma_persist_dir=str(tmp_path / "chroma"),
        ),
    )
    try:
        kb._stamp_vector_collections(
            [kb._chunk_collection, kb._summary_collection, kb._question_collection],
            dimension=256,
        )
        for collection in (
            kb._chunk_collection,
            kb._summary_collection,
            kb._question_collection,
        ):
            assert collection.metadata["nanobot:embedding_dimension"] == 256
    finally:
        kb._lexical_index.close()


def test_stamp_failure_precedes_paper_data_commit(tmp_path: Path, monkeypatch) -> None:
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, min_chunk_chars=5),
    )
    kb._chroma_client = object()
    kb._chunk_collection = MagicMock()
    kb._summary_collection = MagicMock()
    kb._question_collection = MagicMock()
    kb.embed_texts = AsyncMock(side_effect=lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(
        kb,
        "_stamp_vector_collections",
        MagicMock(side_effect=ValueError("immutable metadata")),
    )

    with pytest.raises(ValueError, match="immutable metadata"):
        asyncio.run(kb.upsert_semantic_chunks(
            {"paper_id": "p1", "title": "Paper"},
            [{"text": "long enough chunk", "section": "Method"}],
            [{"summary": "summary", "hypothetical_questions": ["question"]}],
        ))

    assert kb._read_jsonl(kb.docs_file) == []
    assert kb._read_jsonl(kb.chunks_file) == []
    kb._lexical_index.close()


def test_vector_index_detects_mixed_dimensions_and_rebuilds_atomically(
    tmp_path: Path,
):
    chromadb = pytest.importorskip("chromadb")
    chroma_path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_path))
    fixtures = {
        "paper_chunks": (3, "p1:0", "parent chunk"),
        "paper_summaries": (2, "p1:0:summary", "chunk summary"),
        "paper_questions": (2, "p1:0:q0", "what is the method?"),
    }
    for name, (dimension, record_id, document) in fixtures.items():
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            ids=[record_id],
            embeddings=[[1.0] * dimension],
            documents=[document],
            metadatas=[{"paper_id": "p1", "chunk_id": "p1:0"}],
        )

    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(
            enabled=True,
            embedding_model="",
            embedding_fallback="hash",
            embedding_batch_size=2,
            chroma_persist_dir=str(chroma_path),
        ),
    )
    before = kb.get_vector_index_status(refresh=True)
    assert before["compatible"] is False
    assert before["reindex_required"] is True
    assert before["reason"] == "mixed_collection_dimensions"
    assert before["collection_dimensions"] == {
        "paper_chunks": 3,
        "paper_summaries": 2,
        "paper_questions": 2,
    }

    result = asyncio.run(kb.rebuild_vector_index(keep_backup=False))

    assert result["status"] == "ok"
    assert result["dimension"] == 256
    assert result["source_counts"] == {
        "paper_chunks": 1,
        "paper_summaries": 1,
        "paper_questions": 1,
    }
    after = kb.get_vector_index_status(refresh=True)
    assert after["compatible"] is True
    assert after["verified"] is True
    assert after["reindex_required"] is False
    assert set(after["collection_dimensions"].values()) == {256}
    assert {
        collection.name for collection in kb._chroma_client.list_collections()
    } == set(fixtures)


def test_incompatible_vector_index_skips_dense_and_keeps_sparse_results(
    tmp_path: Path,
    monkeypatch,
):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True)
    (kb_dir / "documents.jsonl").write_text(
        json.dumps({"paper_id": "p1", "title": "EEG image generation"}) + "\n",
        encoding="utf-8",
    )
    (kb_dir / "chunks.jsonl").write_text(
        json.dumps({
            "chunk_id": "p1:0",
            "paper_id": "p1",
            "text": "EEG signals can condition an image diffusion model.",
        }) + "\n",
        encoding="utf-8",
    )
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(
            enabled=True,
            embedding_model="",
            embedding_fallback="hash",
            enable_hypothetical_retrieval=False,
        ),
    )

    class _MustNotQuery:
        def query(self, **kwargs):
            raise AssertionError("incompatible dense collection was queried")

    class _ParentCollection(_MustNotQuery):
        def get(self, *, ids, include):
            return {
                "ids": ids,
                "documents": [
                    "EEG signals can condition an image diffusion model."
                    for _ in ids
                ],
                "metadatas": [{
                    "paper_id": "p1",
                    "paper_title": "EEG image generation",
                    "keywords": "[]",
                    "entities": "[]",
                    "claims": "[]",
                } for _ in ids],
                "embeddings": [[1.0] + [0.0] * 255 for _ in ids],
            }

    kb._chroma_client = object()
    kb._chunk_collection = _ParentCollection()
    kb._summary_collection = _MustNotQuery()
    kb._question_collection = _MustNotQuery()
    kb._vector_index_status = {
        **kb._vector_index_status,
        "compatible": False,
        "reindex_required": True,
        "reason": "embedding_dimension_mismatch",
    }
    kb._ensure_vector_index_compatible = AsyncMock(return_value=False)

    async def _direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("nanobot.agent.paper_kb.asyncio.to_thread", _direct_to_thread)
    results = asyncio.run(kb._retrieve_dense_hybrid(
        query="EEG image diffusion",
        top_k=2,
        per_paper_limit=2,
        search_mode="hybrid",
        where_filter=None,
        use_hybrid=True,
    ))

    assert [row["chunk_id"] for row in results] == ["p1:0"]
    assert results[0]["bm25_score"] is not None
    assert results[0]["dense_score"] is None
    assert kb.get_vector_index_status()["reindex_required"] is True


def test_empty_lexical_index_can_be_rebuilt_from_chroma(tmp_path: Path) -> None:
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(
            embedding_model="",
            embedding_fallback="hash",
            min_chunk_chars=20,
        ),
    )
    text = (
        "ATM evaluates EEG decoding performance with within-subject and "
        "cross-subject accuracy experiments. " * 8
    )
    assert kb._chunk_collection is not None
    assert kb._summary_collection is not None
    assert kb._question_collection is not None
    common_metadata = {
        "paper_id": "atm-1",
        "paper_title": "ATM",
        "paper_source": "upload",
        "paper_year": "2024",
        "section": "Experiments",
        "heading_path": "Experiments",
    }
    kb._chunk_collection.upsert(
        ids=["atm-1:0"],
        embeddings=[_hash_embedding(text)],
        documents=[text],
        metadatas=[{**common_metadata, "keywords": json.dumps(["ATM", "EEG", "accuracy"])}],
    )
    summary = "ATM performance on EEG decoding benchmarks."
    kb._summary_collection.upsert(
        ids=["atm-1:0:summary"],
        embeddings=[_hash_embedding(summary)],
        documents=[summary],
        metadatas=[{**common_metadata, "chunk_id": "atm-1:0", "type": "summary"}],
    )
    question = "What is the cross-subject accuracy of ATM?"
    kb._question_collection.upsert(
        ids=["atm-1:0:q0"],
        embeddings=[_hash_embedding(question)],
        documents=[question],
        metadatas=[{
            **common_metadata,
            "chunk_id": "atm-1:0",
            "type": "hypothetical_question",
        }],
    )
    assert kb._lexical_index is not None
    assert kb._lexical_index.count() == 0

    status = kb.rebuild_lexical_index_from_chroma()
    matches = kb._lexical_index.search("cross-subject accuracy", top_n=5)

    assert status["document_count"] == 1
    assert matches and matches[0][0] == "atm-1:0"
    kb._lexical_index.close()


def test_hybrid_retrieval_exposes_separate_dense_sparse_and_rrf_scores(
    tmp_path: Path,
    monkeypatch,
):
    class _DenseCollection:
        def query(self, **kwargs):
            return {
                "ids": [["p2:0:summary"]],
                "metadatas": [[{
                    "chunk_id": "p2:0",
                    "paper_id": "p2",
                    "section": "Method",
                }]],
                "documents": [["dense-only summary"]],
                "distances": [[0.1]],
            }

    class _EmptyCollection:
        def query(self, **kwargs):
            return {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}

    class _ParentCollection:
        def get(self, *, ids, include):
            documents = {
                "p1:0": "中文论文检索使用 BM25。",
                "p2:0": "Dense semantic retrieval.",
            }
            return {
                "ids": ids,
                "documents": [documents[item] for item in ids],
                "metadatas": [{
                    "paper_id": item.split(":")[0],
                    "paper_title": item,
                    "keywords": "[]",
                    "entities": "[]",
                    "claims": "[]",
                } for item in ids],
                "embeddings": [[1.0, 0.0] for _ in ids],
            }

    async def _direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("nanobot.agent.paper_kb.asyncio.to_thread", _direct_to_thread)
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, enable_hypothetical_retrieval=False),
    )
    try:
        kb._lexical_index.replace_paper(
            "p1",
            [{
                "chunk_id": "p1:0",
                "paper_id": "p1",
                "title": "中文论文检索",
                "body": "BM25 稀疏倒排索引",
            }],
            source_mtime_ns=1,
        )
        kb._summary_collection = _DenseCollection()
        kb._question_collection = _EmptyCollection()
        kb._chunk_collection = _ParentCollection()

        results = asyncio.run(kb._retrieve_dense_hybrid(
            query="中文论文检索",
            top_k=2,
            per_paper_limit=2,
            search_mode="hybrid",
            where_filter=None,
            use_hybrid=True,
        ))

        assert {row["chunk_id"] for row in results} == {"p1:0", "p2:0"}
        assert all(row["score_type"] == "weighted_rrf" for row in results)
        sparse = next(row for row in results if row["chunk_id"] == "p1:0")
        dense = next(row for row in results if row["chunk_id"] == "p2:0")
        assert sparse["bm25_score"] is not None
        assert sparse["dense_score"] is None
        assert dense["dense_score"] == pytest.approx(0.9)
        assert dense["bm25_score"] is None
    finally:
        kb._lexical_index.close()


@pytest.mark.asyncio
async def test_default_embedding_fallback_is_explicit(tmp_path: Path):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    status = kb.get_embedding_status()

    assert status["backend"] == "hash_lexical"
    assert status["model"] == "hash-lexical-256"
    assert status["configured_model"] == "text-embedding-3-small"
    assert status["degraded"] is True
    assert status["reason"] == "no_semantic_embedding_backend_configured"
    assert status["dimension"] == 256
    assert any(await kb.embed_text("中文语义检索"))


@pytest.mark.asyncio
async def test_embedding_error_mode_does_not_silently_hash(tmp_path: Path):
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, embedding_fallback="error"),
    )

    with pytest.raises(RuntimeError, match="No embedding backend is available"):
        await kb.embed_text("must fail")


@pytest.mark.asyncio
async def test_embedding_api_batches_and_preserves_empty_positions(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    class _FakeResponse:
        def __init__(self, texts: list[str]):
            self._texts = texts

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"index": index, "embedding": [float(len(text)), float(index + 1)]}
                    for index, text in enumerate(self._texts)
                ]
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, *, headers: dict, json: dict):
            texts = list(json["input"])
            calls.append(texts)
            return _FakeResponse(texts)

    monkeypatch.setattr("nanobot.agent.paper_kb.httpx.AsyncClient", _FakeClient)
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(
            enabled=True,
            embedding_api_key="test-key",
            embedding_batch_size=2,
        ),
    )

    vectors = await kb.embed_texts(["one", "two", "", "three"])

    assert calls == [["one", "two"], ["three"]]
    assert vectors[0] == [3.0, 1.0]
    assert vectors[1] == [3.0, 2.0]
    assert vectors[2] == []
    assert vectors[3] == [5.0, 1.0]
    assert kb.get_embedding_status()["dimension"] == 2


@pytest.mark.asyncio
async def test_embedding_rejects_dimension_changes_between_batches(tmp_path: Path, monkeypatch):
    dimensions = iter((2, 3))

    async def _fake_embed_batch(texts: list[str]) -> list[list[float]]:
        dimension = next(dimensions)
        return [[1.0] * dimension for _ in texts]

    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(
            enabled=True,
            embedding_api_key="test-key",
            embedding_batch_size=1,
        ),
    )
    monkeypatch.setattr(kb, "_embed_api_batch", _fake_embed_batch)

    with pytest.raises(RuntimeError, match="Embedding dimension changed from 2 to 3"):
        await kb.embed_texts(["first", "second"])

    status = kb.get_embedding_status()
    assert status["degraded"] is True
    assert "dimension changed" in status["reason"]


@pytest.mark.asyncio
async def test_embedding_api_failure_never_switches_to_hash(tmp_path: Path, monkeypatch):
    async def _failing_embed_batch(texts: list[str]) -> list[list[float]]:
        raise TimeoutError("embedding service timed out")

    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, embedding_api_key="test-key"),
    )
    monkeypatch.setattr(kb, "_embed_api_batch", _failing_embed_batch)

    with pytest.raises(RuntimeError, match="embedding service timed out"):
        await kb.embed_text("query")

    status = kb.get_embedding_status()
    assert status["backend"] == "openai_compatible_api"
    assert status["degraded"] is True
    assert status["dimension"] is None


@pytest.mark.asyncio
async def test_multi_query_retrieval_batches_query_embeddings(tmp_path: Path):
    class _EmptyCollection:
        def query(self, **kwargs):
            return {
                "ids": [[]],
                "metadatas": [[]],
                "documents": [[]],
                "distances": [[]],
            }

    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, enable_hypothetical_retrieval=False),
    )
    kb._summary_collection = _EmptyCollection()
    kb._question_collection = _EmptyCollection()
    kb.embed_texts = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])

    results = await kb._retrieve_dense_hybrid(
        queries=["中文查询", "English query"],
        top_k=2,
        per_paper_limit=1,
        search_mode="hybrid",
        where_filter=None,
        use_hybrid=False,
    )

    assert results == []
    kb.embed_texts.assert_awaited_once_with(["中文查询", "English query"])


def test_parent_chunk_dense_view_batches_multi_query_search(tmp_path: Path, monkeypatch):
    class _ParentCollection:
        def __init__(self):
            self.query_calls = []

        def query(self, **kwargs):
            self.query_calls.append(kwargs)
            return {
                "ids": [["p1:0"], ["p2:0"]],
                "metadatas": [[{"paper_id": "p1"}], [{"paper_id": "p2"}]],
                "documents": [["parent one"], ["parent two"]],
                "distances": [[0.1], [0.2]],
            }

        def get(self, *, ids, include):
            return {
                "ids": ids,
                "documents": ["hydrated " + chunk_id for chunk_id in ids],
                "metadatas": [{
                    "paper_id": chunk_id.split(":")[0],
                    "paper_title": chunk_id,
                    "keywords": "[]",
                    "entities": "[]",
                    "claims": "[]",
                } for chunk_id in ids],
                "embeddings": [[1.0, 0.0] for _ in ids],
            }

    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, enable_hypothetical_retrieval=False),
    )
    parent_collection = _ParentCollection()
    kb._chunk_collection = parent_collection
    kb.embed_texts = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])

    async def _direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("nanobot.agent.paper_kb.asyncio.to_thread", _direct_to_thread)

    try:
        results = asyncio.run(kb._retrieve_dense_hybrid(
            queries=["first query", "second query"],
            top_k=2,
            per_paper_limit=1,
            search_mode="chunks_only",
            where_filter=None,
            use_hybrid=False,
        ))
    finally:
        kb._lexical_index.close()

    assert {row["chunk_id"] for row in results} == {"p1:0", "p2:0"}
    assert all(row["matched_by"] == "chunk" for row in results)
    assert len(parent_collection.query_calls) == 1
    assert parent_collection.query_calls[0]["query_embeddings"] == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


def test_cross_encoder_pairs_are_batched_and_logits_are_normalized(
    tmp_path: Path,
    monkeypatch,
):
    class _FakeCrossEncoder:
        def __init__(self):
            self.calls = []

        def predict(self, pairs):
            self.calls.append(pairs)
            return [-2.0, 0.2, 2.0]

    async def _direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("nanobot.agent.paper_kb.asyncio.to_thread", _direct_to_thread)
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(
            enabled=True,
            enable_hypothetical_retrieval=False,
            rerank_model="fake",
        ),
    )
    kb.rerank_model = _FakeCrossEncoder()
    try:
        scores = asyncio.run(kb.rerank_pairs([
            ("query", "negative"),
            ("query", "weak positive logit"),
            ("query", "positive"),
        ]))
    finally:
        kb._lexical_index.close()

    assert scores[0] == pytest.approx(1.0 / (1.0 + math.exp(2.0)))
    assert scores[1] == pytest.approx(1.0 / (1.0 + math.exp(-0.2)))
    assert scores[2] == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert len(kb.rerank_model.calls) == 1


def test_relevance_filter_uses_multi_query_max_and_preserves_fusion_score(
    tmp_path: Path,
):
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(
            enabled=True,
            enable_hypothetical_retrieval=False,
            rerank_model="fake",
            retrieval_min_relevance_score=0.5,
        ),
    )
    kb.rerank_pairs = AsyncMock(return_value=[0.1, 0.9, 0.2, 0.3])
    results = asyncio.run(kb.rerank_and_filter_retrieval_results(
        [
            {
                "chunk_id": "p1:0",
                "paper_id": "p1",
                "paper_title": "Relevant",
                "text": "reinforcement learning evidence",
                "score": 1.0,
                "score_type": "weighted_rrf",
            },
            {
                "chunk_id": "p2:0",
                "paper_id": "p2",
                "paper_title": "Irrelevant",
                "text": "unrelated signal processing",
                "score": 0.95,
                "score_type": "weighted_rrf",
            },
        ],
        queries=["强化学习", "reinforcement learning"],
        top_k=5,
        per_paper_limit=3,
    ))

    assert [item["chunk_id"] for item in results] == ["p1:0"]
    assert results[0]["score"] == pytest.approx(0.9)
    assert results[0]["relevance_score"] == pytest.approx(0.9)
    assert results[0]["fusion_score"] == pytest.approx(1.0)
    assert results[0]["score_type"] == "cross_encoder_relevance"
    assert len(kb.rerank_pairs.await_args.args[0]) == 4
    kb._lexical_index.close()


def test_paper_discovery_applies_paper_level_relevance_gate(tmp_path: Path):
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(
            enabled=True,
            enable_hypothetical_retrieval=False,
            rerank_model="fake",
            retrieval_min_relevance_score=0.5,
        ),
    )
    # Two chunk scores followed by one paper-level score per paper.
    kb.rerank_pairs = AsyncMock(return_value=[0.9, 0.8, 0.9, 0.2])
    results = asyncio.run(kb.rerank_and_filter_retrieval_results(
        [
            {
                "chunk_id": "p1:0",
                "paper_id": "p1",
                "paper_title": "RL",
                "text": "reinforcement learning",
                "score": 1.0,
            },
            {
                "chunk_id": "p2:0",
                "paper_id": "p2",
                "paper_title": "EEG",
                "text": "mentions reinforcement learning once",
                "score": 0.9,
            },
        ],
        queries=["有没有强化学习相关的文章"],
        top_k=5,
        per_paper_limit=3,
    ))

    assert [item["paper_id"] for item in results] == ["p1"]
    assert results[0]["paper_relevance_score"] == pytest.approx(0.9)
    kb._lexical_index.close()


def test_filtered_empty_results_do_not_trigger_jsonl_reflow(tmp_path: Path):
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(
            enabled=True,
            enable_hypothetical_retrieval=False,
            rerank_model="fake",
            retrieval_min_relevance_score=0.5,
        ),
    )
    kb._chroma_client = object()
    kb._chunk_collection = object()
    kb._summary_collection = object()
    kb._question_collection = object()
    kb._retrieve_dense_hybrid = AsyncMock(return_value=[{
        "chunk_id": "p1:0",
        "paper_id": "p1",
        "text": "irrelevant chunk",
        "score": 1.0,
        "score_type": "weighted_rrf",
    }])
    kb._retrieve_jsonl_multiquery = AsyncMock(return_value=[{
        "chunk_id": "p1:0",
        "paper_id": "p1",
        "text": "must not reflow",
    }])
    kb.rerank_pairs = AsyncMock(return_value=[0.1])

    results = asyncio.run(kb.retrieve_by_hypothetical_questions(
        query="解释强化学习方法",
        top_k=5,
    ))

    assert results == []
    kb._retrieve_jsonl_multiquery.assert_not_awaited()
    kb._lexical_index.close()


def test_relevance_filter_fails_closed_when_reranker_errors(tmp_path: Path):
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(
            enabled=True,
            enable_hypothetical_retrieval=False,
            rerank_model="missing",
            retrieval_relevance_fail_closed=True,
        ),
    )
    kb.rerank_pairs = AsyncMock(side_effect=RuntimeError("reranker unavailable"))

    results = asyncio.run(kb.rerank_and_filter_retrieval_results(
        [{
            "chunk_id": "p1:0",
            "paper_id": "p1",
            "text": "unverified evidence",
            "score": 1.0,
        }],
        queries=["query"],
        top_k=5,
        per_paper_limit=3,
    ))

    assert results == []
    kb._lexical_index.close()


def test_jsonl_pair_commit_restores_previous_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, enable_hypothetical_retrieval=False),
    )
    previous_doc = {"paper_id": "p1", "title": "old"}
    previous_chunk = {"paper_id": "p1", "chunk_id": "p1:0", "text": "old text"}
    kb._write_jsonl(kb.docs_file, [previous_doc])
    kb._write_jsonl(kb.chunks_file, [previous_chunk])
    original_write = kb._write_jsonl
    failed = False

    def _flaky_write(path: Path, rows: list[dict]) -> None:
        nonlocal failed
        if (
            path == kb.chunks_file
            and not failed
            and any(row.get("text") == "new text" for row in rows)
        ):
            failed = True
            raise OSError("simulated chunks commit failure")
        original_write(path, rows)

    monkeypatch.setattr(kb, "_write_jsonl", _flaky_write)
    with pytest.raises(OSError, match="simulated chunks commit failure"):
        asyncio.run(kb._persist_paper_rows(
            doc_row={"paper_id": "p1", "title": "new"},
            chunk_rows=[{"paper_id": "p1", "chunk_id": "p1:0", "text": "new text"}],
        ))

    assert kb._read_jsonl(kb.docs_file) == [previous_doc]
    assert kb._read_jsonl(kb.chunks_file) == [previous_chunk]


def test_chroma_multi_collection_failure_restores_previous_vectors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _Collection:
        def __init__(self, records: list[dict], *, fail_next: bool = False):
            self.records = {record["id"]: dict(record) for record in records}
            self.fail_next = fail_next

        def get(self, *, where: dict, include=None):
            selected = [
                record for record in self.records.values()
                if record["metadata"].get("paper_id") == where.get("paper_id")
            ]
            return {
                "ids": [record["id"] for record in selected],
                "embeddings": [record["embedding"] for record in selected],
                "documents": [record["document"] for record in selected],
                "metadatas": [record["metadata"] for record in selected],
            }

        def delete(self, *, ids: list[str]):
            for record_id in ids:
                self.records.pop(record_id, None)

        def upsert(self, *, ids, embeddings, documents, metadatas):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("simulated summary upsert failure")
            for index, record_id in enumerate(ids):
                self.records[record_id] = {
                    "id": record_id,
                    "embedding": list(embeddings[index]),
                    "document": documents[index],
                    "metadata": dict(metadatas[index]),
                }

    def _old_record(record_id: str, document: str) -> dict:
        return {
            "id": record_id,
            "embedding": [1.0, 0.0],
            "document": document,
            "metadata": {"paper_id": "p1"},
        }

    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, min_chunk_chars=10),
    )
    chunks = _Collection([_old_record("p1:0", "old chunk")])
    summaries = _Collection(
        [_old_record("p1:0:summary", "old summary")],
        fail_next=True,
    )
    questions = _Collection([_old_record("p1:0:q0", "old question")])
    kb._chroma_client = object()
    kb._chunk_collection = chunks
    kb._summary_collection = summaries
    kb._question_collection = questions

    async def _direct_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("nanobot.agent.paper_kb.asyncio.to_thread", _direct_to_thread)

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]

    kb.embed_texts = _embed
    with pytest.raises(RuntimeError, match="simulated summary upsert failure"):
        asyncio.run(kb.upsert_semantic_chunks(
            {"paper_id": "p1", "title": "new paper"},
            [{
                "section": "Method",
                "heading_path": "Method",
                "text": "new chunk content long enough",
            }],
            [{
                "summary": "new summary",
                "hypothetical_questions": ["new question"],
            }],
        ))

    assert set(chunks.records) == {"p1:0"}
    assert chunks.records["p1:0"]["document"] == "old chunk"
    assert summaries.records["p1:0:summary"]["document"] == "old summary"
    assert questions.records["p1:0:q0"]["document"] == "old question"


class TestMarkdownSemanticChunking:
    """Tests for _split_markdown_semantic function."""

    def test_basic_chunking(self):
        """Test basic Markdown header-based chunking."""
        chunks = _split_markdown_semantic(SAMPLE_MARKDOWN)
        
        assert len(chunks) > 0
        # Should have chunks for Abstract, Introduction, Method, Results, Conclusion
        sections = [c["section"] for c in chunks]
        assert "abstract" in sections or "Abstract" in sections
        
        # Each chunk should have required fields
        for chunk in chunks:
            assert "section" in chunk
            assert "text" in chunk
            assert "heading_level" in chunk
            assert "heading_path" in chunk
            assert len(chunk["text"]) > 0

    def test_noisy_blocks_removed(self):
        """Test that references and acknowledgements are removed."""
        clean_text = _remove_noisy_blocks(SAMPLE_MARKDOWN)
        
        assert "References" not in clean_text
        assert "Acknowledgements" not in clean_text
        assert "Smith et al." not in clean_text
        
        # Should retain main content
        assert "Abstract" in clean_text
        assert "Introduction" in clean_text

    def test_empty_text(self):
        """Test handling of empty text."""
        chunks = _split_markdown_semantic("")
        assert chunks == []

    def test_no_headers(self):
        """Test handling of text without headers."""
        text = "This is plain text without any headers.\n\nAnother paragraph."
        chunks = _split_markdown_semantic(text, min_chunk_chars=10)
        
        # Should return as single chunk
        assert len(chunks) >= 1
        assert chunks[0]["section"] == "content"

    def test_chunk_size_limits(self):
        """Test that chunks respect size limits."""
        # Long text that should be split
        long_text = "# Section\n\n" + ("Very long content paragraph. " * 100)
        chunks = _split_markdown_semantic(long_text, max_chunk_chars=500, min_chunk_chars=10)
        
        for chunk in chunks:
            assert len(chunk["text"]) <= 500 + 50  # Allow some flexibility

    def test_heading_path_preserved(self):
        """Test that heading path is correctly captured."""
        chunks = _split_markdown_semantic(SAMPLE_MARKDOWN)
        
        # Find the Motivation chunk (under Introduction)
        motivation_chunks = [c for c in chunks if "motivation" in c["section"].lower()]
        if motivation_chunks:
            # Should have heading_path showing hierarchy
            assert "Introduction" in motivation_chunks[0]["heading_path"] or \
                   "introduction" in motivation_chunks[0]["heading_path"].lower()


class TestChunkMetadataGeneration:
    """Tests for _generate_chunk_metadata function."""

    @pytest.mark.asyncio
    async def test_fallback_without_provider(self):
        """Test fallback behavior when no LLM provider available."""
        result = await _generate_chunk_metadata(
            text="This is a test chunk about machine learning.",
            section="Introduction",
            provider=None,
            model=None,
            num_questions=3,
        )
        
        assert "summary" in result
        assert "hypothetical_questions" in result
        assert "keywords" in result
        assert len(result["hypothetical_questions"]) <= 3

    @pytest.mark.asyncio
    async def test_with_mock_provider(self):
        """Test LLM-based metadata generation with mock."""
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "summary": "This discusses machine learning approaches.",
            "hypothetical_questions": [
                "What machine learning methods are discussed?",
                "How does this relate to deep learning?",
                "What are the key findings?",
            ],
            "keywords": ["machine learning", "deep learning", "AI"],
        })
        mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)
        
        result = await _generate_chunk_metadata(
            text="This is a test chunk about machine learning and deep learning approaches.",
            section="Method",
            provider=mock_provider,
            model="test-model",
            num_questions=3,
        )
        
        assert result["summary"] == "This discusses machine learning approaches."
        assert len(result["hypothetical_questions"]) == 3
        assert "machine learning" in result["keywords"]

    def test_singleton_list_response_is_normalized(self):
        """Accept a common LLM shape variation without invoking fallback."""
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps([{
            "summary": "Wrapped metadata object.",
            "hypothetical_questions": ["What is wrapped?"],
            "keywords": ["metadata"],
            "entities": ["JSON"],
            "claims": ["The response uses a singleton array."],
        }])
        mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)

        result = asyncio.run(_generate_chunk_metadata(
            text="This chunk describes a singleton JSON metadata response.",
            section="Method",
            provider=mock_provider,
            model="test-model",
            num_questions=1,
        ))

        assert result["summary"] == "Wrapped metadata object."
        assert result["keywords"] == ["metadata"]
        assert result["entities"] == ["JSON"]

    def test_visible_reasoning_before_fenced_json_is_ignored(self):
        payload = {
            "summary": "Grounded result.",
            "hypothetical_questions": ["What result is reported?"],
            "keywords": ["result"],
            "entities": [],
            "claims": [],
        }
        raw = "Thinking Process:\n1. analyze\n```json\n" + json.dumps(payload) + "\n```"

        assert _decode_chunk_metadata_json(raw) == payload

    def test_truncated_reasoning_retries_with_shorter_input(self):
        mock_provider = MagicMock()
        truncated = MagicMock(
            content=None,
            reasoning_content="thinking without a closing token" * 20,
            finish_reason="length",
            usage={"completion_tokens": 1600},
        )
        recovered = MagicMock(
            content=json.dumps({
                "summary": "Recovered metadata.",
                "hypothetical_questions": ["How was metadata recovered?"],
                "keywords": ["metadata"],
                "entities": [],
                "claims": [],
            }),
            reasoning_content=None,
            finish_reason="stop",
            usage={"completion_tokens": 80},
        )
        mock_provider.chat_with_retry = AsyncMock(side_effect=[truncated, recovered])

        result = asyncio.run(_generate_chunk_metadata(
            text="long evidence " * 400,
            section="Results",
            provider=mock_provider,
            model="Qwen3.5-9B",
            num_questions=1,
        ))

        assert result["summary"] == "Recovered metadata."
        assert mock_provider.chat_with_retry.await_count == 2
        first_call, second_call = mock_provider.chat_with_retry.await_args_list
        assert first_call.kwargs["max_tokens"] == 1600
        assert second_call.kwargs["max_tokens"] == 2000
        assert len(second_call.kwargs["messages"][1]["content"]) < len(
            first_call.kwargs["messages"][1]["content"]
        )
        assert first_call.kwargs["reasoning_effort"] is None

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        """Test fallback when LLM returns invalid JSON."""
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Not valid JSON at all"
        mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)
        
        result = await _generate_chunk_metadata(
            text="Test content for fallback.",
            section="Results",
            provider=mock_provider,
            model="test-model",
            num_questions=3,
        )
        
        # Should have fallback values
        assert "summary" in result
        assert "hypothetical_questions" in result
        assert len(result["hypothetical_questions"]) >= 1


class TestPaperKnowledgeBaseWithChroma:
    """Tests for PaperKnowledgeBase Chroma integration."""

    @pytest.fixture
    def kb_config(self, tmp_path: Path):
        """Create test KB config."""
        return PaperKbConfig(
            enabled=True,
            embedding_model="",  # Will use hash embeddings
            retrieval_relevance_filter_enabled=False,
            num_hypothetical_questions=3,
            enable_hypothetical_retrieval=True,
            chroma_persist_dir=str(tmp_path / "chroma"),
            min_chunk_chars=10,
        )

    @pytest.fixture
    def paper_kb(self, tmp_path: Path, kb_config: PaperKbConfig):
        """Create test PaperKnowledgeBase."""
        return PaperKnowledgeBase(tmp_path, kb_config)

    def test_chroma_initialization(self, paper_kb: PaperKnowledgeBase):
        """Test that Chroma collections are initialized."""
        assert paper_kb._chroma_client is not None
        assert paper_kb._summary_collection is not None
        assert paper_kb._question_collection is not None
        assert paper_kb._chunk_collection is not None

    @pytest.mark.asyncio
    async def test_upsert_semantic_chunks(self, paper_kb: PaperKnowledgeBase):
        """Test upserting semantic chunks to Chroma."""
        doc = {
            "paper_id": "test-paper-001",
            "title": "Test Paper",
            "url": "https://example.com/test.pdf",
            "source": "arxiv",
            "year": 2024,
        }
        
        semantic_chunks = [
            {
                "section": "Introduction",
                "heading_level": 1,
                "text": "This paper introduces a new method for document chunking.",
                "heading_path": "Introduction",
            },
            {
                "section": "Method",
                "heading_level": 1,
                "text": "We use Markdown headers to split documents into semantic chunks.",
                "heading_path": "Method",
            },
        ]
        
        chunk_metadata = [
            {
                "summary": "Introduces document chunking method.",
                "hypothetical_questions": [
                    "What is this paper about?",
                    "What problem does it solve?",
                    "What method is proposed?",
                ],
                "keywords": ["chunking", "document", "method"],
            },
            {
                "summary": "Describes Markdown-based splitting approach.",
                "hypothetical_questions": [
                    "How does the method work?",
                    "What is the chunking algorithm?",
                    "What format is used?",
                ],
                "keywords": ["markdown", "headers", "splitting"],
            },
        ]
        
        original_embed_texts = paper_kb.embed_texts
        paper_kb.embed_texts = AsyncMock(side_effect=original_embed_texts)
        result = await paper_kb.upsert_semantic_chunks(doc, semantic_chunks, chunk_metadata)
        
        assert result["paper_id"] == "test-paper-001"
        assert result["chunk_count"] == 2
        assert result["question_count"] == 6  # 3 questions per chunk
        assert result["success"] is True
        assert paper_kb.embed_texts.await_count == 2
        assert len(paper_kb.embed_texts.await_args_list[0].args[0]) == 2
        assert len(paper_kb.embed_texts.await_args_list[1].args[0]) == 8
        
        # Verify Chroma collections have data
        assert paper_kb._chunk_collection.count() >= 2
        assert paper_kb._summary_collection.count() >= 2
        assert paper_kb._question_collection.count() >= 6
        assert result["lexical"]["document_count"] == 2
        assert paper_kb._lexical_index.count() == 2
        parent_meta = paper_kb._chunk_collection.get(
            ids=["test-paper-001:0"], include=["metadatas"]
        )["metadatas"][0]
        summary_meta = paper_kb._summary_collection.get(
            ids=["test-paper-001:0:summary"], include=["metadatas"]
        )["metadatas"][0]
        question_meta = paper_kb._question_collection.get(
            ids=["test-paper-001:0:q0"], include=["metadatas"]
        )["metadatas"][0]
        assert parent_meta["chunk_index"] == 0
        assert summary_meta["chunk_index"] == 0
        assert question_meta["chunk_index"] == 0
        jsonl_chunks = paper_kb._read_jsonl(paper_kb.chunks_file)
        assert sum(1 for chunk in jsonl_chunks if chunk.get("paper_id") == "test-paper-001") == 2

    @pytest.mark.asyncio
    async def test_retrieve_by_hypothetical_questions(self, paper_kb: PaperKnowledgeBase):
        """Test retrieval via hypothetical questions."""
        # First insert some test data
        doc = {
            "paper_id": "test-paper-002",
            "title": "Semantic Chunking Paper",
            "source": "arxiv",
            "year": 2024,
        }
        
        semantic_chunks = [
            {
                "section": "Method",
                "heading_level": 2,
                "text": "Our method uses Markdown headers to identify document boundaries and create semantic chunks.",
                "heading_path": "Method",
            },
        ]
        
        chunk_metadata = [
            {
                "summary": "Markdown-based semantic chunking method.",
                "hypothetical_questions": [
                    "How do you split documents into chunks?",
                    "What method is used for chunking?",
                    "How does Markdown header chunking work?",
                ],
                "keywords": ["markdown", "chunking", "semantic"],
            },
        ]
        
        await paper_kb.upsert_semantic_chunks(doc, semantic_chunks, chunk_metadata)
        
        # Now test retrieval with a similar question
        results = await paper_kb.retrieve_by_hypothetical_questions(
            query="How does the document chunking method work?",
            top_k=5,
            search_mode="hybrid",
        )
        
        assert len(results) >= 1
        assert "text" in results[0]
        assert "chunk_id" in results[0]
        assert "matched_by" in results[0]  # Should show whether matched by question or summary
        assert "embedding" not in results[0]  # Internal MMR vectors must not reach prompts

    @pytest.mark.asyncio
    async def test_jsonl_fallback_keeps_semantic_chunks_retrievable(
        self,
        paper_kb: PaperKnowledgeBase,
    ):
        """A missing Chroma backend must not turn a successful ingest into zero chunks."""
        paper_kb._chroma_client = None
        paper_kb._summary_collection = None
        paper_kb._question_collection = None
        paper_kb._chunk_collection = None

        result = await paper_kb.upsert_semantic_chunks(
            {"paper_id": "fallback-001", "title": "Fallback Paper", "year": 2025},
            [{
                "section": "Method",
                "text": "The fallback method stores semantic parent chunks in JSONL.",
                "heading_path": "Method",
            }],
            [{
                "summary": "A JSONL fallback method.",
                "hypothetical_questions": ["How does fallback storage work?"],
                "keywords": ["fallback", "jsonl"],
            }],
        )

        assert result["success"] is True
        assert result["degraded"] is True
        assert result["chunk_count"] == 1
        retrieved = await paper_kb.retrieve_by_hypothetical_questions(
            query="fallback JSONL storage",
            top_k=1,
        )
        assert retrieved[0]["paper_id"] == "fallback-001"

    @pytest.mark.asyncio
    async def test_delete_paper_from_chroma(self, paper_kb: PaperKnowledgeBase):
        """Test deletion of paper data from Chroma."""
        # Insert test data
        doc = {
            "paper_id": "test-paper-003",
            "title": "Test Paper for Deletion",
            "source": "arxiv",
            "year": 2024,
        }
        
        semantic_chunks = [
            {
                "section": "content",
                "heading_level": 1,
                "text": "Test content for deletion test.",
                "heading_path": "content",
            },
        ]
        
        chunk_metadata = [{
            "summary": "Test summary.",
            "hypothetical_questions": ["Test question?"],
            "keywords": ["test"],
        }]
        
        await paper_kb.upsert_semantic_chunks(doc, semantic_chunks, chunk_metadata)
        
        # Verify data exists
        assert paper_kb._chunk_collection.count() >= 1
        
        # Delete
        paper_kb._delete_paper_from_chroma("test-paper-003")
        
        # Verify deletion
        remaining = paper_kb._chunk_collection.get(
            where={"paper_id": "test-paper-003"},
        ).get("ids", [])
        assert len(remaining) == 0


class TestKBRetrieveToolModes:
    """Tests for KBRetrieveTool retrieval modes."""

    @pytest.fixture
    def mock_kb(self):
        """Create mock PaperKnowledgeBase."""
        kb = MagicMock(spec=PaperKnowledgeBase)
        kb.config = PaperKbConfig(enable_hypothetical_retrieval=True)
        kb.retrieve = AsyncMock(return_value=[
            {"chunk_id": "test:0", "text": "Traditional result", "score": 0.8}
        ])
        kb.retrieve_by_hypothetical_questions = AsyncMock(return_value=[
            {"chunk_id": "test:0", "text": "Hypothetical result", "score": 0.9, "matched_by": "question"}
        ])
        return kb

    # Note: Full KBRetrieveTool tests would require more setup
    # These are placeholder tests for the retrieval mode logic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
