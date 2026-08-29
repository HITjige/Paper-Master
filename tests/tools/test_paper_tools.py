import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.paper_kb import PaperKbConfig, PaperKnowledgeBase
from nanobot.agent.tools.paper import (
    KBRetrieveTool,
    PaperIngestTool,
    PaperRerankTool,
    PaperSearchTool,
    PaperSimilarityTool,
    _ArxivRateLimiter,
    _build_arxiv_query,
    _parse_arxiv,
    _rrf_fuse_paper_rankings,
)


@pytest.mark.asyncio
async def test_paper_search_returns_ranked_results(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    tool = PaperSearchTool(workspace=tmp_path, kb=kb)

    async def _fake_parse_arxiv(query: str, keywords=None, max_results: int = 20, **kwargs):
        return [
            {
                "paper_id": "a1",
                "title": "Transformer methods for time series forecasting",
                "abstract": "A method for forecasting with attention.",
                "url": "https://arxiv.org/abs/a1",
                "source": "arxiv",
                "year": 2025,
            },
            {
                "paper_id": "a2",
                "title": "Graph mining overview",
                "abstract": "This survey is not about forecasting.",
                "url": "https://arxiv.org/abs/a2",
                "source": "arxiv",
                "year": 2021,
            },
        ]

    monkeypatch.setattr("nanobot.agent.tools.paper._parse_arxiv", _fake_parse_arxiv)
    result = await tool.execute(
        "latest time series forecasting transformer papers",
        candidate_queries=["time series forecasting transformer"],
        rerank_top_k=2,
    )
    payload = json.loads(result)
    assert payload["results"][0]["paper_id"] == "a1"
    assert len(payload["results"]) == 2
    assert payload["deduped_total"] == 2


def test_external_multi_query_rrf_rewards_cross_query_agreement():
    fused = _rrf_fuse_paper_rankings(
        [
            [
                {"paper_id": "single", "title": "Single-query hit"},
                {"paper_id": "2401.12345v1", "title": "Shared hit"},
            ],
            [{"paper_id": "2401.12345v2", "title": "Shared hit"}],
        ],
        ["query one", "query two"],
    )

    assert [paper["paper_id"] for paper in fused] == ["2401.12345v1", "single"]
    assert fused[0]["query_hit_count"] == 2
    assert fused[0]["query_rrf_score"] == 1.0
    assert fused[0]["query_matches"] == [
        {"query_index": 0, "rank": 2, "query": "query one"},
        {"query_index": 1, "rank": 1, "query": "query two"},
    ]


def test_arxiv_query_builder_uses_fields_exact_id_and_date_range():
    query = _build_arxiv_query(
        'RAG") OR all:*',
        ["scientific QA", "retrieval"],
        from_year=2024,
        to_year=2026,
    )
    assert 'ti:"RAG OR all:*"' in query
    assert 'abs:"scientific QA"' in query
    assert "submittedDate:[202401010000 TO 202612312359]" in query
    assert _build_arxiv_query("arxiv:2401.12345v2") == "id:2401.12345v2"


def test_arxiv_parser_returns_structured_metadata_and_diagnostics():
    atom = b'''<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>https://arxiv.org/abs/2401.12345v2</id>
        <updated>2026-01-03T00:00:00Z</updated>
        <published>2024-01-02T00:00:00Z</published>
        <title>Scientific RAG</title>
        <summary>Retrieval for scientific QA.</summary>
        <author><name>Alice</name></author>
        <category term="cs.IR"/>
        <arxiv:primary_category term="cs.IR"/>
        <arxiv:doi>10.1000/example</arxiv:doi>
        <link title="pdf" href="https://arxiv.org/pdf/2401.12345v2"/>
      </entry>
    </feed>'''

    class _Response:
        content = atom
        text = atom.decode()

        def raise_for_status(self):
            return None

    class _Client:
        async def get(self, url):
            self.url = url
            return _Response()

    client = _Client()
    result = asyncio.run(_parse_arxiv(
        "scientific RAG",
        ["retrieval"],
        client=client,
        rate_limiter=_ArxivRateLimiter(0),
    ))

    assert result.status == "ok"
    assert result.papers[0]["paper_id"] == "2401.12345v2"
    assert result.papers[0]["primary_category"] == "cs.IR"
    assert result.papers[0]["doi"] == "10.1000/example"
    assert "sortBy=relevance" in client.url


def test_external_search_filters_before_ranking_and_uses_balanced_routes(
    tmp_path: Path,
    monkeypatch,
):
    calls = []

    async def _fake_parse_arxiv(query, keywords=None, max_results=20, **kwargs):
        calls.append(kwargs)
        if kwargs["sort_by"] == "relevance":
            return [
                {"paper_id": "2401.00001v1", "title": "already", "abstract": "x", "year": 2024},
                {"paper_id": "2001.00001v1", "title": "old", "abstract": "x", "year": 2020},
                {"paper_id": "2402.00001v1", "title": "relevant", "abstract": "RAG", "year": 2024},
            ]
        return [
            {"paper_id": "2501.00001v1", "title": "recent", "abstract": "RAG", "year": 2025},
        ]

    monkeypatch.setattr("nanobot.agent.tools.paper._parse_arxiv", _fake_parse_arxiv)
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    tool = PaperSearchTool(workspace=tmp_path, kb=kb)
    payload = json.loads(asyncio.run(tool.execute(
        query="latest RAG",
        candidate_queries=["retrieval augmented generation"],
        exclude_paper_ids=["2401.00001v3"],
        from_year=2024,
        to_year=2025,
        sort_mode="balanced",
        search_topk=10,
        recall_top_k=10,
        rerank_top_k=10,
    )))

    assert {call["sort_by"] for call in calls} == {"relevance", "submittedDate"}
    assert {paper["paper_id"] for paper in payload["results"]} == {
        "2402.00001v1",
        "2501.00001v1",
    }
    assert payload["excluded_total"] == 1
    assert payload["sort_mode"] == "balanced"


def test_multi_query_similarity_uses_translated_variant(tmp_path: Path):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    kb.embed_texts = AsyncMock(return_value=[
        [0.0, 0.0],  # Chinese source query (lexical/hash-style mismatch)
        [1.0, 0.0],  # English rewrite
        [1.0, 0.0],  # Relevant paper
        [0.0, 1.0],  # Unrelated paper
    ])
    tool = PaperSimilarityTool(workspace=tmp_path, kb=kb)
    payload = json.loads(asyncio.run(tool.execute(
        query="论文检索",
        queries=["academic paper retrieval"],
        papers=[
            {"paper_id": "p1", "title": "Academic paper retrieval", "abstract": "RAG"},
            {"paper_id": "p2", "title": "Image classification", "abstract": "vision"},
        ],
        top_k=2,
    )))

    assert payload["results"][0]["paper_id"] == "p1"
    assert payload["results"][0]["matched_query"] == "academic paper retrieval"
    kb.embed_texts.assert_awaited_once()


def test_rerank_uses_configured_cross_encoder_in_one_batch(tmp_path: Path):
    class _FakeKB:
        config = PaperKbConfig(rerank_model="fake-cross-encoder")
        rerank_pairs = AsyncMock(return_value=[0.1, 0.9, 0.8, 0.2])

    kb = _FakeKB()
    tool = PaperRerankTool(workspace=tmp_path, kb=kb)
    payload = json.loads(asyncio.run(tool.execute(
        query="中文问题",
        queries=["english query"],
        papers=[
            {"paper_id": "p1", "title": "one", "abstract": "a", "year": 2024},
            {"paper_id": "p2", "title": "two", "abstract": "b", "year": 2024},
        ],
        top_k=2,
    )))

    assert payload["reranker"] == "cross_encoder"
    assert payload["results"][0]["paper_id"] == "p1"
    assert len(kb.rerank_pairs.await_args.args[0]) == 4


def test_kb_retrieve_modes_map_to_distinct_multi_view_searches(tmp_path: Path):
    class _FakeKB:
        config = PaperKbConfig(enable_hypothetical_retrieval=True)

        def __init__(self):
            self.retrieve = AsyncMock(return_value=[])
            self.retrieve_by_hypothetical_questions = AsyncMock(return_value=[])

        def get_embedding_status(self):
            return {"backend": "test"}

        def get_lexical_status(self):
            return {"backend": "test"}

    kb = _FakeKB()
    tool = KBRetrieveTool(workspace=tmp_path, kb=kb)
    expected = {
        "hypothetical": "questions_only",
        "traditional": "chunks_only",
        "hybrid": "hybrid",
    }

    for retrieval_mode, search_mode in expected.items():
        payload = json.loads(asyncio.run(tool.execute(
            query="paper retrieval",
            retrieval_mode=retrieval_mode,
        )))
        assert payload["retrieval_mode"] == retrieval_mode
        assert (
            kb.retrieve_by_hypothetical_questions.await_args.kwargs["search_mode"]
            == search_mode
        )

    assert kb.retrieve.await_count == 0


@pytest.mark.asyncio
async def test_similarity_and_rerank_accept_stateless_candidates(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    search_tool = PaperSearchTool(workspace=tmp_path, kb=kb)
    sim_tool = PaperSimilarityTool(workspace=tmp_path, kb=kb)
    rerank_tool = PaperRerankTool(workspace=tmp_path, kb=kb)

    async def _fake_parse_arxiv(query: str, keywords=None, max_results: int = 20, **kwargs):
        return [
            {
                "paper_id": "b1",
                "title": "EEG diffusion guidance",
                "abstract": "Diffusion guidance for EEG generation.",
                "url": "https://arxiv.org/abs/b1",
                "source": "arxiv",
                "year": 2026,
            },
            {
                "paper_id": "b2",
                "title": "Unrelated topic",
                "abstract": "Not relevant to EEG generation.",
                "url": "https://arxiv.org/abs/b2",
                "source": "arxiv",
                "year": 2022,
            },
        ]

    monkeypatch.setattr("nanobot.agent.tools.paper._parse_arxiv", _fake_parse_arxiv)
    search_payload = json.loads(await search_tool.execute(
        query="eeg diffusion",
        candidate_queries=["eeg diffusion"],
        rerank_top_k=2,
    ))
    candidates = search_payload["results"]
    assert candidates

    sim_payload = json.loads(
        await sim_tool.execute(query="eeg diffusion", papers=candidates)
    )
    assert sim_payload["results"]

    rerank_payload = json.loads(
        await rerank_tool.execute(query="eeg diffusion", papers=sim_payload["results"], top_k=2)
    )
    assert rerank_payload["results"]


@pytest.mark.asyncio
async def test_paper_search_pipeline_mode(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    tool = PaperSearchTool(workspace=tmp_path, kb=kb)

    async def _fake_parse_arxiv(query: str, keywords=None, max_results: int = 20, **kwargs):
        return [
            {
                "paper_id": "c1",
                "title": "RAG for scientific QA",
                "abstract": "Retrieval augmented generation for scientific QA.",
                "url": "https://arxiv.org/abs/c1",
                "source": "arxiv",
                "year": 2026,
            }
        ]

    monkeypatch.setattr("nanobot.agent.tools.paper._parse_arxiv", _fake_parse_arxiv)
    payload = json.loads(
        await tool.execute(
            query="scientific qa",
            candidate_queries=["scientific qa"],
            rerank_top_k=1,
        )
    )
    assert payload["results"]
    assert "rerank_score" in payload["results"][0]


@pytest.mark.asyncio
async def test_paper_rerank_uses_similarity_and_recency(tmp_path: Path):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    tool = PaperRerankTool(workspace=tmp_path, kb=kb)
    papers = [
        {"paper_id": "p1", "title": "Older high sim", "abstract": "forecasting", "year": 2018, "similarity_score": 0.95, "source": "arxiv"},
        {"paper_id": "p2", "title": "New medium sim", "abstract": "forecasting", "year": 2025, "similarity_score": 0.80, "source": "arxiv"},
    ]
    result = await tool.execute("forecasting", papers, top_k=2)
    payload = json.loads(result)
    assert payload["results"][0]["paper_id"] in {"p1", "p2"}
    assert "rerank_score" in payload["results"][0]


@pytest.mark.asyncio
async def test_paper_similarity_batches_embeddings(tmp_path: Path):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    kb.embed_texts = AsyncMock(return_value=[
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    tool = PaperSimilarityTool(workspace=tmp_path, kb=kb)

    payload = json.loads(await tool.execute(
        query="论文检索",
        papers=[
            {"paper_id": "p1", "title": "论文检索", "abstract": "相关研究"},
            {"paper_id": "p2", "title": "图像分类", "abstract": "无关研究"},
        ],
        top_k=2,
    ))

    kb.embed_texts.assert_awaited_once()
    assert payload["results"][0]["paper_id"] == "p1"


@pytest.mark.asyncio
async def test_paper_ingest_then_retrieve(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    ingest = PaperIngestTool(workspace=tmp_path, kb=kb)
    retrieve = KBRetrieveTool(workspace=tmp_path, kb=kb)

    class _FakeResponse:
        status_code = 200

        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            return _FakeResponse(
                b"# Introduction\n\n"
                + b"Neural retrieval augmented generation for scientific question answering. " * 12
            )

    monkeypatch.setattr("nanobot.agent.tools.paper.httpx.AsyncClient", _FakeClient)
    monkeypatch.setattr("nanobot.agent.tools.paper.validate_url_target", lambda url: (True, ""))
    paper = {
        "paper_id": "ingest-1",
        "title": "RAG for scientific QA",
        "url": "https://example.org/paper.txt",
        "source": "arxiv",
        "year": 2026,
    }
    ingest_result = json.loads(await ingest.execute(paper=paper, parse_mode="text"))
    assert ingest_result["status"] == "ok"

    retrieved = json.loads(await retrieve.execute(query="scientific question answering", top_k=3))
    assert retrieved["results"]
    assert retrieved["results"][0]["paper_id"] == "ingest-1"


@pytest.mark.asyncio
async def test_paper_ingest_batch_mode(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    ingest = PaperIngestTool(workspace=tmp_path, kb=kb)

    class _FakeResponse:
        status_code = 200

        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            return _FakeResponse(
                b"# Introduction\n\n"
                + b"This paper studies retrieval for QA, including methods and experiments. " * 12
            )

    monkeypatch.setattr("nanobot.agent.tools.paper.httpx.AsyncClient", _FakeClient)
    monkeypatch.setattr("nanobot.agent.tools.paper.validate_url_target", lambda url: (True, ""))
    payload = json.loads(
        await ingest.execute(
            papers=[
                {
                    "paper_id": "batch-1",
                    "title": "paper 1",
                    "url": "https://example.org/1.txt",
                    "source": "arxiv",
                },
                {
                    "paper_id": "batch-2",
                    "title": "paper 2",
                    "url": "https://example.org/2.txt",
                    "source": "arxiv",
                },
            ],
            parse_mode="text",
            concurrency=2,
            summarize=False,
        )
    )
    assert payload["total"] == 2
    assert payload["succeeded"] == 2


@pytest.mark.asyncio
async def test_paper_ingest_pdf_uses_local_extractor_without_mineru(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, min_chunk_chars=10),
    )
    ingest = PaperIngestTool(workspace=tmp_path, kb=kb, mineru_api_token="")

    class _FakeResponse:
        content = b"%PDF-1.7 fake test payload"
        headers = {"content-type": "application/pdf"}
        url = "https://example.org/paper.pdf"

        def raise_for_status(self):
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            return _FakeResponse()

    def _fake_extract_text(path: Path) -> str:
        assert path.suffix == ".pdf"
        return "# Method\n\n" + "Local PDF extraction produces grounded paper text. " * 12

    monkeypatch.setattr("nanobot.agent.tools.paper.httpx.AsyncClient", _FakeClient)
    monkeypatch.setattr("nanobot.agent.tools.paper.validate_url_target", lambda url: (True, ""))
    monkeypatch.setattr("nanobot.agent.tools.paper.validate_resolved_url", lambda url: (True, ""))
    monkeypatch.setattr("nanobot.agent.tools.paper.extract_text", _fake_extract_text)

    result = json.loads(await ingest.execute(
        paper={
            "paper_id": "pdf-fallback-1",
            "title": "PDF fallback",
            "pdf_url": "https://example.org/paper.pdf",
        },
        parse_mode="pdf",
    ))

    assert result["status"] == "ok"
    assert result["chunk_count"] > 0


@pytest.mark.asyncio
async def test_paper_ingest_persists_distilled_metadata(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    ingest = PaperIngestTool(workspace=tmp_path, kb=kb)

    class _FakeResponse:
        status_code = 200

        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            return _FakeResponse(
                b"# Introduction\n\n"
                + b"This paper studies scientific QA with small datasets and compute constraints. " * 12
            )

    monkeypatch.setattr("nanobot.agent.tools.paper.httpx.AsyncClient", _FakeClient)
    monkeypatch.setattr("nanobot.agent.tools.paper.validate_url_target", lambda url: (True, ""))
    result = json.loads(
        await ingest.execute(
            paper={
                "paper_id": "distill-1",
                "title": "distilled paper",
                "url": "https://example.org/d.txt",
                "source": "arxiv",
            },
            parse_mode="text",
            summarize=True,
        )
    )
    assert result["status"] == "ok"
    assert result["mode"] == "hypothetical"
    assert result["chunk_count"] > 0


def test_kb_retrieve_uses_metadata_and_diversity(tmp_path: Path):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    kb.base_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        {
            "paper_id": "p1",
            "title": "Paper 1",
            "url": "https://example.org/p1",
            "source": "arxiv",
            "year": 2026,
        },
        {
            "paper_id": "p2",
            "title": "Paper 2",
            "url": "https://example.org/p2",
            "source": "arxiv",
            "year": 2024,
        },
    ]
    kb._write_jsonl(kb.docs_file, docs)

    chunks = [
        {
            "chunk_id": "p1:0",
            "paper_id": "p1",
            "chunk_index": 0,
            "text": "General method summary.",
            "embedding": [],
            "section": "method",
            "kind": "distilled_summary",
            "limitations": [],
        },
        {
            "chunk_id": "p1:1",
            "paper_id": "p1",
            "chunk_index": 1,
            "text": "Limitations include small datasets and bias.",
            "embedding": [],
            "section": "limitations",
            "kind": "distilled_summary",
            "limitations": ["small datasets", "bias"],
        },
        {
            "chunk_id": "p1:2",
            "paper_id": "p1",
            "chunk_index": 2,
            "text": "Another limitations paragraph.",
            "embedding": [],
            "section": "limitations",
            "kind": "distilled_summary",
            "limitations": ["generalization"],
        },
        {
            "chunk_id": "p2:0",
            "paper_id": "p2",
            "chunk_index": 0,
            "text": "Results and experiments.",
            "embedding": [],
            "section": "results",
            "kind": "distilled_summary",
            "limitations": [],
        },
    ]
    kb._write_jsonl(kb.chunks_file, chunks)

    out = kb.retrieve_lexical(
        "what are the limitations",
        top_k=3,
        per_paper_limit=1,
    )
    assert out
    assert out[0].get("section") == "limitations"
    assert sum(1 for x in out if x.get("paper_id") == "p1") <= 1
