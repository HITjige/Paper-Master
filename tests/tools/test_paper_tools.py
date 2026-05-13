import json
from pathlib import Path

import pytest

from nanobot.agent.paper_kb import PaperKbConfig, PaperKnowledgeBase
from nanobot.agent.tools.paper import (
    KBRetrieveTool,
    PaperIngestTool,
    PaperRerankTool,
    PaperSearchTool,
    PaperSimilarityTool,
)


@pytest.mark.asyncio
async def test_paper_search_returns_ranked_results(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enable=True))
    tool = PaperSearchTool(workspace=tmp_path, kb=kb)

    async def _fake_parse_arxiv(query: str, max_results: int = 20):
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
    result = await tool.execute("latest time series forecasting transformer papers", top_k=2)
    payload = json.loads(result)
    assert payload["results"][0]["paper_id"] == "a1"
    assert len(payload["results"]) == 2
    assert payload.get("candidate_set_id")


@pytest.mark.asyncio
async def test_similarity_and_rerank_support_candidate_set_id(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enable=True))
    search_tool = PaperSearchTool(workspace=tmp_path, kb=kb)
    sim_tool = PaperSimilarityTool(workspace=tmp_path, kb=kb)
    rerank_tool = PaperRerankTool(workspace=tmp_path, kb=kb)

    async def _fake_parse_arxiv(query: str, max_results: int = 20):
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
    search_payload = json.loads(await search_tool.execute(query="eeg diffusion", top_k=2))
    candidate_set_id = search_payload.get("candidate_set_id")
    assert candidate_set_id

    sim_payload = json.loads(
        await sim_tool.execute(query="eeg diffusion", candidate_set_id=candidate_set_id)
    )
    assert sim_payload["results"]

    rerank_payload = json.loads(
        await rerank_tool.execute(query="eeg diffusion", candidate_set_id=candidate_set_id, top_k=2)
    )
    assert rerank_payload["results"]


@pytest.mark.asyncio
async def test_paper_search_pipeline_mode(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enable=True))
    tool = PaperSearchTool(workspace=tmp_path, kb=kb)

    async def _fake_parse_arxiv(query: str, max_results: int = 20):
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
        await tool.execute(query="scientific qa", mode="pipeline", top_k=1)
    )
    assert payload.get("stage") == "pipeline_reranked"
    assert payload["results"]


@pytest.mark.asyncio
async def test_paper_rerank_uses_similarity_and_recency(tmp_path: Path):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enable=True))
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
async def test_paper_ingest_then_retrieve(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enable=True))
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
                b"Neural retrieval augmented generation for scientific question answering."
            )

    monkeypatch.setattr("nanobot.agent.tools.paper.httpx.AsyncClient", _FakeClient)
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
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enable=True))
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
                b"Abstract\nThis paper studies retrieval for QA.\nIntroduction\nMethod and experiments."
            )

    monkeypatch.setattr("nanobot.agent.tools.paper.httpx.AsyncClient", _FakeClient)
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
async def test_paper_ingest_persists_distilled_metadata(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enable=True))
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
                b"Abstract\nThis paper studies scientific QA.\nLimitations\nSmall dataset and compute constraints."
            )

    monkeypatch.setattr("nanobot.agent.tools.paper.httpx.AsyncClient", _FakeClient)
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
    assert result.get("distilled") is True


def test_kb_retrieve_uses_metadata_and_diversity(tmp_path: Path):
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enable=True))
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
