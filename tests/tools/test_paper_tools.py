import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from nanobot.agent.paper_kb import PaperKbConfig, PaperKnowledgeBase
from nanobot.agent.tools import paper as paper_module
from nanobot.agent.tools.paper import (
    KBRetrieveTool,
    PaperIngestTool,
    PaperRerankTool,
    PaperSearchTool,
    PaperSimilarityTool,
    _ArxivRateLimiter,
    _build_arxiv_query,
    _build_arxiv_query_group,
    _deduplicate_papers_by_identity,
    _extract_arxiv_ids,
    _parse_arxiv,
    _rrf_fuse_paper_rankings,
    _select_diverse_papers,
    _valid_extracted_text,
)


def test_pdf_page_markers_alone_are_not_valid_extracted_text():
    markers = "\n".join(f"--- Page {page} ---" for page in range(1, 30))
    assert _valid_extracted_text(markers) is False


def test_uploaded_front_matter_extracts_deterministic_identifiers(tmp_path: Path):
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, enable_hypothetical_retrieval=False),
    )
    tool = PaperIngestTool(workspace=tmp_path, kb=kb)
    paper = {"metadata_provenance": {"title": "filename"}}

    tool._enrich_deterministic_identifiers(
        paper,
        "arXiv: 2405.01234v2\nDOI: 10.1145/1234567.7654321",
    )

    assert paper["arxiv_id"] == "2405.01234v2"
    assert paper["doi"] == "10.1145/1234567.7654321"
    assert paper["year"] == 2024
    assert paper["metadata_provenance"]["year"] == "arxiv_id"


@pytest.mark.asyncio
async def test_paper_search_returns_ranked_results(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, embedding_model="", rerank_model=""),
    )
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


def test_external_paper_identity_dedup_handles_versions_doi_and_title():
    deduped = _deduplicate_papers_by_identity([
        {"paper_id": "2401.12345v1", "title": "First title"},
        {"paper_id": "2401.12345v2", "title": "First title revised"},
        {"paper_id": "a", "doi": "10.1000/shared", "title": "DOI copy one"},
        {"paper_id": "b", "doi": "https://doi.org/10.1000/shared", "title": "DOI copy two"},
        {"paper_id": "c", "title": "Exactly Repeated Paper Title"},
        {"paper_id": "d", "title": "Exactly repeated-paper title!"},
    ])

    assert [paper["paper_id"] for paper in deduped] == ["2401.12345v1", "a", "c"]


def test_external_paper_mmr_adds_diversity_only_within_new_candidates():
    selected = _select_diverse_papers([
        {
            "paper_id": "p1",
            "title": "Agent reinforcement learning framework",
            "abstract": "modular agent reinforcement learning",
            "rerank_score": 1.0,
        },
        {
            "paper_id": "p2",
            "title": "Agent reinforcement learning framework variant",
            "abstract": "modular agent reinforcement learning",
            "rerank_score": 0.99,
        },
        {
            "paper_id": "p3",
            "title": "Reinforcement learning evaluation benchmark",
            "abstract": "datasets metrics evaluation",
            "rerank_score": 0.90,
        },
    ], top_k=2)

    assert [paper["paper_id"] for paper in selected] == ["p1", "p3"]


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


def test_arxiv_query_group_combines_variants_and_applies_date_once():
    query = _build_arxiv_query_group(
        [
            ("EEG image reconstruction", ["guided diffusion"]),
            ("brain signal decoding", ["visual generation"]),
        ],
        from_year=2024,
        to_year=2026,
    )

    assert 'ti:"EEG image reconstruction"' in query
    assert 'ti:"brain signal decoding"' in query
    assert ") OR (" in query
    assert query.count("submittedDate:") == 1


def test_arxiv_ids_are_extracted_from_wrapped_queries_and_urls():
    assert _extract_arxiv_ids(
        "去外部搜索2511.14460v2这篇论文：https://arxiv.org/abs/2401.12345"
    ) == ["2511.14460v2", "2401.12345"]


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


def test_arxiv_parser_uses_id_list_for_exact_lookup():
    atom = b'''<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"></feed>'''

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
    asyncio.run(_parse_arxiv(
        "ignored semantic query",
        paper_ids=["2511.14460v2"],
        from_year=2020,
        to_year=2021,
        client=client,
        rate_limiter=_ArxivRateLimiter(0),
    ))

    assert "id_list=2511.14460v2" in client.url
    assert "search_query=" not in client.url
    assert "submittedDate" not in client.url


@pytest.mark.asyncio
async def test_arxiv_rate_limiter_serializes_complete_requests():
    limiter = _ArxivRateLimiter(0)
    active_requests = 0
    max_active_requests = 0

    async def _request():
        nonlocal active_requests, max_active_requests
        active_requests += 1
        max_active_requests = max(max_active_requests, active_requests)
        await asyncio.sleep(0.01)
        active_requests -= 1
        return type("Response", (), {"status_code": 200})()

    await asyncio.gather(
        limiter.request(_request),
        limiter.request(_request),
        limiter.request(_request),
    )

    assert max_active_requests == 1


@pytest.mark.asyncio
async def test_arxiv_429_starts_shared_cooldown_without_immediate_retries():
    limiter = _ArxivRateLimiter(0, default_cooldown_seconds=7)

    class _Client:
        def __init__(self):
            self.calls = 0

        async def get(self, url):
            self.calls += 1
            request = httpx.Request("GET", url)
            return httpx.Response(
                429,
                request=request,
                headers={"Retry-After": "1"},
            )

    client = _Client()
    first = await _parse_arxiv(
        "EEG reconstruction",
        client=client,
        rate_limiter=limiter,
    )
    second = await _parse_arxiv(
        "brain decoding",
        client=client,
        rate_limiter=limiter,
    )

    assert first.status == "rate_limited"
    assert first.attempts == 1
    assert first.retry_after_seconds >= 6.0
    assert second.status == "rate_limited"
    assert client.calls == 1


def test_exact_id_search_bypasses_exclusion_and_semantic_rerank(
    tmp_path: Path,
    monkeypatch,
):
    calls = []

    async def _fake_parse_arxiv(query, keywords=None, max_results=20, **kwargs):
        calls.append((query, kwargs))
        return [{
            "paper_id": "2511.14460v2",
            "title": "Agent-R1",
            "abstract": "A unified agentic reinforcement learning framework.",
            "url": "https://arxiv.org/abs/2511.14460v2",
            "source": "arxiv",
            "year": 2025,
        }]

    monkeypatch.setattr("nanobot.agent.tools.paper._parse_arxiv", _fake_parse_arxiv)
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, embedding_model="", rerank_model=""),
    )
    tool = PaperSearchTool(workspace=tmp_path, kb=kb)
    payload = json.loads(asyncio.run(tool.execute(
        query="去外部搜索2511.14460v2这篇论文",
        candidate_queries=[
            "Agent-R1 unified modular framework Agentic RL paper 2511.14460v2",
            "Agentic reinforcement learning modular framework",
        ],
        exclude_paper_ids=["2511.14460v1"],
        from_year=2020,
        to_year=2021,
    )))

    assert payload["sort_mode"] == "exact_id"
    assert payload["explicit_paper_ids"] == ["2511.14460v2"]
    assert [paper["paper_id"] for paper in payload["results"]] == ["2511.14460v2"]
    assert payload["results"][0]["identity_match"] is True
    assert calls[0][1]["paper_ids"] == ["2511.14460v2"]
    assert calls[0][1]["from_year"] is None


def test_exact_id_search_batches_multiple_ids_in_one_request(
    tmp_path: Path,
    monkeypatch,
):
    calls = []

    async def _fake_parse_arxiv(query, keywords=None, max_results=20, **kwargs):
        calls.append(kwargs)
        return [
            {
                "paper_id": paper_id,
                "title": f"Paper {paper_id}",
                "abstract": "Abstract",
                "year": 2025,
            }
            for paper_id in kwargs["paper_ids"]
        ]

    monkeypatch.setattr("nanobot.agent.tools.paper._parse_arxiv", _fake_parse_arxiv)
    kb = PaperKnowledgeBase(tmp_path, PaperKbConfig(enabled=True))
    tool = PaperSearchTool(workspace=tmp_path, kb=kb)
    payload = json.loads(asyncio.run(tool.execute(
        query="compare 2511.14460v2 and 2401.12345",
        rerank_top_k=5,
    )))

    assert len(calls) == 1
    assert calls[0]["paper_ids"] == ["2511.14460v2", "2401.12345"]
    assert [paper["paper_id"] for paper in payload["results"]] == [
        "2511.14460v2",
        "2401.12345",
    ]


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
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, embedding_model="", rerank_model=""),
    )
    tool = PaperSearchTool(workspace=tmp_path, kb=kb)
    payload = json.loads(asyncio.run(tool.execute(
        query="latest RAG",
        candidate_queries=[
            "retrieval augmented generation",
            "scientific question answering",
        ],
        exclude_paper_ids=["2401.00001v3"],
        from_year=2024,
        to_year=2025,
        sort_mode="balanced",
        search_topk=10,
        recall_top_k=10,
        rerank_top_k=10,
    )))

    assert len(calls) == 2
    assert {call["sort_by"] for call in calls} == {"relevance", "submittedDate"}
    assert all(len(call["query_variants"]) == 2 for call in calls)
    assert {paper["paper_id"] for paper in payload["results"]} == {
        "2402.00001v1",
        "2501.00001v1",
    }
    assert payload["excluded_total"] == 1
    assert payload["sort_mode"] == "balanced"


def test_external_search_cache_is_shared_between_tool_instances(
    tmp_path: Path,
    monkeypatch,
):
    calls = 0

    async def _fake_parse_arxiv(query, keywords=None, max_results=20, **kwargs):
        nonlocal calls
        calls += 1
        return [{
            "paper_id": "2401.12345",
            "title": "Cached paper",
            "abstract": "Shared arXiv cache result.",
            "year": 2024,
        }]

    monkeypatch.setattr("nanobot.agent.tools.paper._parse_arxiv", _fake_parse_arxiv)
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, embedding_model="", rerank_model=""),
    )
    first = PaperSearchTool(workspace=tmp_path, kb=kb)

    first_payload = json.loads(asyncio.run(first.execute(
        query="cached retrieval",
        candidate_queries=["cached retrieval"],
    )))
    cache_path = (tmp_path / "kb" / "arxiv_search_cache.json").resolve()
    paper_module._ARXIV_SEARCH_CACHES.pop(str(cache_path), None)
    second = PaperSearchTool(workspace=tmp_path, kb=kb)
    second_payload = json.loads(asyncio.run(second.execute(
        query="cached retrieval",
        candidate_queries=["cached retrieval"],
    )))

    assert calls == 1
    assert first_payload["results"]
    assert second_payload["route_diagnostics"][0]["cached"] is True
    assert cache_path.exists()


def test_external_search_stops_remaining_routes_during_shared_cooldown(
    tmp_path: Path,
    monkeypatch,
):
    calls = 0

    async def _fake_parse_arxiv(query, keywords=None, max_results=20, **kwargs):
        nonlocal calls
        calls += 1
        return paper_module._ArxivSearchResult(
            papers=[],
            status="rate_limited",
            query=query,
            sort_by=kwargs["sort_by"],
            error="429 Too Many Requests",
            retry_after_seconds=300,
        )

    monkeypatch.setattr("nanobot.agent.tools.paper._parse_arxiv", _fake_parse_arxiv)
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, embedding_model="", rerank_model=""),
    )
    tool = PaperSearchTool(workspace=tmp_path, kb=kb)
    payload = json.loads(asyncio.run(tool.execute(
        query="latest EEG reconstruction",
        candidate_queries=["EEG reconstruction"],
        sort_mode="balanced",
    )))

    assert calls == 1
    assert payload["search_status"] == "rate_limited"
    assert payload["reason"] == "rate_limited"
    assert payload["retry_after_seconds"] == 300
    assert len(payload["route_diagnostics"]) == 2
    assert payload["route_diagnostics"][1]["attempts"] == 0


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


def test_kb_retrieve_default_is_hybrid_and_omits_internal_embeddings(tmp_path: Path):
    class _FakeKB:
        config = PaperKbConfig(enable_hypothetical_retrieval=True)

        def __init__(self):
            self.retrieve_by_hypothetical_questions = AsyncMock(return_value=[{
                "chunk_id": f"p1:{index}",
                "paper_id": "p1",
                "paper_title": "ATM",
                "section": "Experiments",
                "text": "performance evidence " * 500,
                "embedding": [0.01] * 1024,
                "score": 0.9 - index * 0.01,
                "linked_assets": [{
                    "key": f"table-{index}",
                    "type": "table",
                    "caption": "Quantitative performance table",
                    "content": "<table>metric values</table>" * 200,
                }],
            } for index in range(10)])

        def get_embedding_status(self):
            return {"backend": "test", "degraded": False}

        def get_lexical_status(self):
            return {"backend": "test", "document_count": 10, "degraded": False}

    kb = _FakeKB()
    tool = KBRetrieveTool(workspace=tmp_path, kb=kb)
    raw = asyncio.run(tool.execute(query="ATM 模型性能和实验指标"))
    payload = json.loads(raw)

    assert payload["retrieval_mode"] == "hybrid"
    assert kb.retrieve_by_hypothetical_questions.await_args.kwargs["search_mode"] == "hybrid"
    assert len(raw) <= tool._MAX_MODEL_PAYLOAD_CHARS
    assert all("embedding" not in result for result in payload["results"])
    assert payload["returned_hits"] <= payload["total_hits"]


@pytest.mark.asyncio
async def test_similarity_and_rerank_accept_stateless_candidates(tmp_path: Path, monkeypatch):
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, embedding_model="", rerank_model=""),
    )
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
    kb = PaperKnowledgeBase(
        tmp_path,
        PaperKbConfig(enabled=True, embedding_model="", rerank_model=""),
    )
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
    monkeypatch.setattr(ingest, "_extract_pdf_text", _fake_extract_text)

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
