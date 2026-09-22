"""Focused regression tests for paper multi-agent routing and source handling."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.multi_agent.agents import (
    SYNTHESIS_SYSTEM_PROMPT,
    collect_answer_citations,
    format_sources_section,
)
from nanobot.agent.multi_agent.conditions import (
    research_phase_conditional,
    retrieval_conditional,
)
from nanobot.agent.multi_agent.graph import MultiAgentGraph
from nanobot.agent.multi_agent.nodes import AgentNodes


def test_hybrid_requires_external_search_even_with_sufficient_local_results():
    state = {
        "routing_decision": "hybrid",
        "retrieval_quality": "sufficient",
        "external_search_completed": False,
    }
    assert retrieval_conditional(state) == "research"


def test_retrieval_does_not_repeat_completed_external_search():
    state = {
        "routing_decision": "hybrid",
        "retrieval_quality": "insufficient",
        "external_search_completed": True,
    }
    assert retrieval_conditional(state) == "synthesis"


def test_external_abstracts_are_included_and_labelled():
    sources = format_sources_section(
        [],
        external_papers=[{
            "paper_id": "2401.12345",
            "title": "An External Paper",
            "abstract": "Only the abstract is available.",
            "url": "https://arxiv.org/abs/2401.12345",
        }],
    )
    assert 'id="2401.12345" evidence_level="abstract_only"' in sources
    assert "Only the abstract is available." in sources


def test_synthesis_prompt_requires_web_safe_math_and_architecture_formatting():
    assert "$...$" in SYNTHESIS_SYSTEM_PROMPT
    assert "$$...$$" in SYNTHESIS_SYSTEM_PROMPT
    assert "ASCII-art diagrams" in SYNTHESIS_SYSTEM_PROMPT
    assert "nested lists" in SYNTHESIS_SYSTEM_PROMPT


def test_synthesis_orders_chunks_by_source_position_within_each_paper():
    sources = format_sources_section(
        [
            {
                "paper_id": "p1",
                "chunk_id": "p1:10",
                "chunk_index": 10,
                "text": "later chunk",
                "score": 0.95,
            },
            {
                "paper_id": "p1",
                "chunk_id": "p1:2",
                "chunk_index": 2,
                "text": "earlier chunk",
                "score": 0.70,
            },
        ],
        docs_meta={"p1": {"title": "Ordered paper"}},
    )

    assert sources.index("earlier chunk") < sources.index("later chunk")
    assert 'chunk_index="2"' in sources
    assert 'chunk_index="10"' in sources


def test_synthesis_recovers_source_order_from_legacy_chunk_ids():
    sources = format_sources_section([
        {
            "paper_id": "arxiv:1",
            "chunk_id": "arxiv:1:10",
            "text": "ten",
            "score": 1.0,
        },
        {
            "paper_id": "arxiv:1",
            "chunk_id": "arxiv:1:2",
            "text": "two",
            "score": 0.9,
        },
    ])

    assert sources.index("two") < sources.index("ten")


def test_insufficient_retrieval_verdict_removes_untrusted_chunks():
    class _Provider:
        def get_default_model(self):
            return "test-model"

        async def chat_with_retry(self, **kwargs):
            return SimpleNamespace(content='{"quality":"insufficient"}')

    class _KB:
        config = SimpleNamespace(use_hybrid_retrieval=True)

        async def retrieve_by_hypothetical_questions(self, **kwargs):
            return [{
                "chunk_id": "p1:0",
                "paper_id": "p1",
                "paper_title": "Unrelated paper",
                "text": "Evidence unrelated to reinforcement learning.",
                "score": 1.0,
                "score_type": "weighted_rrf",
            }]

        def get_embedding_status(self):
            return {"backend": "test", "degraded": False, "reason": ""}

    nodes = AgentNodes(provider=_Provider(), kb=_KB(), tools={})
    state = {
        "user_query": "有没有强化学习相关的文章",
        "rewritten_queries": ["强化学习相关论文"],
        "extracted_entities": [],
    }

    result = asyncio.run(nodes.retrieval_node(state))

    assert result["retrieval_quality"] == "insufficient"
    assert result["retrieval_results"] == []


def test_citation_validation_ignores_markdown_links_and_flags_unknown_paper_ids():
    citations, invalid = collect_answer_citations(
        "Supported [2401.12345], [documentation](https://example.org), "
        "but fabricated [2501.99999].",
        external_papers=[{
            "paper_id": "2401.12345",
            "title": "An External Paper",
            "url": "https://arxiv.org/abs/2401.12345",
        }],
    )
    assert len(citations) == 1
    assert "2401.12345" in citations[0]
    assert invalid == ["2501.99999"]


def test_rewrite_degradation_does_not_compare_normalized_rrf_as_cosine():
    original = [{"chunk_id": "p1:0", "score": 1.0, "score_type": "weighted_rrf"}]
    rewritten = [{"chunk_id": "p2:0", "score": 0.2, "score_type": "weighted_rrf"}]

    assert AgentNodes._check_rewrite_degradation(original, rewritten) is False


def test_query_variants_keep_source_language_and_deduplicate_cleaned_forms():
    variants = AgentNodes._merge_query_variants(
        "帮我找 中文论文检索？",
        ["Chinese academic paper retrieval", "中文论文检索"],
    )

    assert variants == [
        "帮我找 中文论文检索？",
        "中文论文检索",
        "Chinese academic paper retrieval",
    ]


def test_external_query_selection_prefers_non_cjk_translation_when_available():
    assert AgentNodes._select_external_queries([
        "中文论文检索",
        "Chinese academic paper retrieval",
    ]) == ["Chinese academic paper retrieval"]
    assert AgentNodes._select_external_queries(["中文论文检索"]) == ["中文论文检索"]


def test_external_time_filters_normalize_exact_relative_and_structured_ranges():
    assert AgentNodes._resolve_external_time_range(
        ["2024"], "查找 2024 年论文", current_year=2026
    ) == (2024, 2024)
    assert AgentNodes._resolve_external_time_range(
        ["last 2 years"], "latest papers", current_year=2026
    ) == (2025, 2026)
    assert AgentNodes._resolve_external_time_range(
        [{"from_year": 2020, "to_year": 2023}], "papers", current_year=2026
    ) == (2020, 2023)


def test_explicit_external_id_is_a_deterministic_route():
    class _Provider:
        def get_default_model(self):
            return "test-model"

        async def chat_with_retry(self, **kwargs):
            raise AssertionError("route LLM must not run for explicit external lookup")

    nodes = AgentNodes(provider=_Provider(), kb=SimpleNamespace(), tools={})
    state = {"user_query": "去外部搜索2511.14460v2这篇论文"}

    result = asyncio.run(nodes.router_node(state))

    assert result["routing_decision"] == "external"
    assert result["external_search_requested"] is True
    assert result["external_search_authorized"] is True
    assert result["explicit_paper_ids"] == ["2511.14460v2"]

    prepared = asyncio.run(nodes._prepare_queries(state["user_query"], result))
    assert "2511.14460v2" in prepared
    assert result["rewrite_reasoning"] == "deterministic exact arXiv ID lookup"


def test_research_pauses_before_unrequested_external_search():
    class _Provider:
        def get_default_model(self):
            return "test-model"

    class _SearchTool:
        async def execute(self, **kwargs):
            raise AssertionError("external search must not run before confirmation")

    nodes = AgentNodes(
        provider=_Provider(),
        kb=SimpleNamespace(),
        tools={"paper_search": _SearchTool()},
    )
    result = asyncio.run(nodes._research_search_phase({
        "user_query": "详细解读这篇论文",
        "external_search_requested": False,
        "external_search_authorized": False,
    }))

    assert result["research_phase"] == "confirm_search"
    assert result["awaiting_external_search_confirmation"] is True
    assert "是否允许" in result["final_answer"]


def test_resume_external_search_confirmation_controls_research_phase():
    graph = object.__new__(MultiAgentGraph)
    graph.nodes = MagicMock()
    graph.nodes.bind_progress_callback.return_value = None
    graph.graph = MagicMock()
    graph.graph.ainvoke = AsyncMock(side_effect=lambda state: state)
    saved = {
        "user_query": "详细解读这篇论文",
        "session_id": "test",
        "research_phase": "confirm_search",
        "retrieval_results": [],
    }

    authorized = asyncio.run(graph.resume(saved, "可以"))

    assert authorized["external_search_authorized"] is True
    assert authorized["research_phase"] == "search"
    assert authorized["resume_phase"] == "search"

    declined = asyncio.run(graph.resume(saved, "只使用知识库"))

    assert declined["external_search_authorized"] is False
    assert declined["research_phase"] == "complete"
    assert declined["external_search_completed"] is True


def test_novelty_retrieval_excludes_previously_presented_papers():
    class _Provider:
        def get_default_model(self):
            return "test-model"

    class _KB:
        config = SimpleNamespace(use_hybrid_retrieval=True)

        def __init__(self):
            self.last_kwargs = None

        async def retrieve_by_hypothetical_questions(self, **kwargs):
            self.last_kwargs = kwargs
            return [
                {
                    "chunk_id": "2401.00001v2:0",
                    "paper_id": "2401.00001v2",
                    "text": "old paper",
                    "score": 0.95,
                    "score_type": "cross_encoder_relevance",
                },
                {
                    "chunk_id": "2501.00002v1:0",
                    "paper_id": "2501.00002v1",
                    "text": "new paper",
                    "score": 0.90,
                    "score_type": "cross_encoder_relevance",
                },
            ]

        def get_embedding_status(self):
            return {"backend": "test", "degraded": False, "reason": ""}

    kb = _KB()
    nodes = AgentNodes(provider=_Provider(), kb=kb, tools={}, top_k=5)
    state = {
        "user_query": "还有没有其他相关的论文？",
        "last_search_topic": "reinforcement learning",
        "presented_paper_ids": ["2401.00001v1"],
        "rewritten_queries": ["reinforcement learning"],
        "extracted_entities": [{"paper_id": "2401.00001v2", "title": "Old"}],
    }

    result = asyncio.run(nodes.retrieval_node(state))

    assert result["novelty_required"] is True
    assert result["extracted_entities"] == []
    assert [item["paper_id"] for item in result["retrieval_results"]] == [
        "2501.00002v1"
    ]
    assert result["retrieval_quality"] == "sufficient"
    assert kb.last_kwargs["top_k"] == 20


def test_novelty_retrieval_with_only_old_paper_is_insufficient():
    class _Provider:
        def get_default_model(self):
            return "test-model"

    class _KB:
        config = SimpleNamespace(use_hybrid_retrieval=True)

        async def retrieve_by_hypothetical_questions(self, **kwargs):
            return [{
                "chunk_id": "2401.00001v2:0",
                "paper_id": "2401.00001v2",
                "text": "old paper",
                "score": 0.99,
                "score_type": "cross_encoder_relevance",
            }]

        def get_embedding_status(self):
            return {"backend": "test", "degraded": False, "reason": ""}

    nodes = AgentNodes(provider=_Provider(), kb=_KB(), tools={})
    state = {
        "user_query": "还有更多论文吗？",
        "last_search_topic": "reinforcement learning",
        "presented_paper_ids": ["2401.00001v1"],
        "rewritten_queries": ["reinforcement learning"],
        "extracted_entities": [],
    }

    result = asyncio.run(nodes.retrieval_node(state))

    assert result["retrieval_results"] == []
    assert result["retrieval_quality"] == "insufficient"


def test_research_excludes_presented_not_every_paper_already_in_kb():
    class _Provider:
        def get_default_model(self):
            return "test-model"

    class _KB:
        def load_docs_meta(self):
            return {"kb-only": {"title": "Known but not shown"}}

    class _SearchTool:
        def __init__(self):
            self.kwargs = None

        async def execute(self, **kwargs):
            self.kwargs = kwargs
            return '{"search_status":"ok","results":[{"paper_id":"new-1","title":"New","abstract":"new evidence"}]}'

    search_tool = _SearchTool()
    nodes = AgentNodes(
        provider=_Provider(),
        kb=_KB(),
        tools={"paper_search": search_tool},
    )
    state = {
        "user_query": "还有没有其他相关论文？",
        "last_search_topic": "reinforcement learning",
        "presented_paper_ids": ["shown-1"],
        "rewritten_queries": ["reinforcement learning"],
        "sub_queries_detail": [],
        "external_search_top_k": 20,
        "external_rerank_top_k": 5,
        "external_search_authorized": True,
    }

    result = asyncio.run(nodes._research_search_phase(state))

    assert search_tool.kwargs["exclude_paper_ids"] == ["shown-1"]
    assert "kb-only" not in search_tool.kwargs["exclude_paper_ids"]
    assert result["research_outcome"] == "found"
    assert result["research_phase"] == "select"


def test_research_preserves_rate_limit_as_provider_error():
    class _Provider:
        def get_default_model(self):
            return "test-model"

    class _KB:
        def load_docs_meta(self):
            return {}

    class _SearchTool:
        async def execute(self, **kwargs):
            return (
                '{"search_status":"rate_limited","retry_after_seconds":17,'
                '"results":[]}'
            )

    progress = []
    nodes = AgentNodes(
        provider=_Provider(),
        kb=_KB(),
        tools={"paper_search": _SearchTool()},
        progress_callback=progress.append,
    )
    state = {
        "user_query": "latest EEG papers",
        "rewritten_queries": ["EEG image reconstruction"],
        "sub_queries_detail": [],
        "external_search_top_k": 20,
        "external_rerank_top_k": 5,
    }

    result = asyncio.run(nodes._research_search_phase(state))

    assert result["research_outcome"] == "provider_error"
    assert result["external_retry_after_seconds"] == 17
    assert "not evidence" in result["error_message"]
    assert any("17" in message and "限流" in message for message in progress)


def test_synthesis_uses_configured_model_and_recovers_length_limit():
    class _Provider:
        def __init__(self):
            self.calls = []
            self.responses = [
                SimpleNamespace(content="first ", finish_reason="length"),
                SimpleNamespace(content="second", finish_reason="stop"),
            ]

        def get_default_model(self):
            return "provider-default"

        async def chat_with_retry(self, **kwargs):
            self.calls.append(kwargs)
            return self.responses.pop(0)

    class _KB:
        def load_docs_meta(self):
            return {}

    provider = _Provider()
    nodes = AgentNodes(
        provider=provider,
        kb=_KB(),
        tools={},
        model="configured-model",
        provider_retry_mode="persistent",
        context_window_tokens=4096,
        max_completion_tokens=512,
    )
    state = {
        "user_query": "Explain this method",
        "routing_decision": "direct",
        "retrieval_results": [],
        "external_papers": [],
    }

    result = asyncio.run(nodes.synthesis_node(state))

    assert result["draft_answer"] == "first second"
    assert len(provider.calls) == 2
    assert all(call["model"] == "configured-model" for call in provider.calls)
    assert all(call["retry_mode"] == "persistent" for call in provider.calls)
    assert all(call["max_tokens"] == 512 for call in provider.calls)
    assert "Output limit reached" in provider.calls[1]["messages"][-1]["content"]


def test_synthesis_retries_empty_visible_response_without_thinking():
    class _Provider:
        def __init__(self):
            self.calls = []
            self.responses = [
                SimpleNamespace(content="", finish_reason="stop"),
                SimpleNamespace(content="recovered answer", finish_reason="stop"),
            ]

        def get_default_model(self):
            return "test-model"

        async def chat_with_retry(self, **kwargs):
            self.calls.append(kwargs)
            return self.responses.pop(0)

    class _KB:
        def load_docs_meta(self):
            return {}

    provider = _Provider()
    nodes = AgentNodes(provider=provider, kb=_KB(), tools={})

    result = asyncio.run(nodes.synthesis_node({
        "user_query": "Explain this method",
        "routing_decision": "direct",
        "retrieval_results": [],
        "external_papers": [],
    }))

    assert result["draft_answer"] == "recovered answer"
    assert provider.calls[1]["disable_thinking"] is True
    assert "Provide the final response" in str(provider.calls[1]["messages"][-1]["content"])


def test_research_only_returns_to_retrieval_after_successful_ingest():
    assert research_phase_conditional({
        "research_phase": "confirm_search",
    }) == "wait_for_search_confirmation"
    assert research_phase_conditional({
        "research_phase": "complete",
        "research_outcome": "no_new_results",
        "post_research_retrieval": False,
    }) == "to_synthesis"
    assert research_phase_conditional({
        "research_phase": "complete",
        "research_outcome": "provider_error",
        "post_research_retrieval": False,
    }) == "to_synthesis"
    assert research_phase_conditional({
        "research_phase": "complete",
        "research_outcome": "ingested",
        "post_research_retrieval": True,
    }) == "to_retrieval"


def test_no_new_results_use_deterministic_synthesis_and_critic():
    class _Provider:
        def get_default_model(self):
            return "test-model"

        async def chat_with_retry(self, **kwargs):
            raise AssertionError("terminal no-result outcome must not call the LLM")

    nodes = AgentNodes(provider=_Provider(), kb=SimpleNamespace(), tools={})
    state = {
        "user_query": "还有没有其他相关论文？",
        "research_phase": "complete",
        "research_outcome": "no_new_results",
        "novelty_required": True,
        "resolved_topic": "reinforcement learning",
        "retrieval_results": [],
        "external_papers": [],
    }

    result = asyncio.run(nodes.synthesis_node(state))
    result = asyncio.run(nodes.critic_node(result))

    assert "没有发现新的" in result["final_answer"]
    assert result["critic_verdict"] == "passed"
    assert result["is_complete"] is True
