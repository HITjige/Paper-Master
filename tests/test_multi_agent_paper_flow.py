"""Focused regression tests for paper multi-agent routing and source handling."""

from nanobot.agent.multi_agent.agents import (
    collect_answer_citations,
    format_sources_section,
)
from nanobot.agent.multi_agent.conditions import retrieval_conditional
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
