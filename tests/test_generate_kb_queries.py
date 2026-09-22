"""Tests for automatic query generation from KB chunks."""

import pytest

from scripts.generate_kb_queries import (
    is_near_duplicate,
    parse_generated_query,
    select_source_chunks,
)


def test_generated_question_requires_verbatim_support():
    text = "Adaptive CSLS uses local density to reduce hubness in image retrieval."
    generated = parse_generated_query(
        '{"method_name":"Adaptive CSLS",'
        '"question":"Adaptive CSLS 如何缓解图像检索中的 hubness？",'
        '"answer":"利用局部密度调整相似度",'
        '"quote":"uses local density to reduce hubness"}',
        text,
    )
    assert generated["query"].endswith("？")
    with pytest.raises(ValueError, match="exact substring"):
        parse_generated_query(
            '{"method_name":"Adaptive CSLS",'
            '"question":"Adaptive CSLS 如何缓解图像检索中的 hubness？",'
            '"answer":"利用局部密度", "quote":"uses global density"}',
            text,
        )


def test_generated_question_rejects_non_self_contained_and_duplicates():
    text = "A long passage about a method uses local density to reduce hubness."
    with pytest.raises(ValueError, match="self-contained"):
        parse_generated_query(
            '{"method_name":"local density",'
            '"question":"这篇论文怎样缓解检索中的 hubness？",'
            '"answer":"通过局部密度", "quote":"uses local density to reduce hubness"}',
            text,
        )
    assert is_near_duplicate("CSLS 如何消除 hubness？", ["csls如何消除hubness"])
    with pytest.raises(ValueError, match="Chinese"):
        parse_generated_query(
            '{"method_name":"local density",'
            '"question":"How does adaptive CSLS reduce hubness in image retrieval?",'
            '"answer":"通过局部密度", "quote":"uses local density to reduce hubness"}',
            text,
        )
    with pytest.raises(ValueError, match="too generic"):
        parse_generated_query(
            '{"method_name":"EEG encoder",'
            '"question":"EEG encoder 的主要功能是什么？",'
            '"answer":"提取 EEG 表征", "quote":"uses local density to reduce hubness"}',
            text + " EEG encoder",
        )


def test_source_selection_interleaves_papers_and_excludes_background():
    chunks = {
        "a:0": {"chunk_id": "a:0", "paper_id": "a", "section": "3 Method", "text": "x" * 400},
        "a:1": {"chunk_id": "a:1", "paper_id": "a", "section": "3.1 Encoder", "text": "y" * 400},
        "a:2": {"chunk_id": "a:2", "paper_id": "a", "section": "Introduction", "text": "z" * 400},
        "b:0": {"chunk_id": "b:0", "paper_id": "b", "section": "4 Results", "text": "w" * 400},
    }
    selected = select_source_chunks(chunks, per_paper_candidates=2)
    assert {selected[0]["paper_id"], selected[1]["paper_id"]} == {"a", "b"}
    assert [row["chunk_id"] for row in selected].count("a:2") == 0
