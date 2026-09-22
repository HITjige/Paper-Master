"""Tests for the offline, single-query KB evaluation prototype."""

import asyncio
import json
from argparse import Namespace

import pytest

from nanobot.agent.paper_kb import PaperKbConfig
from scripts.eval_kb import (
    corpus_manifest,
    evaluate,
    load_queries,
    metrics_at_k,
    parse_judge_response,
    ranked_pool_sources,
    same_corpus,
    supplement_pool,
)


def test_metrics_use_all_relevant_chunks_and_deduplicate_predictions():
    result = metrics_at_k(["a", "a", "b"], {"a", "b", "c"}, 3)
    assert result == {"recall": 2 / 3, "hit": 1.0, "mrr": 1.0}


def test_metrics_no_hit_and_empty_gold():
    assert metrics_at_k(["x", "y"], {"z"}, 2) == {
        "recall": 0.0, "hit": 0.0, "mrr": 0.0,
    }
    with pytest.raises(ValueError, match="at least one"):
        metrics_at_k(["x"], set(), 1)


def test_ranked_pool_sources_ignores_seed_and_validates_ranks():
    rows = [
        {"chunk_id": "a", "sources": {"hybrid": 2, "dense": 1, "seed": 0}},
        {"chunk_id": "b", "sources": {"hybrid": 1, "bm25": 2}},
        {"chunk_id": "c", "sources": {"bm25": 1, "seed": 0}},
    ]
    assert ranked_pool_sources(rows) == {
        "hybrid": ["b", "a"], "dense": ["a"], "bm25": ["c", "b"],
    }
    with pytest.raises(ValueError, match="Duplicate hybrid"):
        ranked_pool_sources([rows[0], {"chunk_id": "b", "sources": {"hybrid": 2}}])


def test_grade_two_requires_verbatim_evidence():
    text = "Adaptive CSLS reduces hubness in EEG-to-image retrieval."
    accepted = parse_judge_response(
        '{"grade":2,"quote":"reduces hubness","reason":"direct support"}', text,
    )
    assert accepted["status"] == "accepted"
    review = parse_judge_response(
        '{"grade":2,"quote":"improves accuracy","reason":"unsupported"}', text,
    )
    assert review["status"] == "needs_review"
    assert parse_judge_response('{"grade":1,"quote":"","reason":"topic"}', text)["status"] == "accepted"


def test_query_seed_must_exist_and_corpus_must_match(tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "documents.jsonl").write_text('{"paper_id":"p"}\n', encoding="utf-8")
    (kb_dir / "chunks.jsonl").write_text('{"chunk_id":"p:0"}\n', encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({
        "id": "q", "query": "what?", "seed_chunk_ids": ["p:1"],
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed chunk missing"):
        load_queries(queries, {"p:0": {"chunk_id": "p:0"}})
    with pytest.raises(ValueError, match="differ"):
        same_corpus({"corpus": {}}, tmp_path)


def test_supplement_pool_adds_only_unjudged_predictions(tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "documents.jsonl").write_text('{"paper_id":"p"}\n', encoding="utf-8")
    (kb_dir / "chunks.jsonl").write_text(
        '{"chunk_id":"p:0","text":"first"}\n'
        '{"chunk_id":"p:1","text":"second"}\n',
        encoding="utf-8",
    )
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps({
        "corpus": corpus_manifest(tmp_path), "queries": [{"id": "q", "query": "why?"}],
        "candidates": [{"query_id": "q", "chunk_id": "p:0"}],
    }), encoding="utf-8")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({
        "corpus": corpus_manifest(tmp_path),
        "queries": [{"id": "q", "unjudged": ["p:1", "p:1"]}],
    }), encoding="utf-8")
    supplement_pool(Namespace(
        workspace=tmp_path, pool=pool_path, report=report_path, output=pool_path,
    ))
    candidates = json.loads(pool_path.read_text(encoding="utf-8"))["candidates"]
    assert len(candidates) == 2
    assert candidates[-1]["sources"] == {"evaluation_unjudged": 0}


def test_evaluate_compares_pooled_retrieval_sources_on_same_query(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "documents.jsonl").write_text('{"paper_id":"p"}\n', encoding="utf-8")
    (kb_dir / "chunks.jsonl").write_text(
        ''.join(json.dumps({"chunk_id": cid, "paper_id": "p", "text": cid}) + "\n"
                for cid in ("a", "b", "c")),
        encoding="utf-8",
    )
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps({
        "corpus": corpus_manifest(tmp_path), "depth_per_source": 2,
        "pool_config": {"use_hybrid_retrieval": True},
        "queries": [{"id": "q", "query": "why?", "tags": ["test"]}],
        "candidates": [
            {"query_id": "q", "chunk_id": "a", "sources": {"hybrid": 1, "dense": 2}},
            {"query_id": "q", "chunk_id": "b", "sources": {"hybrid": 2, "dense": 1, "bm25": 2}},
            {"query_id": "q", "chunk_id": "c", "sources": {"bm25": 1}},
        ],
    }), encoding="utf-8")
    judgments_path = tmp_path / "judgments.jsonl"
    judgments_path.write_text(
        ''.join(json.dumps({"query_id": "q", "chunk_id": cid, "grade": grade,
                            "status": "accepted"}) + "\n"
                for cid, grade in (("a", 2), ("b", 0), ("c", 0))),
        encoding="utf-8",
    )

    class FakeKb:
        def __init__(self, workspace, config):
            pass

        def get_embedding_status(self):
            return {"degraded": False}

        def get_lexical_status(self):
            return {"degraded": False}

        async def retrieve_by_hypothetical_questions(self, **kwargs):
            return [{"chunk_id": "a"}, {"chunk_id": "b"}]

    monkeypatch.setattr("scripts.eval_kb.PaperKnowledgeBase", FakeKb)
    monkeypatch.setattr("scripts.eval_kb.kb_config", lambda config: PaperKbConfig())
    report_path = tmp_path / "report.json"
    asyncio.run(evaluate(Namespace(
        workspace=tmp_path, pool=pool_path, judgments=judgments_path,
        overrides=None, config=None, ks="1,2", output=report_path, allow_degraded=False,
    )))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    methods = report["summary"]["source_methods"]
    assert methods["hybrid"]["metrics"]["1"]["recall"] == 1.0
    assert methods["dense"]["metrics"]["1"]["recall"] == 0.0
    assert methods["bm25"]["metrics"]["1"]["recall"] == 0.0
    assert all(methods[source]["count"] == 1 for source in methods)
    assert report["queries"][0]["source_methods"]["bm25"]["predicted"] == ["c", "b"]

    source_report_path = tmp_path / "source_report.json"
    asyncio.run(evaluate(Namespace(
        workspace=tmp_path, pool=pool_path, judgments=judgments_path,
        overrides=None, config=None, ks="1,2", output=source_report_path,
        allow_degraded=False, source_only=True,
    )))
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    assert source_report["summary"]["metrics"] == {}
    assert source_report["summary"]["source_methods"] == methods
    assert source_report["evaluation_mode"] == "pool_sources_only"
