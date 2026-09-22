"""Checks for constrained automatic review of unresolved grounding claims."""

import asyncio
import json
from types import SimpleNamespace

from scripts.auto_review_grounding import (
    _conservative_reviews,
    _review_batch,
    _visible_passages,
    command,
)
from scripts.eval_agent_hallucination import JUDGE_VERSION, _checked_reviews, _contains_quote
from scripts.eval_kb import corpus_manifest, read_jsonl


def test_visible_passages_preserve_exact_source_text():
    json_section = json.dumps({"papers": [{"title": "Paper", "chunks": [
        {"text": "The encoder has two layers.\nIt uses attention."},
    ]}]}, ensure_ascii=False)
    xml_section = '<sources><chunk section="methods">The model uses attention.</chunk></sources>'
    for section in (json_section, xml_section):
        passages = _visible_passages([section])
        assert passages
        assert all(_contains_quote(section, passage["quote"]) for passage in passages)


def test_two_pass_review_requires_agreement(monkeypatch):
    async def selected(_client, **kwargs):
        if "passages" in kwargs["user"]:
            return {"judgments": [{"index": 0, "verdict": "supported", "passage_id": "p0"}]}
        return {"judgments": [{"index": 0, "verdict": "supported"}]}

    monkeypatch.setattr("scripts.auto_review_grounding._model_json", selected)
    result = asyncio.run(_review_batch(
        object(), endpoint=("judge", "http://localhost/v1", "EMPTY"),
        run={"query": "How many layers?"},
        claims=[{"index": 4, "claim": "The encoder has two layers"}],
        passages=[{"id": "p0", "text": "The encoder has two layers.",
                   "quote": "The encoder has two layers."}], disable_thinking=False,
    ))
    assert result[0]["status"] == "supported"
    assert result[0]["index"] == 4

    async def disagreed(_client, **kwargs):
        if "passages" in kwargs["user"]:
            return {"judgments": [{"index": 0, "verdict": "supported", "passage_id": "p0"}]}
        return {"judgments": [{"index": 0, "verdict": "insufficient"}]}

    monkeypatch.setattr("scripts.auto_review_grounding._model_json", disagreed)
    result = asyncio.run(_review_batch(
        object(), endpoint=("judge", "http://localhost/v1", "EMPTY"),
        run={"query": "How many layers?"},
        claims=[{"index": 4, "claim": "The encoder has two layers"}],
        passages=[{"id": "p0", "text": "The encoder has two layers.",
                   "quote": "The encoder has two layers."}], disable_thinking=False,
    ))
    assert result[0]["status"] == "needs_review"


def test_command_writes_resumable_reviews_accepted_by_report(tmp_path, monkeypatch):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "documents.jsonl").write_text('{"paper_id":"p"}\n')
    (kb / "chunks.jsonl").write_text('{"chunk_id":"p:1","text":"The encoder has two layers."}\n')
    corpus = corpus_manifest(tmp_path)
    evidence = [json.dumps({"papers": [{"title": "Paper", "chunks": [
        {"text": "The encoder has two layers."},
    ]}]})]
    run = {"run_id": "workflow:q:0:single", "query": "How many layers?",
           "answer": "Two layers", "status": "completed", "evidence": evidence,
           "corpus": corpus}
    judgment = {"run_id": run["run_id"], "corpus": corpus,
                "judge_version": JUDGE_VERSION,
                "claims": [{"index": 0, "claim": "The encoder has two layers",
                            "status": "needs_review"}]}
    runs_path, judgments_path, output = (tmp_path / name for name in
                                         ("runs.jsonl", "judgments.jsonl", "reviews.jsonl"))
    runs_path.write_text(json.dumps(run) + "\n")
    judgments_path.write_text(json.dumps(judgment) + "\n")

    async def fake_model(_client, **kwargs):
        if "passages" in kwargs["user"]:
            return {"judgments": [{"index": 0, "verdict": "supported", "passage_id": "p1"}]}
        return {"judgments": [{"index": 0, "verdict": "supported"}]}

    monkeypatch.setattr("scripts.auto_review_grounding._model_json", fake_model)
    monkeypatch.setattr("scripts.auto_review_grounding._judge_endpoint",
                        lambda _args: ("judge", "http://localhost/v1", "EMPTY"))
    args = SimpleNamespace(workspace=tmp_path, runs=runs_path, judgments=judgments_path,
                           output=output, limit=0, batch_size=4, concurrency=1,
                           timeout=5, disable_thinking=False)
    asyncio.run(command(args))
    asyncio.run(command(args))
    reviews = read_jsonl(output)
    assert len(reviews) == 1
    assert reviews[0]["status"] == "supported"
    assert _checked_reviews(output, [run])[(run["run_id"], 0)]["status"] == "supported"


def test_conservative_reviews_defer_contradictions_and_unbound_years():
    rows = [
        {"status": "supported", "quote": "The encoder has two layers."},
        {"status": "contradicted", "quote": "Another experiment has 50% accuracy."},
        {"status": "supported", "quote": '"year": 2017'},
    ]
    safe = _conservative_reviews(rows)
    assert [row["status"] for row in safe] == ["supported", "needs_review", "needs_review"]
    assert safe[1]["candidate_status"] == "contradicted"
    assert safe[1]["candidate_quote"] == rows[1]["quote"]
    assert safe[2]["candidate_status"] == "supported"
    assert rows[1]["status"] == "contradicted"
