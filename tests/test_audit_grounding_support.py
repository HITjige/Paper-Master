"""Checks that final support labels retain source and entailment constraints."""

import asyncio
import json

from scripts.audit_grounding_support import _audit_batch, _source_title, _unbound_entities


def test_source_title_finds_paper_for_json_and_xml_passages():
    json_section = json.dumps({"papers": [{"title": "Paper A", "chunks": [
        {"text": "Paper A reports 20%."},
    ]}]})
    xml_section = ("<sources><paper id=\"p2\"><metadata><title>Paper B</title></metadata>"
                   "<chunk>Paper B reports 30%.</chunk></paper></sources>")
    assert _source_title([json_section], "Paper A reports 20%.") == "Paper A"
    assert _source_title([xml_section], "Paper B reports 30%.") == "Paper B"


def test_audit_defers_absence_claim_even_if_model_accepts(monkeypatch):
    async def fake_model(_client, **_kwargs):
        return {"judgments": [{"index": 0, "verdict": "supported"}]}

    monkeypatch.setattr("scripts.audit_grounding_support._model_json", fake_model)
    section = json.dumps({"papers": [{"title": "Paper A", "chunks": [
        {"text": "This paper uses an image encoder."},
    ]}]})
    result = asyncio.run(_audit_batch(
        object(), endpoint=("judge", "http://localhost/v1", "EMPTY"),
        pairs=[({"query": "Does Paper A use contrastive learning?", "evidence": [section]},
                {"claim": "论文未明确使用对比学习术语。"},
                {"run_id": "r", "index": 0, "status": "supported",
                 "quote": "This paper uses an image encoder."})],
        disable_thinking=False,
    ))
    assert result[0]["status"] == "needs_review"
    assert result[0]["candidate_status"] == "supported"


def test_external_method_number_requires_name_in_passage_or_source():
    claim = "ATM 的 Top-5 准确率为 79.7%。"
    passage = "Our method achieves top-5 accuracy of 79.7%."
    assert _unbound_entities(claim, passage, "Uncertainty-Aware Blur Prior") == ["ATM"]
    assert _unbound_entities(claim, passage, "ATM: EEG Embedding") == []
