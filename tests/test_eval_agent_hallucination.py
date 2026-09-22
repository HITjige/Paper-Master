"""Offline checks for single-turn agent evidence and answer evaluation."""

import asyncio
import json
from hashlib import sha256
from types import SimpleNamespace

import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.bus.events import OutboundMessage
from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from scripts.eval_agent_hallucination import (
    EvidenceHook,
    _fixed_results,
    _hash,
    _judge_run,
    _one_run,
    _run_fixed,
    _checked_reviews,
    _checked_answer_reviews,
    _judge_answer_run,
    ANSWER_JUDGE_VERSION,
    ANSWER_SYSTEM,
    build_report,
    judge_answers_command,
    JUDGE_VERSION,
    load_answer_rubrics,
    parse_claim_judgment,
)


def test_claim_judge_checks_visible_quote_and_marks_absence_unsupported():
    exposed = ['{"text":"The encoder has two layers."}']
    accepted = parse_claim_judgment(
        {"verdict": "supported", "quote": "two layers"},
        "The encoder has two layers", exposed)
    assert accepted["status"] == "supported"

    invalid = parse_claim_judgment(
        {"verdict": "supported", "quote": "three layers"},
        "The encoder has three layers", exposed)
    assert invalid["status"] == "needs_review"

    absent = parse_claim_judgment(
        {"verdict": "insufficient", "quote": ""}, "An unsupported claim", exposed)
    assert absent["status"] == "unsupported"


def test_fixed_evidence_rejects_missing_chunks_and_overlong_selection():
    assert _fixed_results(["p:1"], {"p:1": {"text": "evidence"}})[0]["score"] == 1.0
    with pytest.raises(ValueError, match="absent"):
        _fixed_results(["p:2"], {"p:1": {"text": "evidence"}})
    with pytest.raises(ValueError, match="visible result limit"):
        _fixed_results(["p:1"] * 11, {"p:1": {"text": "evidence"}})


def test_evidence_hook_captures_normalized_tool_message_once():
    hook = EvidenceHook()
    context = AgentHookContext(
        iteration=1,
        messages=[{"role": "tool", "tool_call_id": "one", "name": "kb_retrieve",
                   "content": '{"papers":[{"chunks":[{"text":"fact"}]}]}'},
                  {"role": "tool", "tool_call_id": "two", "name": "other",
                   "content": "ignored"}],
        tool_calls=[ToolCallRequest(id="one", name="kb_retrieve", arguments={})],
        stop_reason="completed",
    )
    asyncio.run(hook.after_iteration(context))
    asyncio.run(hook.after_iteration(context))
    assert len(hook.evidence) == 1
    assert hook.tool_calls == [{"name": "kb_retrieve", "arguments": {}}]
    assert hook.stop_reason == "completed"


def test_fixed_single_injects_standard_tool_result_and_disables_retrieval():
    class FakeTools:
        def __init__(self):
            self.removed = []

        def unregister(self, name):
            self.removed.append(name)

    class FakeLoop:
        def __init__(self):
            self.kb = SimpleNamespace(load_docs_meta=lambda: {"p": {"title": "Paper"}})
            self.context = SimpleNamespace(build_messages=lambda **kwargs: [
                {"role": "system", "content": "answer from evidence"},
                {"role": "user", "content": kwargs["current_message"]},
            ])
            self.tools = FakeTools()

        async def _run_agent_loop(self, messages):
            assert messages[-2]["tool_calls"][0]["function"]["name"] == "kb_retrieve"
            assert messages[-1]["role"] == "tool"
            assert "two layers" in messages[-1]["content"]
            return "Two layers.", [], messages, "completed", False

    loop, hook = FakeLoop(), EvidenceHook()
    answer, status, _details = asyncio.run(_run_fixed(
        loop, "single", "How many layers?",
        [{"chunk_id": "p:1", "paper_id": "p", "text": "two layers", "score": 1.0}],
        hook,
    ))
    assert (answer, status) == ("Two layers.", "completed")
    assert loop.tools.removed == ["kb_retrieve"]
    assert len(hook.evidence) == 1


def _run(mode, status="completed"):
    return {"run_id": f"workflow:q:0:{mode}", "query_id": "q", "mode": mode,
            "repeat": 0, "status": status, "elapsed_s": 2.0,
            "query": "Question?", "answer": "a fact",
            "corpus": {"chunks.jsonl": "x"}, "protocol": "workflow",
            "tags": ["method"]}


def _judgment(mode, status):
    return {"run_id": f"workflow:q:0:{mode}", "corpus": {"chunks.jsonl": "x"},
            "query_sha256": sha256(b"Question?").hexdigest(),
            "answer_sha256": sha256(b"a fact").hexdigest(), "answer_type": "answer",
            "evidence_sha256": _hash([]), "judge_version": JUDGE_VERSION,
            "claims": [{"index": 0, "claim": "a fact", "status": status}]}


def test_report_excludes_pending_claims_until_reviewed_and_pairs_by_query():
    runs = [_run("single"), _run("multi")]
    judgments = [_judgment("single", "supported"), _judgment("multi", "needs_review")]
    pending = build_report(runs, judgments, {})
    assert pending["summary"]["single"]["query_ungrounded_rate"] == 0
    assert pending["summary"]["multi"]["query_ungrounded_rate"] is None
    assert pending["summary"]["multi"]["pending_review"] == 1
    assert pending["paired_query_difference"]["query_count"] == 0

    reviewed = build_report(runs, judgments, {
        ("workflow:q:0:multi", 0): {"status": "unsupported"},
    })
    assert reviewed["summary"]["multi"]["query_ungrounded_rate"] == 1
    assert reviewed["summary"]["multi"]["claim_ungrounded_rate"] == 1
    assert reviewed["paired_query_difference"]["difference"] == 1
    assert reviewed["by_tag"]["method"]["multi"]["query_ungrounded_rate"] == 1


def test_report_counts_pause_and_abstention_separately():
    runs = [_run("single"), _run("multi", "paused")]
    judgments = [{**_judgment("single", "supported"), "answer_type": "abstain", "claims": []}]
    result = build_report(runs, judgments, {})
    assert result["summary"]["single"]["abstained"] == 1
    assert result["summary"]["single"]["scored_answers"] == 0
    assert result["summary"]["multi"]["status_counts"] == {"paused": 1}


def test_workflow_run_marks_multi_agent_pause_without_followup(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "chunks.jsonl").write_text('{"chunk_id":"p:1","text":"fact"}\n')
    (kb_dir / "documents.jsonl").write_text('{"paper_id":"p"}\n')

    class FakeGraph:
        def __init__(self):
            self.nodes = SimpleNamespace(_bounded_sources_section=lambda section: section)

        async def run(self, *args, **kwargs):
            self.nodes._bounded_sources_section("visible source")
            return {"research_phase": "confirm_search", "routing_decision": "internal"}

    class FakeLoop:
        def __init__(self):
            self.kb = SimpleNamespace(get_embedding_status=lambda: {"degraded": False})
            self._multi_agent_graph = FakeGraph()

        async def process_with_multi_agent(self, *args, **kwargs):
            await self._multi_agent_graph.run(user_query=args[0])
            return OutboundMessage(channel="cli", chat_id="test", content="confirm?",
                                   metadata={"multi_agent": True})

        async def close_mcp(self):
            pass

    monkeypatch.setattr("scripts.eval_agent_hallucination._make_loop", lambda *_: FakeLoop())
    monkeypatch.setattr("scripts.eval_agent_hallucination.sync_workspace_templates", lambda *_args, **_kw: [])
    item = {"id": "q", "query": "what?", "seed_chunk_ids": ["p:1"]}
    config = SimpleNamespace(agents=SimpleNamespace(defaults=SimpleNamespace(
        model="fake", temperature=0.0, max_tokens=100, reasoning_effort=None,
    )))
    result = asyncio.run(_one_run(tmp_path, config, item, "multi", 0, "workflow", {}))
    assert result["status"] == "paused"
    assert result["evidence"] == ["visible source"]
    assert result["details"]["routing_decision"] == "internal"


def test_fixed_run_records_model_visible_evidence_and_usage(tmp_path, monkeypatch):
    class CannedProvider(LLMProvider):
        def get_default_model(self):
            return "canned"

        async def chat(self, **_kwargs):
            return LLMResponse(content="Two layers.", usage={
                "prompt_tokens": 10, "completion_tokens": 3,
            })

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    chunk = {"paper_id": "p", "chunk_id": "p:1", "text": "The encoder has two layers."}
    (kb_dir / "documents.jsonl").write_text(json.dumps({"paper_id": "p", "title": "Paper"}) + "\n")
    (kb_dir / "chunks.jsonl").write_text(json.dumps(chunk) + "\n")
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _cfg: CannedProvider())
    monkeypatch.setattr("scripts.eval_agent_hallucination.sync_workspace_templates", lambda *_a, **_k: [])
    config = Config()
    config.agents.defaults.model = "canned"
    config.agents.defaults.max_tool_iterations = 2
    config.tools.paper.embedding_model = "/missing/embedding-model"
    config.tools.paper.embedding_fallback = "hash"
    item = {"id": "q", "query": "How many layers?", "fixed_chunk_ids": ["p:1"]}
    result = asyncio.run(_one_run(
        tmp_path, config, item, "single", 0, "fixed", {"p:1": chunk},
        allow_degraded=True,
    ))
    assert result["status"] == "completed"
    assert result["answer"] == "Two layers."
    assert result["evidence_capture"] == "model_request"
    assert "The encoder has two layers." in result["evidence"][0]
    assert result["model_calls"] == 1
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 3}


def test_judge_batches_claims_against_full_visible_evidence(monkeypatch):
    calls = []

    async def fake_model_json(_client, **kwargs):
        if "answer" in kwargs["user"]:
            return {"answer_type": "answer", "claims": ["Two layers", "Three layers"]}
        calls.append(kwargs["user"])
        assert kwargs["user"]["exposed_evidence"] == ["x" * 20001 + " two layers"]
        return {"judgments": [
            {"index": 0, "verdict": "supported", "quote": "two layers"},
            {"index": 1, "verdict": "insufficient", "quote": ""},
        ]}

    monkeypatch.setattr("scripts.eval_agent_hallucination._model_json", fake_model_json)
    row = {"run_id": "fixed:q:0:single", "query": "How many layers?",
           "answer": "Two or three layers.", "corpus": {"chunks.jsonl": "x"},
           "evidence": ["x" * 20001 + " two layers"]}
    result = asyncio.run(_judge_run(
        row, object(), ("judge", "http://localhost/v1", "EMPTY"), False,
    ))
    assert result["answer_type"] == "answer"
    assert result["claims"][0]["status"] == "supported"
    assert result["claims"][1]["status"] == "unsupported"
    assert len(calls) == 1


def test_review_quotes_must_be_in_the_answering_agents_evidence(tmp_path):
    path = tmp_path / "reviews.jsonl"
    path.write_text(json.dumps({
        "run_id": "workflow:q:0:single", "index": 0, "status": "supported",
        "quote": "two layers",
    }) + "\n")
    run = {**_run("single"), "evidence": ["The encoder has two layers."]}
    assert _checked_reviews(path, [run])[(run["run_id"], 0)]["status"] == "supported"
    run["evidence"] = ["unrelated"]
    with pytest.raises(ValueError, match="exposed evidence"):
        _checked_reviews(path, [run])


def test_report_rejects_old_judgment_protocol():
    old = {**_judgment("single", "supported"), "judge_version": "agent-claims-v1"}
    with pytest.raises(ValueError, match="older protocol"):
        build_report([_run("single")], [old], {})


def test_report_requires_matching_visible_evidence():
    run = {**_run("single"), "evidence": ["different evidence"]}
    result = build_report([run], [_judgment("single", "supported")], {})
    assert result["summary"]["single"]["pending_review"] == 1
    assert result["summary"]["single"]["scored_answers"] == 0


def test_judge_needs_no_verification_call_without_visible_evidence(monkeypatch):
    calls = []

    async def fake_model_json(_client, **kwargs):
        calls.append(kwargs["user"])
        return {"answer_type": "answer", "claims": ["A fact"]}

    monkeypatch.setattr("scripts.eval_agent_hallucination._model_json", fake_model_json)
    row = {"run_id": "workflow:q:0:single", "query": "Question?", "answer": "A fact",
           "corpus": {"chunks.jsonl": "x"}, "evidence": []}
    result = asyncio.run(_judge_run(
        row, object(), ("judge", "http://localhost/v1", "EMPTY"), False,
    ))
    assert len(calls) == 1
    assert result["claims"][0]["status"] == "unsupported"


def _rubric():
    return {"query_id": "q", "query": "Question?", "answerability": "answerable",
            "required_points": [{"id": "p1", "text": "a fact", "chunk_id": "p:1",
                                 "quote": "a fact"}],
            "review_status": "reviewed", "reviewer": "human", "review_reason": "checked source"}


def _answer_judgment(mode, rubric, answer_result, coverage_rate):
    covered = answer_result == "full"
    return {"run_id": f"workflow:q:0:{mode}", "corpus": {"chunks.jsonl": "x"},
            "query_sha256": sha256(b"Question?").hexdigest(),
            "answer_sha256": sha256(b"a fact").hexdigest(),
            "rubric_sha256": _hash(rubric), "judge_version": ANSWER_JUDGE_VERSION,
            "prompt_sha256": sha256(ANSWER_SYSTEM.encode()).hexdigest(),
            "answer_type": "answer", "points": [
                {"id": "p1", "covered": covered, "quote": "a fact" if covered else ""},
            ],
            "answer_result": answer_result, "coverage_rate": coverage_rate}


def test_rubrics_require_review_and_source_quote(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "documents.jsonl").write_text('{"paper_id":"p"}\n')
    (kb / "chunks.jsonl").write_text('{"chunk_id":"p:1","text":"a fact"}\n')
    from scripts.eval_kb import corpus_manifest

    rubric = {**_rubric(), "corpus": corpus_manifest(tmp_path)}
    path = tmp_path / "rubrics.jsonl"
    path.write_text(json.dumps(rubric) + "\n")
    assert load_answer_rubrics(path, tmp_path, [_run("single")])["q"]["review_status"] == "reviewed"
    rubric["required_points"][0]["quote"] = "different text"
    path.write_text(json.dumps(rubric) + "\n")
    with pytest.raises(ValueError, match="absent from frozen KB"):
        load_answer_rubrics(path, tmp_path, [_run("single")])
    rubric["required_points"][0]["quote"] = "a fact"
    rubric["reviewer"] = ""
    path.write_text(json.dumps(rubric) + "\n")
    with pytest.raises(ValueError, match="requires reviewer"):
        load_answer_rubrics(path, tmp_path, [_run("single")])


def test_answer_judge_checks_coverage_and_answer_quotes(monkeypatch):
    async def fake_model_json(_client, **_kwargs):
        return {"answer_type": "answer", "points": [
            {"id": "p1", "covered": True, "quote": "a fact"},
        ]}

    monkeypatch.setattr("scripts.eval_agent_hallucination._model_json", fake_model_json)
    result = asyncio.run(_judge_answer_run(
        _run("single"), _rubric(), object(), ("judge", "http://localhost/v1", "EMPTY"), False,
    ))
    assert result["answer_result"] == "full"
    assert result["coverage_rate"] == 1

    async def bad_quote(_client, **_kwargs):
        return {"answer_type": "answer", "points": [
            {"id": "p1", "covered": True, "quote": "invented text"},
        ]}

    monkeypatch.setattr("scripts.eval_agent_hallucination._model_json", bad_quote)
    result = asyncio.run(_judge_answer_run(
        _run("single"), _rubric(), object(), ("judge", "http://localhost/v1", "EMPTY"), False,
    ))
    assert result["answer_result"] == "needs_review"


def test_answer_quality_and_latency_use_matched_completed_runs():
    runs = [_run("single"), {**_run("multi"), "elapsed_s": 4.0}]
    rubric = _rubric()
    result = build_report(
        runs, [_judgment("single", "supported"), _judgment("multi", "supported")], {},
        {"q": rubric}, [
            _answer_judgment("single", rubric, "full", 1.0),
            _answer_judgment("multi", rubric, "unanswered", 0.0),
        ],
    )
    assert result["summary"]["single"]["answer_quality"]["full_answer_rate"] == 1
    assert result["summary"]["multi"]["answer_quality"]["full_answer_rate"] == 0
    assert result["paired_task_success_difference"]["difference"] == -1
    assert result["paired_latency_difference_s"]["difference"] == 2
    assert result["summary"]["multi"]["completed_p95_elapsed_s"] == 4


def test_provisional_rubric_is_not_scored_and_manual_review_can_override(tmp_path):
    run = _run("single")
    rubric = _rubric()
    provisional = {**rubric, "review_status": "provisional"}
    result = build_report([run], [_judgment("single", "supported")], {}, {"q": provisional})
    assert result["summary"]["single"]["answer_quality"]["pending_review"] == 1
    assert result["summary"]["single"]["answer_quality"]["scored_answerable"] == 0

    path = tmp_path / "answer_reviews.jsonl"
    path.write_text(json.dumps({"run_id": run["run_id"], "answer_type": "answer",
                                "covered_points": [{"id": "p1", "quote": "a fact"}],
                                "reviewer": "human", "reason": "checked the answer"}) + "\n")
    reviews = _checked_answer_reviews(path, [run], {"q": rubric})
    result = build_report([run], [_judgment("single", "supported")], {},
                          {"q": rubric}, [], reviews)
    assert result["summary"]["single"]["answer_quality"]["full_answer_rate"] == 1


def test_unanswerable_query_scores_abstention_separately_from_grounding(monkeypatch):
    async def fake_model_json(_client, **_kwargs):
        return {"answer_type": "abstain", "points": []}

    monkeypatch.setattr("scripts.eval_agent_hallucination._model_json", fake_model_json)
    run = {**_run("single"), "answer": "I cannot answer from the KB."}
    rubric = {**_rubric(), "answerability": "unanswerable", "required_points": []}
    decision = asyncio.run(_judge_answer_run(
        run, rubric, object(), ("judge", "http://localhost/v1", "EMPTY"), False,
    ))
    assert decision["answer_result"] == "correct_abstain"
    report = build_report([run], [], {}, {"q": rubric}, [decision])
    quality = report["summary"]["single"]["answer_quality"]
    assert quality["correct_abstain_rate"] == 1
    assert quality["task_success_rate"] == 1
    assert report["summary"]["single"]["pending_review"] == 1


def test_answer_judge_command_resumes_only_reviewed_rubrics(tmp_path, monkeypatch):
    from scripts.eval_kb import corpus_manifest

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "documents.jsonl").write_text('{"paper_id":"p"}\n')
    (kb / "chunks.jsonl").write_text('{"chunk_id":"p:1","text":"a fact"}\n')
    run = {**_run("single"), "corpus": corpus_manifest(tmp_path)}
    rubric = {**_rubric(), "corpus": run["corpus"]}
    runs_path, rubrics_path, output = (tmp_path / name for name in
                                       ("runs.jsonl", "rubrics.jsonl", "answers.jsonl"))
    runs_path.write_text(json.dumps(run) + "\n")
    rubrics_path.write_text(json.dumps(rubric) + "\n")
    calls = []

    async def fake_model_json(_client, **_kwargs):
        calls.append(1)
        return {"answer_type": "answer", "points": [
            {"id": "p1", "covered": True, "quote": "a fact"},
        ]}

    monkeypatch.setattr("scripts.eval_agent_hallucination._judge_endpoint",
                        lambda _args: ("judge", "http://localhost/v1", "EMPTY"))
    monkeypatch.setattr("scripts.eval_agent_hallucination._model_json", fake_model_json)
    args = SimpleNamespace(workspace=tmp_path, runs=runs_path, rubrics=rubrics_path,
                           output=output, limit=0, timeout=5, disable_thinking=False)
    asyncio.run(judge_answers_command(args))
    asyncio.run(judge_answers_command(args))
    assert len(calls) == 1
    assert json.loads(output.read_text())["answer_result"] == "full"
