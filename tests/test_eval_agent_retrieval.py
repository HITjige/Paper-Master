"""First-retrieval agent evaluation stops before answer generation."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from scripts.eval_agent_retrieval import (
    FirstRetrievalCapturedError,
    FirstRetrievalHook,
    _first_multi,
    _run_on_loop,
    _single_prediction,
    build_report,
)


def _payload(ids):
    return json.dumps({
        "quality": "sufficient", "retrieval_mode": "hybrid",
        "total_hits": len(ids), "returned_hits": len(ids),
        "papers": [{"paper_id": "p", "chunks": [{"chunk_id": cid} for cid in ids]}],
    })


def test_first_tool_hook_records_visible_order_and_stops():
    hook = FirstRetrievalHook()
    context = AgentHookContext(
        iteration=0,
        messages=[{"role": "tool", "tool_call_id": "one", "name": "kb_retrieve",
                   "content": _payload(["p:2", "p:1"]) }],
        tool_calls=[ToolCallRequest(id="one", name="kb_retrieve",
                                    arguments={"query": "why?", "retrieval_mode": "hybrid"})],
    )
    with pytest.raises(FirstRetrievalCapturedError):
        asyncio.run(hook.after_iteration(context))
    result = _single_prediction(hook)
    assert result["predicted"] == ["p:2", "p:1"]
    assert result["tool_arguments"]["query"] == "why?"


def test_runner_never_calls_model_after_first_kb_tool_result():
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="", finish_reason="tool_calls",
        tool_calls=[ToolCallRequest(id=call_id, name="kb_retrieve",
                                    arguments={"query": "why?"})
                    for call_id in ("first", "second")],
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = [{
        "type": "function", "function": {"name": "kb_retrieve"},
    }]
    tools.prepare_call.side_effect = lambda _name, params: (None, params, None)
    tools.execute = AsyncMock(return_value=_payload(["p:1"]))
    hook = FirstRetrievalHook()
    with pytest.raises(FirstRetrievalCapturedError):
        asyncio.run(AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "why?"}],
            tools=tools, model="fake", max_iterations=5,
            max_tool_result_chars=40000, hook=hook,
        )))
    assert provider.chat_with_retry.await_count == 1
    assert tools.execute.await_count == 1
    assert _single_prediction(hook)["predicted"] == ["p:1"]


def test_real_loop_stops_before_answer_and_does_not_save_one(tmp_path, monkeypatch):
    class CannedProvider(LLMProvider):
        calls = 0

        def get_default_model(self):
            return "canned"

        async def chat(self, **_kwargs):
            self.calls += 1
            return LLMResponse(content="", finish_reason="tool_calls",
                               tool_calls=[ToolCallRequest(
                                   id=call_id, name="kb_retrieve",
                                   arguments={"query": "why?"},
                               ) for call_id in ("first", "second")])

    provider = CannedProvider()
    monkeypatch.setattr("nanobot.cli.commands._make_provider", lambda _cfg: provider)
    monkeypatch.setattr("scripts.eval_agent_retrieval.sync_workspace_templates", lambda *_a, **_k: [])
    from scripts.eval_agent_hallucination import _make_loop as make_real_loop

    def make_loop(workspace, config, hook):
        loop = make_real_loop(workspace, config, hook)
        loop.tools.get("kb_retrieve").execute = AsyncMock(return_value=_payload(["p:1"]))
        return loop

    monkeypatch.setattr("scripts.eval_agent_retrieval._make_loop", make_loop)
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "documents.jsonl").write_text('{"paper_id":"p","title":"Paper"}\n')
    (kb_dir / "chunks.jsonl").write_text(
        '{"paper_id":"p","chunk_id":"p:1","text":"Evidence."}\n',
    )
    config = Config()
    config.agents.defaults.model = "canned"
    config.tools.paper.embedding_model = "/missing/embedding-model"
    config.tools.paper.embedding_fallback = "hash"
    hook = FirstRetrievalHook()
    loop = make_loop(tmp_path, config, hook)

    async def run_two_queries():
        try:
            return [await _run_on_loop(
                loop, hook, {"id": qid, "query": "why?"}, "single", 0, 30,
            ) for qid in ("q1", "q2")]
        finally:
            await loop.close_mcp()

    rows = asyncio.run(run_two_queries())
    assert [row["status"] for row in rows] == ["retrieved", "retrieved"]
    assert [row["predicted"] for row in rows] == [["p:1"], ["p:1"]]
    assert provider.calls == 2
    assert loop.tools.get("kb_retrieve").execute.await_count == 2
    assert all("answer" not in row for row in rows)


def test_multi_executes_router_and_one_retrieval_without_synthesis():
    calls = []

    class Nodes:
        async def router_node(self, state):
            calls.append("router")
            state["routing_decision"] = "internal"
            return state

        async def retrieval_node(self, state):
            calls.append("retrieval")
            state["retrieval_quality"] = "sufficient"
            state["retrieval_results"] = [
                {"paper_id": "p", "chunk_id": "p:2", "text": "B", "score": 0.9},
                {"paper_id": "p", "chunk_id": "p:1", "text": "A", "score": 0.8},
            ]
            return state

        async def synthesis_node(self, _state):
            raise AssertionError("synthesis must not run")

    graph = SimpleNamespace(nodes=Nodes(), agent_config=SimpleNamespace(
        max_iterations=3, rewrite_fallback_delta_threshold=0.05,
        rewrite_context_chars=60000, external_search_top_k=60,
        external_rerank_top_k=5,
    ))
    loop = SimpleNamespace(
        _multi_agent_graph=graph,
        _build_multi_agent_context_snapshot=lambda **_kwargs: {},
        tools_config=SimpleNamespace(paper=SimpleNamespace(
            multi_agent_retrieval_judge_margin=0.02,
        )),
    )
    result = asyncio.run(_first_multi(loop, "why?", "test:one"))
    assert calls == ["router", "retrieval"]
    assert result["status"] == "retrieved"
    # Synthesis groups chunks within a paper in source order.
    assert result["predicted"] == ["p:1", "p:2"]

    async def clarification(state):
        state["requires_clarification"] = True
        return state

    graph.nodes.retrieval_node = clarification
    skipped = asyncio.run(_first_multi(loop, "which paper?", "test:two"))
    assert skipped["status"] == "no_retrieval"
    assert skipped["reason"] == "requires_clarification"


def _pool():
    return {
        "corpus": {"documents.jsonl": "d", "chunks.jsonl": "c"},
        "queries": [{"id": "q", "query": "why?", "tags": ["method"]}],
        "candidates": [{"query_id": "q", "chunk_id": "a"},
                       {"query_id": "q", "chunk_id": "b"}],
    }


def _run(mode, predicted, status="retrieved"):
    return {
        "run_id": f"first:q:0:{mode}", "query_id": "q", "query": "why?",
        "mode": mode, "repeat": 0, "status": status, "predicted": predicted,
        "elapsed_s": 1.0, "corpus": _pool()["corpus"],
    }


def _judgments():
    return [{"query_id": "q", "chunk_id": "a", "grade": 2, "status": "accepted"},
            {"query_id": "q", "chunk_id": "b", "grade": 0, "status": "accepted"}]


def test_report_scores_no_retrieval_as_zero_and_pairs_modes():
    report = build_report(_pool(), [_run("single", ["a"]),
                                    _run("multi", [], "no_retrieval")],
                          _judgments(), [], [1, 5])
    assert report["summary"]["single"]["metrics"]["1"]["recall"] == 1
    assert report["summary"]["multi"]["metrics"]["1"]["recall"] == 0
    assert report["summary"]["paired"]["metrics"]["1"]["recall"] == -1
    assert report["summary"]["multi"]["retrieval_attempt_rate"] == 0


def test_report_skips_unjudged_prediction_and_exports_supplement_ids():
    report = build_report(_pool(), [_run("single", ["outside"]),
                                    _run("multi", ["a"])],
                          _judgments(), [], [1])
    assert report["runs"][0]["status"] == "unjudged_predictions"
    assert report["summary"]["single"]["scored_runs"] == 0
    assert report["queries"] == [{"id": "q", "unjudged": ["outside"]}]
    missing = build_report(_pool(), [_run("single", ["a"])],
                           _judgments()[:1], [], [1])
    assert missing["runs"][0]["status"] == "incomplete_labels"
