import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse
from nanobot.session.manager import Session


def _mk_loop() -> AgentLoop:
    loop = AgentLoop.__new__(AgentLoop)
    from nanobot.config.schema import AgentDefaults

    loop.max_tool_result_chars = AgentDefaults().max_tool_result_chars
    return loop


def _make_full_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")


def test_save_turn_skips_multimodal_user_when_only_runtime_context() -> None:
    loop = _mk_loop()
    session = Session(key="test:runtime-only")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [{"role": "user", "content": [{"type": "text", "text": runtime}]}],
        skip=0,
    )
    assert session.messages == []


def test_save_turn_keeps_image_placeholder_with_path_after_runtime_strip() -> None:
    loop = _mk_loop()
    session = Session(key="test:image")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": runtime},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}, "_meta": {"path": "/media/feishu/photo.jpg"}},
            ],
        }],
        skip=0,
    )
    assert session.messages[0]["content"] == [{"type": "text", "text": "[image: /media/feishu/photo.jpg]"}]


def test_save_turn_keeps_image_placeholder_without_meta() -> None:
    loop = _mk_loop()
    session = Session(key="test:image-no-meta")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": runtime},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }],
        skip=0,
    )
    assert session.messages[0]["content"] == [{"type": "text", "text": "[image]"}]


def test_save_turn_keeps_tool_results_under_16k() -> None:
    loop = _mk_loop()
    session = Session(key="test:tool-result")
    content = "x" * 12_000

    loop._save_turn(
        session,
        [{"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": content}],
        skip=0,
    )

    assert session.messages[0]["content"] == content


def test_save_turn_skips_transient_length_recovery_messages() -> None:
    loop = _mk_loop()
    session = Session(key="test:length-recovery")

    loop._save_turn(
        session,
        [
            {
                "role": "assistant",
                "content": "partial",
                "_nanobot_transient": "length_recovery",
            },
            {
                "role": "user",
                "content": "internal continuation prompt",
                "_nanobot_transient": "length_recovery",
            },
            {"role": "assistant", "content": "partialfinal"},
        ],
        skip=0,
    )

    assert [message["content"] for message in session.messages] == ["partialfinal"]


def test_restore_runtime_checkpoint_rehydrates_completed_and_pending_tools() -> None:
    loop = _mk_loop()
    session = Session(
        key="test:checkpoint",
        metadata={
            AgentLoop._RUNTIME_CHECKPOINT_KEY: {
                "assistant_message": {
                    "role": "assistant",
                    "content": "working",
                    "tool_calls": [
                        {
                            "id": "call_done",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                        {
                            "id": "call_pending",
                            "type": "function",
                            "function": {"name": "exec", "arguments": "{}"},
                        },
                    ],
                },
                "completed_tool_results": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_done",
                        "name": "read_file",
                        "content": "ok",
                    }
                ],
                "pending_tool_calls": [
                    {
                        "id": "call_pending",
                        "type": "function",
                        "function": {"name": "exec", "arguments": "{}"},
                    }
                ],
            }
        },
    )

    restored = loop._restore_runtime_checkpoint(session)

    assert restored is True
    assert session.metadata.get(AgentLoop._RUNTIME_CHECKPOINT_KEY) is None
    assert session.messages[0]["role"] == "assistant"
    assert session.messages[1]["tool_call_id"] == "call_done"
    assert session.messages[2]["tool_call_id"] == "call_pending"
    assert "interrupted before this tool finished" in session.messages[2]["content"].lower()


def test_restore_runtime_checkpoint_dedupes_overlapping_tail() -> None:
    loop = _mk_loop()
    session = Session(
        key="test:checkpoint-overlap",
        messages=[
            {
                "role": "assistant",
                "content": "working",
                "tool_calls": [
                    {
                        "id": "call_done",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    },
                    {
                        "id": "call_pending",
                        "type": "function",
                        "function": {"name": "exec", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_done",
                "name": "read_file",
                "content": "ok",
            },
        ],
        metadata={
            AgentLoop._RUNTIME_CHECKPOINT_KEY: {
                "assistant_message": {
                    "role": "assistant",
                    "content": "working",
                    "tool_calls": [
                        {
                            "id": "call_done",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                        {
                            "id": "call_pending",
                            "type": "function",
                            "function": {"name": "exec", "arguments": "{}"},
                        },
                    ],
                },
                "completed_tool_results": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_done",
                        "name": "read_file",
                        "content": "ok",
                    }
                ],
                "pending_tool_calls": [
                    {
                        "id": "call_pending",
                        "type": "function",
                        "function": {"name": "exec", "arguments": "{}"},
                    }
                ],
            }
        },
    )

    restored = loop._restore_runtime_checkpoint(session)

    assert restored is True
    assert session.metadata.get(AgentLoop._RUNTIME_CHECKPOINT_KEY) is None
    assert len(session.messages) == 3
    assert session.messages[0]["role"] == "assistant"
    assert session.messages[1]["tool_call_id"] == "call_done"
    assert session.messages[2]["tool_call_id"] == "call_pending"


@pytest.mark.asyncio
async def test_process_message_persists_user_message_before_turn_completes(tmp_path: Path) -> None:
    loop = _make_full_loop(tmp_path)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
    loop._run_agent_loop = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

    msg = InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="persist me")
    with pytest.raises(RuntimeError, match="boom"):
        await loop._process_message(msg)

    loop.sessions.invalidate("feishu:c1")
    persisted = loop.sessions.get_or_create("feishu:c1")
    assert [m["role"] for m in persisted.messages] == ["user"]
    assert persisted.messages[0]["content"] == "persist me"
    assert persisted.metadata.get(AgentLoop._PENDING_USER_TURN_KEY) is True
    assert persisted.updated_at >= persisted.created_at


@pytest.mark.asyncio
async def test_process_message_does_not_duplicate_early_persisted_user_message(tmp_path: Path) -> None:
    loop = _make_full_loop(tmp_path)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
    loop._run_agent_loop = AsyncMock(return_value=(
        "done",
        None,
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "done"},
        ],
        "stop",
        False,
    ))  # type: ignore[method-assign]

    result = await loop._process_message(
        InboundMessage(channel="feishu", sender_id="u1", chat_id="c2", content="hello")
    )

    assert result is not None
    assert result.content == "done"
    session = loop.sessions.get_or_create("feishu:c2")
    assert [
        {k: v for k, v in m.items() if k in {"role", "content"}}
        for m in session.messages
    ] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "done"},
    ]
    assert AgentLoop._PENDING_USER_TURN_KEY not in session.metadata


@pytest.mark.asyncio
async def test_next_turn_after_crash_closes_pending_user_turn_before_new_input(tmp_path: Path) -> None:
    loop = _make_full_loop(tmp_path)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
    loop.provider.chat_with_retry = AsyncMock(return_value=MagicMock())  # unused because _run_agent_loop is stubbed

    session = loop.sessions.get_or_create("feishu:c3")
    session.add_message("user", "old question")
    session.metadata[AgentLoop._PENDING_USER_TURN_KEY] = True
    loop.sessions.save(session)

    loop._run_agent_loop = AsyncMock(return_value=(
        "new answer",
        None,
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "Error: Task interrupted before a response was generated."},
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "new answer"},
        ],
        "stop",
        False,
    ))  # type: ignore[method-assign]

    result = await loop._process_message(
        InboundMessage(channel="feishu", sender_id="u1", chat_id="c3", content="new question")
    )

    assert result is not None
    assert result.content == "new answer"
    session = loop.sessions.get_or_create("feishu:c3")
    assert [
        {k: v for k, v in m.items() if k in {"role", "content"}}
        for m in session.messages
    ] == [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "Error: Task interrupted before a response was generated."},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
    ]
    assert AgentLoop._PENDING_USER_TURN_KEY not in session.metadata


@pytest.mark.asyncio
async def test_stop_preserves_runtime_checkpoint_for_next_turn(tmp_path: Path) -> None:
    from nanobot.command.builtin import cmd_stop
    from nanobot.command.router import CommandContext

    loop = _make_full_loop(tmp_path)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

    checkpoint_saved = asyncio.Event()

    async def interrupted_run_agent_loop(_initial_messages, *, session=None, **_kwargs):
        assert session is not None
        loop._set_runtime_checkpoint(
            session,
            {
                "assistant_message": {
                    "role": "assistant",
                    "content": "working",
                    "tool_calls": [
                        {
                            "id": "call_done",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                        {
                            "id": "call_pending",
                            "type": "function",
                            "function": {"name": "exec", "arguments": "{}"},
                        },
                    ],
                },
                "completed_tool_results": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_done",
                        "name": "read_file",
                        "content": "ok",
                    }
                ],
                "pending_tool_calls": [
                    {
                        "id": "call_pending",
                        "type": "function",
                        "function": {"name": "exec", "arguments": "{}"},
                    }
                ],
            },
        )
        checkpoint_saved.set()
        await asyncio.Event().wait()

    loop._run_agent_loop = interrupted_run_agent_loop  # type: ignore[method-assign]

    first_msg = InboundMessage(channel="feishu", sender_id="u1", chat_id="c4", content="keep progress")
    task = asyncio.create_task(loop._process_message(first_msg))
    loop._active_tasks[first_msg.session_key] = [task]
    await asyncio.wait_for(checkpoint_saved.wait(), timeout=1.0)

    stop_msg = InboundMessage(channel="feishu", sender_id="u1", chat_id="c4", content="/stop")
    stop_ctx = CommandContext(msg=stop_msg, session=None, key=stop_msg.session_key, raw="/stop", loop=loop)
    stop_result = await cmd_stop(stop_ctx)

    assert "Stopped 1 task" in stop_result.content
    assert task.done()

    loop.sessions.invalidate("feishu:c4")
    interrupted = loop.sessions.get_or_create("feishu:c4")
    assert interrupted.metadata.get(AgentLoop._PENDING_USER_TURN_KEY) is True
    assert interrupted.metadata.get(AgentLoop._RUNTIME_CHECKPOINT_KEY) is not None

    async def resumed_run_agent_loop(initial_messages, **_kwargs):
        return (
            "next answer",
            None,
            [*initial_messages, {"role": "assistant", "content": "next answer"}],
            "stop",
            False,
        )

    loop._run_agent_loop = resumed_run_agent_loop  # type: ignore[method-assign]
    result = await loop._process_message(
        InboundMessage(channel="feishu", sender_id="u1", chat_id="c4", content="continue here")
    )

    assert result is not None
    assert result.content == "next answer"

    session = loop.sessions.get_or_create("feishu:c4")
    assert [
        {k: v for k, v in m.items() if k in {"role", "content", "tool_call_id", "name"}}
        for m in session.messages
    ] == [
        {"role": "user", "content": "keep progress"},
        {"role": "assistant", "content": "working"},
        {"role": "tool", "tool_call_id": "call_done", "name": "read_file", "content": "ok"},
        {
            "role": "tool",
            "tool_call_id": "call_pending",
            "name": "exec",
            "content": "Error: Task interrupted before this tool finished.",
        },
        {"role": "user", "content": "continue here"},
        {"role": "assistant", "content": "next answer"},
    ]
    assert AgentLoop._PENDING_USER_TURN_KEY not in session.metadata
    assert AgentLoop._RUNTIME_CHECKPOINT_KEY not in session.metadata


@pytest.mark.asyncio
async def test_system_subagent_followup_is_persisted_before_prompt_assembly(tmp_path: Path) -> None:
    loop = _make_full_loop(tmp_path)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.add_message("user", "question")
    session.add_message("assistant", "working")
    loop.sessions.save(session)

    seen: dict[str, list[dict]] = {}

    async def fake_run_agent_loop(initial_messages, **_kwargs):
        seen["initial_messages"] = initial_messages
        return (
            "done",
            [],
            [*initial_messages, {"role": "assistant", "content": "done"}],
            "stop",
            False,
        )

    loop._run_agent_loop = fake_run_agent_loop  # type: ignore[method-assign]

    await loop._process_message(
        InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id="cli:test",
            content="subagent result",
            metadata={"subagent_task_id": "sub-1"},
        )
    )

    non_system = [m for m in seen["initial_messages"] if m.get("role") != "system"]
    assert [m["content"] for m in non_system[:2]] == ["question", "working"]
    assert non_system[2]["content"].count("subagent result") == 1
    assert "Current Time:" in non_system[2]["content"]

    loop.sessions.invalidate("cli:test")
    persisted = loop.sessions.get_or_create("cli:test")
    assert [
        {k: v for k, v in m.items() if k in {"role", "content", "injected_event", "subagent_task_id"}}
        for m in persisted.messages
    ] == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "working"},
        {
            "role": "assistant",
            "content": "subagent result",
            "injected_event": "subagent_result",
            "subagent_task_id": "sub-1",
        },
        {"role": "assistant", "content": "done"},
    ]


@pytest.mark.asyncio
async def test_multiple_subagent_followups_all_persist_as_standalone_history(tmp_path: Path) -> None:
    loop = _make_full_loop(tmp_path)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

    async def fake_run_agent_loop(initial_messages, **_kwargs):
        return (
            "ack",
            [],
            [*initial_messages, {"role": "assistant", "content": "ack"}],
            "stop",
            False,
        )

    loop._run_agent_loop = fake_run_agent_loop  # type: ignore[method-assign]

    for idx in range(3):
        await loop._process_message(
            InboundMessage(
                channel="system",
                sender_id="subagent",
                chat_id="cli:multi",
                content=f"subagent result {idx}",
                metadata={"subagent_task_id": f"sub-{idx}"},
            )
        )

    loop.sessions.invalidate("cli:multi")
    persisted = loop.sessions.get_or_create("cli:multi")
    followups = [m for m in persisted.messages if m.get("injected_event") == "subagent_result"]
    assert [m["content"] for m in followups] == [
        "subagent result 0",
        "subagent result 1",
        "subagent result 2",
    ]


def test_prompt_merge_does_not_replace_standalone_subagent_history_entry(tmp_path: Path) -> None:
    loop = _mk_loop()
    session = Session(key="cli:merge")
    session.add_message("assistant", "previous assistant")

    inserted = loop._persist_subagent_followup(
        session,
        InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id="cli:merge",
            content="subagent result",
            metadata={"subagent_task_id": "sub-1"},
        ),
    )

    assert inserted is True

    builder = ContextBuilder(tmp_path)
    projected = builder.build_messages(
        history=session.get_history(max_messages=0),
        current_message="",
        current_role="assistant",
        channel="cli",
        chat_id="merge",
    )

    non_system = [m for m in projected if m.get("role") != "system"]
    assert len(non_system) == 2
    assert "subagent result" in non_system[-1]["content"]
    assert session.messages[-1]["content"] == "subagent result"
    assert session.messages[-1]["injected_event"] == "subagent_result"


def test_subagent_followup_dedupes_by_task_id() -> None:
    loop = _mk_loop()
    session = Session(key="cli:dedupe")
    msg = InboundMessage(
        channel="system",
        sender_id="subagent",
        chat_id="cli:dedupe",
        content="subagent result",
        metadata={"subagent_task_id": "sub-1"},
    )

    assert loop._persist_subagent_followup(session, msg) is True
    assert loop._persist_subagent_followup(session, msg) is False
    assert len(session.messages) == 1


def test_subagent_followup_skips_empty_content() -> None:
    loop = _mk_loop()
    session = Session(key="cli:empty")
    msg = InboundMessage(
        channel="system",
        sender_id="subagent",
        chat_id="cli:empty",
        content="",
        metadata={"subagent_task_id": "sub-empty"},
    )

    assert loop._persist_subagent_followup(session, msg) is False
    assert session.messages == []


def test_recent_paper_session_routes_referential_performance_followup(tmp_path: Path) -> None:
    loop = _make_full_loop(tmp_path)
    loop.tools_config.paper.multi_agent_orchestrator_enabled = True
    loop._multi_agent_graph = MagicMock()
    session = Session(key="websocket:paper-followup")
    session.add_message("user", "详细解读第一篇论文")
    session.add_message("assistant", "ATM 是该论文使用的 EEG 编码模型。")
    session.metadata[AgentLoop._MULTI_AGENT_LAST_ROUTING_KEY] = "internal"
    session.metadata[AgentLoop._MULTI_AGENT_CONTEXT_MESSAGE_COUNT_KEY] = len(session.messages)
    session.metadata[AgentLoop._MULTI_AGENT_ACTIVE_PAPERS_KEY] = [
        {"paper_id": "atm-paper", "title": "ATM"}
    ]

    use_multi, context = asyncio.run(
        loop._decide_multi_agent_with_orchestrator("详细讲一下该模型的性能", session)
    )

    assert use_multi is True
    assert context["orchestrator_decision"] == "multi_agent"
    assert "follow-up" in context["orchestrator_reasoning"]

    use_multi, context = asyncio.run(
        loop._decide_multi_agent_with_orchestrator("具体的优化算法是怎么设计的", session)
    )
    assert use_multi is True
    assert context["orchestrator_decision"] == "multi_agent"

    use_multi, context = asyncio.run(
        loop._decide_multi_agent_with_orchestrator("谢谢", session)
    )
    assert use_multi is False
    assert context["orchestrator_reasoning"] == "simple acknowledgement"


def test_recent_paper_route_fails_open_when_structured_classifier_is_empty(
    tmp_path: Path,
) -> None:
    loop = _make_full_loop(tmp_path)
    loop.tools_config.paper.multi_agent_orchestrator_enabled = True
    loop._multi_agent_graph = MagicMock()
    loop.provider.chat_structured_with_retry = AsyncMock(return_value=LLMResponse(
        content=None,
        reasoning_content="unfinished classifier reasoning",
    ))
    session = Session(key="websocket:paper-route-empty")
    session.add_message("user", "解读 ATM 论文")
    session.add_message("assistant", "ATM 方法概览。")
    session.metadata[AgentLoop._MULTI_AGENT_CONTEXT_MESSAGE_COUNT_KEY] = len(
        session.messages
    )

    use_multi, context = asyncio.run(
        loop._decide_multi_agent_with_orchestrator("为什么会这样", session)
    )

    assert use_multi is True
    assert context["orchestrator_decision"] == "multi_agent"
    assert "preserve recent paper context" in context["orchestrator_reasoning"]
    kwargs = loop.provider.chat_structured_with_retry.await_args.kwargs
    assert kwargs["disable_thinking"] is True
    assert kwargs["json_schema"]["properties"]["decision"]["enum"] == [
        "multi_agent",
        "single_agent",
    ]


def test_recent_paper_route_allows_high_confidence_topic_change(tmp_path: Path) -> None:
    loop = _make_full_loop(tmp_path)
    loop._multi_agent_graph = MagicMock()
    loop.provider.chat_structured_with_retry = AsyncMock(return_value=LLMResponse(
        content=(
            '{"decision":"single_agent","confidence":0.95,'
            '"reasoning":"clear topic change"}'
        ),
    ))
    session = Session(key="websocket:paper-route-topic-change")
    session.add_message("user", "解读 ATM 论文")
    session.add_message("assistant", "ATM 方法概览。")
    session.metadata[AgentLoop._MULTI_AGENT_CONTEXT_MESSAGE_COUNT_KEY] = len(
        session.messages
    )

    use_multi, context = asyncio.run(
        loop._decide_multi_agent_with_orchestrator("帮我写一个请假邮件", session)
    )

    assert use_multi is False
    assert context["orchestrator_decision"] == "single_agent"


def test_single_agent_paper_tool_refreshes_multi_agent_context(tmp_path: Path) -> None:
    async def scenario() -> None:
        loop = _make_full_loop(tmp_path)
        loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
        loop.tools_config.paper.multi_agent_orchestrator_enabled = False
        loop._run_agent_loop = AsyncMock(return_value=(
            "论文已经入库。",
            ["paper_ingest"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "请处理这个文件"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "ingest-1",
                        "type": "function",
                        "function": {
                            "name": "paper_ingest",
                            "arguments": "{}",
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "ingest-1",
                    "name": "paper_ingest",
                    "content": (
                        '{"status":"ok","paper_id":"atm-1",'
                        '"title":"ATM: EEG-to-image"}'
                    ),
                },
                {"role": "assistant", "content": "论文已经入库。"},
            ],
            "completed",
            False,
        ))  # type: ignore[method-assign]

        result = await loop._process_message(InboundMessage(
            channel="websocket",
            sender_id="u1",
            chat_id="single-paper-tool",
            content="请处理这个文件",
        ))

        assert result is not None
        session = loop.sessions.get_or_create("websocket:single-paper-tool")
        assert session.metadata[AgentLoop._MULTI_AGENT_LAST_ROUTING_KEY] == (
            "single_agent_paper_tools"
        )
        assert session.metadata[AgentLoop._MULTI_AGENT_ACTIVE_PAPERS_KEY] == [
            {"paper_id": "atm-1", "title": "ATM: EEG-to-image"}
        ]

        loop.tools_config.paper.multi_agent_orchestrator_enabled = True
        loop._multi_agent_graph = MagicMock()
        use_multi, _context = await loop._decide_multi_agent_with_orchestrator(
            "具体的优化算法是怎么设计的",
            session,
        )
        assert use_multi is True

    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_loop_hook_streams_paper_answer_immediately() -> None:
    from nanobot.agent.hook import AgentHookContext
    from nanobot.agent.loop import _LoopHook

    streamed: list[str] = []
    stream_ends: list[bool] = []

    async def on_stream(delta: str) -> None:
        streamed.append(delta)

    async def on_stream_end(*, resuming: bool = False) -> None:
        stream_ends.append(resuming)

    hook = _LoopHook(
        _mk_loop(),
        on_stream=on_stream,
        on_stream_end=on_stream_end,
    )
    context = AgentHookContext(
        iteration=1,
        messages=[{
            "role": "tool",
            "name": "kb_retrieve",
            "tool_call_id": "kb-1",
            "content": '{"quality":"sufficient"}',
        }],
    )

    await hook.on_stream(context, "streamed draft")
    await hook.on_stream_end(context, resuming=False)

    assert streamed == ["streamed draft"]
    assert stream_ends == [False]


@pytest.mark.asyncio
async def test_single_agent_streams_paper_answer_without_post_generation_repair(
    tmp_path: Path,
) -> None:
    loop = _make_full_loop(tmp_path)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
    loop.provider.chat_with_retry = AsyncMock()
    tool_payload = json.dumps({
        "quality": "sufficient",
        "quality_reason": "strong match",
        "query": "EEG papers",
        "queries": ["EEG papers"],
        "papers": [{
            "paper_id": "p1",
            "title": "Grounded EEG Paper",
            "authors": ["Ada Researcher"],
            "abstract": "Evidence about EEG.",
            "evidence_level": "full_text",
            "chunks": [{
                "chunk_id": "p1:0",
                "text": "Grounded answer.",
                "score": 0.9,
            }],
        }],
    })
    streamed: list[str] = []
    stream_ends: list[bool] = []

    async def run_agent_loop(*args: object, **kwargs: object):
        on_stream = kwargs["on_stream"]
        on_stream_end = kwargs["on_stream_end"]
        assert callable(on_stream)
        assert callable(on_stream_end)
        await on_stream("Grounded ")
        await on_stream("answer [p1].")
        await on_stream_end(resuming=False)
        return (
            "Grounded answer [p1].",
            ["kb_retrieve"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "有没有 EEG 论文"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "kb-1",
                        "type": "function",
                        "function": {"name": "kb_retrieve", "arguments": "{}"},
                    }],
                },
                {
                    "role": "tool",
                    "name": "kb_retrieve",
                    "tool_call_id": "kb-1",
                    "content": tool_payload,
                },
                {"role": "assistant", "content": "Grounded answer [p1]."},
            ],
            "completed",
            False,
        )

    loop._run_agent_loop = run_agent_loop  # type: ignore[method-assign]

    async def on_stream(delta: str) -> None:
        streamed.append(delta)

    async def on_stream_end(*, resuming: bool = False) -> None:
        stream_ends.append(resuming)

    result = await loop._process_message(
        InboundMessage(
            channel="websocket",
            sender_id="u1",
            chat_id="paper-stream-once",
            content="有没有 EEG 论文",
        ),
        on_stream=on_stream,
        on_stream_end=on_stream_end,
    )

    assert result is not None
    assert result.content == "Grounded answer [p1]."
    assert result.metadata["_streamed"] is True
    assert "paper_evidence" not in result.metadata
    assert streamed == ["Grounded ", "answer [p1]."]
    assert stream_ends == [False]
    loop.provider.chat_with_retry.assert_not_awaited()
    session = loop.sessions.get_or_create("websocket:paper-stream-once")
    assert session.metadata[AgentLoop._MULTI_AGENT_PRESENTED_PAPER_IDS_KEY] == ["p1"]
    assert session.metadata[AgentLoop._MULTI_AGENT_LAST_SEARCH_TOPIC_KEY] == (
        "有没有 EEG 论文"
    )


@pytest.mark.asyncio
async def test_single_agent_novelty_prompt_includes_presented_paper_ids(
    tmp_path: Path,
) -> None:
    loop = _make_full_loop(tmp_path)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
    session = loop.sessions.get_or_create("websocket:paper-novelty-context")
    session.add_message("user", "有没有 EEG 论文")
    session.add_message("assistant", "Found one [2401.00001v2].")
    session.metadata[AgentLoop._MULTI_AGENT_PRESENTED_PAPER_IDS_KEY] = [
        "2401.00001v2"
    ]
    loop.sessions.save(session)
    loop._run_agent_loop = AsyncMock(return_value=(
        "No additional papers found.",
        [],
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "有没有其他论文"},
            {"role": "assistant", "content": "No additional papers found."},
        ],
        "completed",
        False,
    ))  # type: ignore[method-assign]

    await loop._process_message(InboundMessage(
        channel="websocket",
        sender_id="u1",
        chat_id="paper-novelty-context",
        content="有没有其他论文",
    ))

    initial_messages = loop._run_agent_loop.await_args.args[0]
    current_user = next(
        message for message in reversed(initial_messages)
        if message.get("role") == "user"
    )
    assert "paper_novelty_context" in str(current_user["content"])
    assert "2401.00001v2" in str(current_user["content"])
    assert "exclude_paper_ids" in str(current_user["content"])


def test_three_turn_paper_followup_stays_on_multi_agent_path(tmp_path: Path) -> None:
    async def scenario() -> None:
        loop = _make_full_loop(tmp_path)
        loop.tools_config.paper.multi_agent_orchestrator_enabled = True
        loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
        graph = MagicMock()
        graph.run = AsyncMock(side_effect=[
            {
                "final_answer": "找到 ATM 论文。",
                "routing_decision": "external",
                "iteration_count": 1,
                "external_papers": [{"paper_id": "atm-1", "title": "ATM"}],
                "retrieval_results": [],
                "citations": ["[atm-1] ATM"],
                "discovery_request": True,
                "resolved_topic": "EEG-to-image",
            },
            {
                "final_answer": "ATM 方法解读。",
                "routing_decision": "internal",
                "iteration_count": 1,
                "external_papers": [],
                "retrieval_results": [{
                    "paper_id": "atm-1",
                    "paper_title": "ATM",
                    "text": "method",
                }],
                "citations": ["atm-1"],
            },
            {
                "final_answer": "ATM 性能解读。",
                "routing_decision": "internal",
                "iteration_count": 1,
                "external_papers": [],
                "retrieval_results": [{
                    "paper_id": "atm-1",
                    "paper_title": "ATM",
                    "text": "performance",
                }],
                "citations": ["atm-1"],
            },
        ])
        loop._multi_agent_graph = graph

        for content in (
            "EEG-to-image有什么论文",
            "详细解读第一篇论文",
            "详细讲一下该模型的性能",
        ):
            await loop._process_message(InboundMessage(
                channel="websocket",
                sender_id="u1",
                chat_id="paper-three-turns",
                content=content,
            ))

        assert graph.run.await_count == 3
        third_call = graph.run.call_args_list[2].kwargs
        assert third_call["last_routing_decision"] == "internal"
        assert third_call["referenced_papers"][0]["paper_id"] == "atm-1"
        assert third_call["presented_paper_ids"] == ["atm-1"]
        assert third_call["last_search_topic"] == "EEG-to-image"

    asyncio.run(scenario())


def test_novelty_query_abandons_paused_selection_and_starts_fresh(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        loop = _make_full_loop(tmp_path)
        loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
        graph = MagicMock()
        graph.run = AsyncMock(return_value={
            "final_answer": "没有发现新的论文。",
            "routing_decision": "internal",
            "iteration_count": 0,
            "external_papers": [],
            "retrieval_results": [],
            "citations": [],
            "novelty_required": True,
        })
        graph.resume = AsyncMock(return_value={"final_answer": "should not resume"})
        loop._multi_agent_graph = graph

        session = loop.sessions.get_or_create("websocket:novelty-after-selection")
        session.metadata["multi_agent_paused_state"] = {
            "user_query": "强化学习论文",
            "research_phase": "select",
            "papers_for_selection": [{"paper_id": "old-1", "title": "Old"}],
        }
        loop.sessions.save(session)

        result = await loop.process_with_multi_agent(
            "还有没有其他相关论文？",
            session_key="websocket:novelty-after-selection",
            channel="websocket",
            chat_id="novelty-after-selection",
        )

        assert result is not None
        graph.resume.assert_not_awaited()
        graph.run.assert_awaited_once()
        assert "multi_agent_paused_state" not in session.metadata

    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_multi_agent_pauses_and_resumes_external_search_confirmation(
    tmp_path: Path,
) -> None:
    loop = _make_full_loop(tmp_path)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
    graph = MagicMock()
    graph.run = AsyncMock(return_value={
        "final_answer": "当前知识库证据不足，是否允许我搜索外部 arXiv 论文？",
        "draft_answer": "当前知识库证据不足，是否允许我搜索外部 arXiv 论文？",
        "routing_decision": "internal",
        "research_phase": "confirm_search",
        "awaiting_external_search_confirmation": True,
        "retrieval_results": [],
        "papers_for_selection": [],
        "user_query": "详细解读这篇论文",
        "session_id": "websocket:external-confirm",
    })
    graph.resume = AsyncMock(return_value={
        "final_answer": "外部搜索后的回答。",
        "routing_decision": "external",
        "research_phase": "complete",
        "iteration_count": 1,
        "retrieval_results": [],
        "external_papers": [{"paper_id": "2501.00001"}],
        "citations": [],
    })
    loop._multi_agent_graph = graph

    paused = await loop.process_with_multi_agent(
        "详细解读这篇论文",
        session_key="websocket:external-confirm",
        channel="websocket",
        chat_id="external-confirm",
    )

    assert paused is not None
    assert paused.metadata["awaiting_external_search_confirmation"] is True
    session = loop.sessions.get_or_create("websocket:external-confirm")
    assert session.metadata["multi_agent_paused_state"]["research_phase"] == (
        "confirm_search"
    )

    resumed = await loop.process_with_multi_agent(
        "可以",
        session_key="websocket:external-confirm",
        channel="websocket",
        chat_id="external-confirm",
    )

    assert resumed is not None
    assert "外部搜索后的回答" in resumed.content
    graph.resume.assert_awaited_once()
    assert "multi_agent_paused_state" not in session.metadata


@pytest.mark.asyncio
async def test_multi_agent_persists_only_user_and_final_answer_in_strict_mode(tmp_path: Path) -> None:
    loop = _make_full_loop(tmp_path)
    loop.tools_config.paper.multi_agent_orchestrator_enabled = True
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

    graph = MagicMock()
    graph.run = AsyncMock(return_value={
        "final_answer": "final synthesis answer",
        "routing_decision": "hybrid",
        "routing_reasoning": "need both internal and external",
        "iteration_count": 1,
        "citations": ["arxiv:1111.1111"],
        "retrieval_results": [{"id": "kb-1"}],
        "external_papers": [{"id": "arxiv:1111.1111"}],
        "critic_verdict": "passed",
    })
    loop._multi_agent_graph = graph

    result = await loop._process_message(
        InboundMessage(channel="feishu", sender_id="u1", chat_id="ma1", content="Please summarize this paper")
    )

    assert result is not None
    assert "final synthesis answer" in result.content

    session = loop.sessions.get_or_create("feishu:ma1")
    assert [
        {k: v for k, v in m.items() if k in {"role", "content"}}
        for m in session.messages
    ] == [
        {"role": "user", "content": "Please summarize this paper"},
        {"role": "assistant", "content": "final synthesis answer"},
    ]
    assert session.metadata.get(AgentLoop._MULTI_AGENT_LAST_ROUTING_KEY) == "hybrid"
    assert session.metadata.get(AgentLoop._MULTI_AGENT_LAST_SOURCES_KEY) == [
        "internal_kb",
        "external_search",
    ]


@pytest.mark.asyncio
async def test_multi_agent_router_receives_last_routing_and_memory_context(tmp_path: Path) -> None:
    loop = _make_full_loop(tmp_path)
    loop.tools_config.paper.multi_agent_orchestrator_enabled = True
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
    loop.tools_config.paper.multi_agent_memory_mode = "strict_with_citations"

    graph = MagicMock()
    graph.run = AsyncMock(side_effect=[
        {
            "final_answer": "first answer",
            "routing_decision": "external",
            "routing_reasoning": "latest papers requested",
            "iteration_count": 1,
            "citations": ["arxiv:2001.00001"],
            "retrieval_results": [],
            "external_papers": [{"id": "arxiv:2001.00001"}],
            "critic_verdict": "passed",
        },
        {
            "final_answer": "second answer",
            "routing_decision": "internal",
            "routing_reasoning": "follow-up asks details of previous paper",
            "iteration_count": 1,
            "citations": [],
            "retrieval_results": [{"id": "kb-2"}],
            "external_papers": [],
            "critic_verdict": "passed",
        },
    ])
    loop._multi_agent_graph = graph

    await loop._process_message(
        InboundMessage(channel="feishu", sender_id="u1", chat_id="ma2", content="Find latest EEG paper")
    )
    await loop._process_message(
        InboundMessage(channel="feishu", sender_id="u1", chat_id="ma2", content="Compare it with previous one")
    )

    first_call = graph.run.call_args_list[0].kwargs
    second_call = graph.run.call_args_list[1].kwargs
    assert first_call["last_routing_decision"] == "none"
    assert second_call["last_routing_decision"] == "external"
    assert "Find latest EEG paper" in second_call["router_memory_short"]

    session = loop.sessions.get_or_create("feishu:ma2")
    assert "References:" in session.messages[1]["content"]
    assert "arxiv:2001.00001" in session.messages[1]["content"]


@pytest.mark.asyncio
async def test_multi_agent_uses_shared_context_without_repeating_current_query(
    tmp_path: Path,
) -> None:
    loop = _make_full_loop(tmp_path)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]
    (tmp_path / "AGENTS.md").write_text("shared bootstrap marker", encoding="utf-8")
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    )
    graph = MagicMock()
    graph.run = AsyncMock(return_value={
        "final_answer": "reviewed answer",
        "routing_decision": "internal",
        "iteration_count": 1,
        "citations": [],
        "retrieval_results": [],
        "external_papers": [],
    })
    loop._multi_agent_graph = graph
    streamed = []
    stream_ends = []

    async def on_stream(delta: str) -> None:
        streamed.append(delta)

    async def on_stream_end(*, resuming: bool) -> None:
        stream_ends.append(resuming)

    result = await loop.process_with_multi_agent(
        "current query must not repeat",
        session_key="websocket:shared-context",
        channel="websocket",
        chat_id="shared-context",
        media=[str(image_path)],
        on_stream=on_stream,
        on_stream_end=on_stream_end,
        session_summary="compacted session marker",
    )

    kwargs = graph.run.await_args.kwargs
    assert kwargs["recent_dialog_context"] == "(empty)"
    assert "current query must not repeat" not in kwargs["recent_dialog_context"]
    assert "shared bootstrap marker" in kwargs["shared_system_context"]
    assert "compacted session marker" in kwargs["runtime_context"]
    assert kwargs["media_context"][0]["type"] == "image_url"
    assert result is not None
    assert result.metadata["_streamed"] is True
    assert streamed == [result.content]
    assert stream_ends == [False]

    session = loop.sessions.get_or_create("websocket:shared-context")
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert session.messages[0]["content"] == "current query must not repeat"


@pytest.mark.asyncio
async def test_orchestrator_can_force_single_agent_path(tmp_path: Path) -> None:
    loop = _make_full_loop(tmp_path)
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=False)  # type: ignore[method-assign]

    loop._decide_multi_agent_with_orchestrator = AsyncMock(return_value=(
        False,
        {
            "orchestrator_decision": "single",
            "orchestrator_reasoning": "simple question",
            "orchestrator_confidence": 0.92,
        },
    ))  # type: ignore[method-assign]

    graph = MagicMock()
    graph.run = AsyncMock(return_value={"final_answer": "should not run"})
    loop._multi_agent_graph = graph

    loop._run_agent_loop = AsyncMock(return_value=(
        "single-agent answer",
        None,
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "single-agent answer"},
        ],
        "stop",
        False,
    ))  # type: ignore[method-assign]

    result = await loop._process_message(
        InboundMessage(channel="feishu", sender_id="u1", chat_id="orchestrator1", content="hello")
    )

    assert result is not None
    assert result.content == "single-agent answer"
    graph.run.assert_not_called()
