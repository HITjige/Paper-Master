from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.builtin import cmd_multi_agent
from nanobot.command.router import CommandContext


@pytest.mark.asyncio
async def test_multi_agent_command_forwards_request_callbacks() -> None:
    response = OutboundMessage(channel="websocket", chat_id="chat-1", content="answer")
    loop = SimpleNamespace(
        _multi_agent_graph=object(),
        process_with_multi_agent=AsyncMock(return_value=response),
    )
    msg = InboundMessage(
        channel="websocket",
        sender_id="user-1",
        chat_id="chat-1",
        content="/multi-agent explain the paper",
        media=["paper.pdf"],
        session_key_override="unified:default",
    )

    async def on_progress(_content: str, **_kwargs) -> None:
        return None

    async def on_stream(_delta: str) -> None:
        return None

    async def on_stream_end(**_kwargs) -> None:
        return None

    ctx = CommandContext(
        msg=msg,
        session=None,
        key="effective:session",
        raw=msg.content,
        args="explain the paper",
        loop=loop,
        on_progress=on_progress,
        on_stream=on_stream,
        on_stream_end=on_stream_end,
        session_summary="summary",
    )

    assert await cmd_multi_agent(ctx) is response
    kwargs = loop.process_with_multi_agent.await_args.kwargs
    assert kwargs["session_key"] == "effective:session"
    assert kwargs["media"] == ["paper.pdf"]
    assert kwargs["on_progress"] is on_progress
    assert kwargs["on_stream"] is on_stream
    assert kwargs["on_stream_end"] is on_stream_end
    assert kwargs["session_summary"] == "summary"


@pytest.mark.asyncio
async def test_multi_agent_slash_command_relays_progress_to_message_bus(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )
    loop._multi_agent_graph = object()
    captured: dict = {}

    async def process_with_multi_agent(**kwargs):
        captured.update(kwargs)
        await kwargs["on_progress"]("retrieving papers")
        return OutboundMessage(
            channel=kwargs["channel"],
            chat_id=kwargs["chat_id"],
            content="answer",
        )

    loop.process_with_multi_agent = process_with_multi_agent
    msg = InboundMessage(
        channel="websocket",
        sender_id="user-1",
        chat_id="chat-2",
        content="/multi-agent explain the paper",
    )

    result = await loop._process_message(msg)
    progress = await loop.bus.consume_outbound()

    assert result is not None
    assert result.content == "answer"
    assert captured["content"] == "explain the paper"
    assert progress.content == "retrieving papers"
    assert progress.metadata["_progress"] is True
    assert progress.metadata["_tool_hint"] is False
