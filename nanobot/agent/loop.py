"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.autocompact import AutoCompact
from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.agent.memory import Consolidator, Dream
from nanobot.agent.runner import _MAX_INJECTIONS_PER_TURN, AgentRunner, AgentRunSpec
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.notebook import NotebookEditTool
from nanobot.agent.tools.paper import (
    KBRetrieveTool,
    PaperIngestTool,
    PaperRerankTool,
    PaperSearchTool,
    PaperSimilarityTool,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.search import GlobTool, GrepTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.self import MyTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.document import extract_documents
from nanobot.utils.helpers import image_placeholder_text
from nanobot.utils.helpers import truncate_text as truncate_text_fn
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE
from nanobot.agent.paper_kb import PaperKbConfig, PaperKnowledgeBase

# Multi-agent system imports
try:
    from nanobot.agent.multi_agent import build_multi_agent_graph
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, ToolsConfig, WebToolsConfig
    from nanobot.cron.service import CronService


UNIFIED_SESSION_KEY = "unified:default"


class _LoopHook(AgentHook):
    """Core hook for the main loop."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
    ) -> None:
        super().__init__(reraise=True)
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        from nanobot.utils.helpers import strip_think

        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean) :]
        if incremental and self._on_stream:
            await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._loop._current_iteration = context.iteration

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            if not self._on_stream:
                thought = self._loop._strip_think(
                    context.response.content if context.response else None
                )
                if thought:
                    await self._on_progress(thought)
            tool_hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            await self._on_progress(tool_hint, tool_hint=True)
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        self._loop._set_tool_context(self._channel, self._chat_id, self._message_id)

    async def after_iteration(self, context: AgentHookContext) -> None:
        u = context.usage or {}
        logger.debug(
            "LLM usage: prompt={} completion={} cached={}",
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            u.get("cached_tokens", 0),
        )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._loop._strip_think(content)


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_USER_TURN_KEY = "pending_user_turn"

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        max_tool_result_chars: int | None = None,
        provider_retry_mode: str = "standard",
        web_config: WebToolsConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        session_ttl_minutes: int = 0,
        hooks: list[AgentHook] | None = None,
        unified_session: bool = False,
        disabled_skills: list[str] | None = None,
        tools_config: ToolsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, ToolsConfig, WebToolsConfig

        _tc = tools_config or ToolsConfig()
        defaults = AgentDefaults()
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        self.context_window_tokens = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        self.context_block_limit = context_block_limit
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        self.web_config = web_config or WebToolsConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.tools_config = _tc
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        self.context = ContextBuilder(
            workspace,
            timezone=timezone,
            disabled_skills=disabled_skills,
        )
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_config=self.web_config,
            max_tool_result_chars=self.max_tool_result_chars,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            disabled_skills=disabled_skills,
        )
        self._unified_session = unified_session
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stacks: dict[str, AsyncExitStack] = {}
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Per-session pending queues for mid-turn message injection.
        # When a session has an active task, new messages for that session
        # are routed here instead of creating a new task.
        self._pending_queues: dict[str, asyncio.Queue] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.consolidator = Consolidator(
            store=self.context.memory,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=self.context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
        )
        self.auto_compact = AutoCompact(
            sessions=self.sessions,
            consolidator=self.consolidator,
            session_ttl_minutes=session_ttl_minutes,
        )
        self.dream = Dream(
            store=self.context.memory,
            provider=provider,
            model=self.model,
        )

        if self.tools_config.paper.enable:
            paper_cfg = self.tools_config.paper
            self.kb = PaperKnowledgeBase(
                workspace=self.workspace,
                config=PaperKbConfig(
                    enabled=paper_cfg.enable,
                    embedding_api_key=paper_cfg.embedding_api_key,
                    embedding_api_base=paper_cfg.embedding_api_base,
                    embedding_model=paper_cfg.embedding_model,
                    retrieval_top_k=paper_cfg.retrieval_top_k,
                    max_chunk_chars=paper_cfg.max_chunk_chars,
                    min_chunk_chars=paper_cfg.min_chunk_chars,
                ),
            )

        self._register_default_tools()
        if _tc.my.enable:
            self.tools.register(MyTool(loop=self, modify_allowed=_tc.my.allow_set))
        self._runtime_vars: dict[str, Any] = {}
        self._current_iteration: int = 0
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)
        
        # Initialize multi-agent graph if paper tools are enabled
        self._multi_agent_graph = None
        if MULTI_AGENT_AVAILABLE and self.tools_config.paper.enable:
            try:
                paper_tools = {
                    "paper_search": self.tools.get("paper_search"),
                    "paper_similarity": self.tools.get("paper_similarity"),
                    "paper_rerank": self.tools.get("paper_rerank"),
                    "paper_ingest": self.tools.get("paper_ingest"),
                    "kb_retrieve": self.tools.get("kb_retrieve"),
                }
                if self.tools_config.paper.enable:
                    self._multi_agent_graph = build_multi_agent_graph(
                        provider=self.provider,
                        kb=self.kb,
                        tools=paper_tools,
                        max_iterations=3,
                        similarity_threshold=0.2,
                        top_k=self.tools_config.paper.auto_context_top_k or 5,
                        ingest_limit=3,
                        memory_store=self.context.memory,
                    )
                    logger.info("Multi-agent graph initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize multi-agent graph: {}", e)
                self._multi_agent_graph = None

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = (
            self.workspace if (self.restrict_to_workspace or self.exec_config.sandbox) else None
        )
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(
            ReadFileTool(
                workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read
            )
        )
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        for cls in (GlobTool, GrepTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(NotebookEditTool(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    sandbox=self.exec_config.sandbox,
                    path_append=self.exec_config.path_append,
                    allowed_env_keys=self.exec_config.allowed_env_keys,
                )
            )
        if self.web_config.enable:
            self.tools.register(
                WebSearchTool(config=self.web_config.search, proxy=self.web_config.proxy)
            )
            self.tools.register(WebFetchTool(proxy=self.web_config.proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )
        if self.tools_config.paper.enable:
            self.tools.register(PaperSearchTool(workspace=self.workspace, kb=self.kb))
            self.tools.register(PaperSimilarityTool(workspace=self.workspace, kb=self.kb))
            self.tools.register(PaperRerankTool(workspace=self.workspace, kb=self.kb))
            self.tools.register(
                PaperIngestTool(
                    workspace=self.workspace,
                    kb=self.kb,
                    provider=self.provider,
                    model=self.model,
                )
            )
            self.tools.register(KBRetrieveTool(workspace=self.workspace, kb=self.kb))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers

        try:
            self._mcp_stacks = await connect_mcp_servers(self._mcp_servers, self.tools)
            if self._mcp_stacks:
                self._mcp_connected = True
            else:
                logger.warning("No MCP servers connected successfully (will retry next message)")
        except asyncio.CancelledError:
            logger.warning("MCP connection cancelled (will retry next message)")
            self._mcp_stacks.clear()
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            self._mcp_stacks.clear()
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron", "my"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think

        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hints with smart abbreviation."""
        from nanobot.utils.tool_hints import format_tool_hints

        return format_tool_hints(tool_calls)

    def _effective_session_key(self, msg: InboundMessage) -> str:
        """Return the session key used for task routing and mid-turn injections."""
        if self._unified_session and not msg.session_key_override:
            return UNIFIED_SESSION_KEY
        return msg.session_key

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
        *,
        session: Session | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> tuple[str | None, list[str], list[dict], str, bool]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.

        Returns (final_content, tools_used, messages, stop_reason, had_injections).
        """
        loop_hook = _LoopHook(
            self,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
        )
        hook: AgentHook = (
            CompositeHook([loop_hook] + self._extra_hooks) if self._extra_hooks else loop_hook
        )

        async def _checkpoint(payload: dict[str, Any]) -> None:
            if session is None:
                return
            self._set_runtime_checkpoint(session, payload)

        async def _drain_pending(*, limit: int = _MAX_INJECTIONS_PER_TURN) -> list[dict[str, Any]]:
            """Non-blocking drain of follow-up messages from the pending queue."""
            if pending_queue is None:
                return []
            items: list[dict[str, Any]] = []
            while len(items) < limit:
                try:
                    pending_msg = pending_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                content = pending_msg.content
                media = pending_msg.media if pending_msg.media else None
                if media:
                    content, media = extract_documents(content, media)
                    media = media or None
                user_content = self.context._build_user_content(content, media)
                runtime_ctx = self.context._build_runtime_context(
                    pending_msg.channel,
                    pending_msg.chat_id,
                    self.context.timezone,
                )
                if isinstance(user_content, str):
                    merged: str | list[dict[str, Any]] = f"{runtime_ctx}\n\n{user_content}"
                else:
                    merged = [{"type": "text", "text": runtime_ctx}] + user_content
                items.append({"role": "user", "content": merged})
            return items

        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=self.tools,
            model=self.model,
            max_iterations=self.max_iterations,
            max_tool_result_chars=self.max_tool_result_chars,
            hook=hook,
            error_message="Sorry, I encountered an error calling the AI model.",
            concurrent_tools=True,
            workspace=self.workspace,
            session_key=session.key if session else None,
            context_window_tokens=self.context_window_tokens,
            context_block_limit=self.context_block_limit,
            provider_retry_mode=self.provider_retry_mode,
            progress_callback=on_progress,
            retry_wait_callback=on_retry_wait,
            checkpoint_callback=_checkpoint,
            injection_callback=_drain_pending,
        ))
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            # Push final content through stream so streaming channels (e.g. Feishu)
            # update the card instead of leaving it empty.
            if on_stream and on_stream_end:
                await on_stream(result.final_content or "")
                await on_stream_end(resuming=False)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages, result.stop_reason, result.had_injections

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                self.auto_compact.check_expired(
                    self._schedule_background,
                    active_session_keys=self._pending_queues.keys(),
                )
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            effective_key = self._effective_session_key(msg)
            # If this session already has an active pending queue (i.e. a task
            # is processing this session), route the message there for mid-turn
            # injection instead of creating a competing task.
            if effective_key in self._pending_queues:
                pending_msg = msg
                if effective_key != msg.session_key:
                    pending_msg = dataclasses.replace(
                        msg,
                        session_key_override=effective_key,
                    )
                try:
                    self._pending_queues[effective_key].put_nowait(pending_msg)
                except asyncio.QueueFull:
                    logger.warning(
                        "Pending queue full for session {}, falling back to queued task",
                        effective_key,
                    )
                else:
                    logger.info(
                        "Routed follow-up message to pending queue for session {}",
                        effective_key,
                    )
                    continue
            # Compute the effective session key before dispatching
            # This ensures /stop command can find tasks correctly when unified session is enabled
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(effective_key, []).append(task)
            task.add_done_callback(
                lambda t, k=effective_key: self._active_tasks.get(k, [])
                and self._active_tasks[k].remove(t)
                if t in self._active_tasks.get(k, [])
                else None
            )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        session_key = self._effective_session_key(msg)
        if session_key != msg.session_key:
            msg = dataclasses.replace(msg, session_key_override=session_key)
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()

        # Register a pending queue so follow-up messages for this session are
        # routed here (mid-turn injection) instead of spawning a new task.
        pending = asyncio.Queue(maxsize=20)
        self._pending_queues[session_key] = pending

        try:
            async with lock, gate:
                try:
                    on_stream = on_stream_end = None
                    if msg.metadata.get("_wants_stream"):
                        # Split one answer into distinct stream segments.
                        stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                        stream_segment = 0

                        def _current_stream_id() -> str:
                            return f"{stream_base_id}:{stream_segment}"

                        async def on_stream(delta: str) -> None:
                            meta = dict(msg.metadata or {})
                            meta["_stream_delta"] = True
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content=delta,
                                metadata=meta,
                            ))

                        async def on_stream_end(*, resuming: bool = False) -> None:
                            nonlocal stream_segment
                            meta = dict(msg.metadata or {})
                            meta["_stream_end"] = True
                            meta["_resuming"] = resuming
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="",
                                metadata=meta,
                            ))
                            stream_segment += 1

                    response = await self._process_message(
                        msg, on_stream=on_stream, on_stream_end=on_stream_end,
                        pending_queue=pending,
                    )
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="", metadata=msg.metadata or {},
                        ))
                except asyncio.CancelledError:
                    logger.info("Task cancelled for session {}", session_key)
                    raise
                except Exception:
                    logger.exception("Error processing message for session {}", session_key)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="Sorry, I encountered an error.",
                    ))
        finally:
            # Drain any messages still in the pending queue and re-publish
            # them to the bus so they are processed as fresh inbound messages
            # rather than silently lost.
            queue = self._pending_queues.pop(session_key, None)
            if queue is not None:
                leftover = 0
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await self.bus.publish_inbound(item)
                    leftover += 1
                if leftover:
                    logger.info(
                        "Re-published {} leftover message(s) to bus for session {}",
                        leftover, session_key,
                    )

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        for name, stack in self._mcp_stacks.items():
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                logger.debug("MCP server '{}' cleanup error (can be ignored)", name)
        self._mcp_stacks.clear()

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def _append_turn_history(self, user_text: str, assistant_text: str) -> None:
        if not (user_text or assistant_text):
            return
        user_snippet = truncate_text_fn(user_text or "", 2000)
        assistant_snippet = truncate_text_fn(assistant_text or "", 2000)
        entry = f"[TURN] USER: {user_snippet}\nASSISTANT: {assistant_snippet}"
        self.context.memory.append_history(entry)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            if self._restore_runtime_checkpoint(session):
                self.sessions.save(session)
            if self._restore_pending_user_turn(session):
                self.sessions.save(session)

            session, pending = self.auto_compact.prepare_session(session, key)

            await self.consolidator.maybe_consolidate_by_tokens(
                session,
                session_summary=pending,
            )
            # Persist subagent follow-ups into durable history BEFORE prompt
            # assembly. ContextBuilder merges adjacent same-role messages for
            # provider compatibility, which previously caused the follow-up to
            # disappear from session.messages while still being visible to the
            # LLM via the merged prompt. See _persist_subagent_followup.
            is_subagent = msg.sender_id == "subagent"
            if is_subagent and self._persist_subagent_followup(session, msg):
                self.sessions.save(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            current_role = "assistant" if is_subagent else "user"

            # Subagent content is already in `history` above; passing it again
            # as current_message would double-project it into the prompt.
            messages = self.context.build_messages(
                history=history,
                current_message="" if is_subagent else msg.content,
                channel=channel,
                chat_id=chat_id,
                session_summary=pending,
                current_role=current_role,
            )
            final_content, _, all_msgs, _, _ = await self._run_agent_loop(
                messages, session=session, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            self._clear_runtime_checkpoint(session)
            self.sessions.save(session)
            self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))
            # self._schedule_background(self.dream.run())
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
            )

        # Extract document text from media at the processing boundary so all
        # channels benefit without format-specific logic in ContextBuilder.
        if msg.media:
            new_content, image_only = extract_documents(msg.content, msg.media)
            msg = dataclasses.replace(msg, content=new_content, media=image_only)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        if self._restore_runtime_checkpoint(session):
            self.sessions.save(session)
        if self._restore_pending_user_turn(session):
            self.sessions.save(session)

        session, pending = self.auto_compact.prepare_session(session, key)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        # --- Define _bus_progress BEFORE the multi-agent check so
        # process_with_multi_agent can relay progress to the bus. ---
        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        # Check if we should use multi-agent workflow
        if self.should_use_multi_agent(msg.content):
            logger.info("Using multi-agent workflow for query: {}", preview)
            return await self.process_with_multi_agent(
                content=msg.content,
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                on_progress=on_progress or _bus_progress,
                session_summary=pending,
            )

        await self.consolidator.maybe_consolidate_by_tokens(
            session,
            session_summary=pending,
        )

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)

        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            session_summary=pending,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        async def _on_retry_wait(content: str) -> None:
            meta = dict(msg.metadata or {})
            meta["_retry_wait"] = True
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        # Persist the triggering user message immediately, before running the
        # agent loop. If the process is killed mid-turn (OOM, SIGKILL, self-
        # restart, etc.), the existing runtime_checkpoint preserves the
        # in-flight assistant/tool state but NOT the user message itself, so
        # the user's prompt is silently lost on recovery. Saving it up front
        # makes recovery possible from the session log alone.
        user_persisted_early = False
        if isinstance(msg.content, str) and msg.content.strip():
            session.add_message("user", msg.content)
            self._mark_pending_user_turn(session)
            self.sessions.save(session)
            user_persisted_early = True

        final_content, _, all_msgs, stop_reason, had_injections = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            on_retry_wait=_on_retry_wait,
            session=session,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            pending_queue=pending_queue,
        )

        if final_content is None or not final_content.strip():
            final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        # Skip the already-persisted user message when saving the turn
        save_skip = 1 + len(history) + (1 if user_persisted_early else 0)
        self._save_turn(session, all_msgs, save_skip)
        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        self.sessions.save(session)
        # self._append_turn_history(msg.content, final_content or "")
        self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))
        # self._schedule_background(self.dream.run())

        # When follow-up messages were injected mid-turn, a later natural
        # language reply may address those follow-ups and should not be
        # suppressed just because MessageTool was used earlier in the turn.
        # However, if the turn falls back to the empty-final-response
        # placeholder, suppress it when the real user-visible output already
        # came from MessageTool.
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == "empty_final_response":
                return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None and stop_reason != "error":
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
        )

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        should_truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if block.get("type") == "image_url" and block.get("image_url", {}).get(
                "url", ""
            ).startswith("data:image/"):
                path = (block.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": image_placeholder_text(path)})
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if should_truncate_text and len(text) > self.max_tool_result_chars:
                    text = truncate_text_fn(text, self.max_tool_result_chars)
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    entry["content"] = truncate_text_fn(content, self.max_tool_result_chars)
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, should_truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the entire runtime-context block (including any session summary).
                    # The block is bounded by _RUNTIME_CONTEXT_TAG and _RUNTIME_CONTEXT_END.
                    end_marker = ContextBuilder._RUNTIME_CONTEXT_END
                    end_pos = content.find(end_marker)
                    if end_pos >= 0:
                        after = content[end_pos + len(end_marker):].lstrip("\n")
                        if after:
                            entry["content"] = after
                        else:
                            continue
                    else:
                        # Fallback: no end marker found, strip the tag prefix
                        after_tag = content[len(ContextBuilder._RUNTIME_CONTEXT_TAG):].lstrip("\n")
                        if after_tag.strip():
                            entry["content"] = after_tag
                        else:
                            continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    def _persist_subagent_followup(self, session: Session, msg: InboundMessage) -> bool:
        """Persist subagent follow-ups before prompt assembly so history stays durable.

        Returns True if a new entry was appended; False if the follow-up was
        deduped (same ``subagent_task_id`` already in session) or carries no
        content worth persisting.
        """
        if not msg.content:
            return False
        task_id = msg.metadata.get("subagent_task_id") if isinstance(msg.metadata, dict) else None
        if task_id and any(
            m.get("injected_event") == "subagent_result" and m.get("subagent_task_id") == task_id
            for m in session.messages
        ):
            return False
        session.add_message(
            "assistant",
            msg.content,
            sender_id=msg.sender_id,
            injected_event="subagent_result",
            subagent_task_id=task_id,
        )
        return True

    def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        """Persist the latest in-flight turn state into session metadata."""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)

    def _mark_pending_user_turn(self, session: Session) -> None:
        session.metadata[self._PENDING_USER_TURN_KEY] = True

    def _clear_pending_user_turn(self, session: Session) -> None:
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )

    def _restore_runtime_checkpoint(self, session: Session) -> bool:
        """Materialize an unfinished turn into session history before a new request."""
        from datetime import datetime

        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

        restored_messages: list[dict[str, Any]] = []
        if isinstance(assistant_message, dict):
            restored = dict(assistant_message)
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(message)
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": name,
                    "content": "Error: Task interrupted before this tool finished.",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        for size in range(max_overlap, 0, -1):
            existing = session.messages[-size:]
            restored = restored_messages[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
        session.messages.extend(restored_messages[overlap:])

        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        return True

    def _restore_pending_user_turn(self, session: Session) -> bool:
        """Close a turn that only persisted the user message before crashing."""
        from datetime import datetime

        if not session.metadata.get(self._PENDING_USER_TURN_KEY):
            return False

        if session.messages and session.messages[-1].get("role") == "user":
            session.messages.append(
                {
                    "role": "assistant",
                    "content": "Error: Task interrupted before a response was generated.",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            session.updated_at = datetime.now()

        self._clear_pending_user_turn(session)
        return True

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        media: list[str] | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel, sender_id="user", chat_id=chat_id,
            content=content, media=media or [],
        )
        return await self._process_message(
            msg,
            session_key=session_key,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
        )

    @staticmethod
    def _extract_savable_state(state: dict[str, Any]) -> dict[str, Any]:
        """Extract JSON-serializable state fields for session persistence.

        Only preserves the minimal subset needed to resume workflow execution,
        discarding ephemeral fields (callbacks, temp references, etc.).
        """
        SAVABLE_KEYS = {
            "research_phase", "papers_for_selection", "search_completed",
            "external_papers", "ingested_papers",
            "routing_decision", "routing_reasoning",
            "retrieval_results", "retrieval_quality",
            "rewritten_queries", "rewrite_reasoning",
            "sub_queries_detail", "extracted_entities",
            "referenced_papers", "requires_clarification",
            "iteration_count", "max_iterations",
            "user_query", "session_id",
            "recent_dialog_context",
            "long_term_memory_context",
            "user_profile_context",
            "soul_context",
            "session_summary_context",
            "draft_answer",
            "rewrite_fallback_used",
        }
        return {k: v for k, v in state.items() if k in SAVABLE_KEYS and v is not None}

    async def process_with_multi_agent(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        session_summary: str | None = None,
    ) -> OutboundMessage | None:
        """Process a message using the multi-agent workflow.

        Supports pause/resume:
        1. If the session contains a paused state (from a previous "select" phase),
           the user's response is treated as paper selection input and the workflow
           resumes via :meth:`MultiAgentGraph.resume`.
        2. Otherwise, starts a fresh workflow.  If after execution the workflow is
           paused at "select" phase (waiting for user to choose papers), the state
           is saved to session metadata for the next turn.

        Args:
            content: User query content
            session_key: Session identifier
            channel: Channel name
            chat_id: Chat identifier
            on_progress: Optional progress callback

        Returns:
            OutboundMessage with the final answer or paper selection UI
        """
        if not self._multi_agent_graph:
            logger.warning("Multi-agent graph not available, falling back to standard processing")
            return await self.process_direct(
                content=content,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                on_progress=on_progress,
            )

        session = self.sessions.get_or_create(session_key)

        # ------------------------------------------------------------------
        # Check for paused state — resume mode
        # ------------------------------------------------------------------
        paused_state = session.metadata.pop("multi_agent_paused_state", None)
        if paused_state is not None:
            logger.info(
                "process_with_multi_agent: found paused state, resuming workflow"
            )
            if on_progress:
                await on_progress("🔄 收到选择，继续处理论文...")

            # Persist the user's selection message so the conversation
            # history stays coherent across turns.
            if content.strip():
                session.add_message("user", content)
                self._mark_pending_user_turn(session)

            # Save the session metadata change (cleared paused state) right away
            self.sessions.save(session)

            result = await self._multi_agent_graph.resume(
                saved_state=paused_state,
                user_input=content,
            )

            final_answer = result.get("final_answer", "")
            routing_decision = result.get("routing_decision", "unknown")
            iteration_count = result.get("iteration_count", 0)

            metadata: dict[str, Any] = {
                "multi_agent": True,
                "resumed": True,
                "routing_decision": routing_decision,
                "iterations": iteration_count,
                "sources_used": [],
            }
            if result.get("retrieval_results"):
                metadata["sources_used"].append("internal_kb")
            if result.get("external_papers"):
                metadata["sources_used"].append("external_search")

            # Build response parts
            response_parts = []
            if iteration_count > 0:
                response_parts.append(
                    f"*[Multi-Agent Workflow: {routing_decision} mode, "
                    f"{iteration_count} iteration(s)]*\n\n"
                )
            response_parts.append(final_answer)

            citations = result.get("citations", [])
            if citations:
                response_parts.append("\n\n**References:**")
                for i, citation in enumerate(citations[:10], 1):
                    response_parts.append(f"\n{i}. {citation}")

            full_response = "".join(response_parts)

            # Persist assistant response into session history
            if final_answer.strip():
                session.add_message("assistant", final_answer)
                self._clear_pending_user_turn(session)
                self.sessions.save(session)
                self._schedule_background(
                    self.consolidator.maybe_consolidate_by_tokens(session)
                )

            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=full_response,
                metadata=metadata,
            )

        # ------------------------------------------------------------------
        # Normal (non-resume) mode — fresh workflow execution
        # ------------------------------------------------------------------
        try:
            if on_progress:
                await on_progress("🔄 Starting multi-agent workflow...")

            # Persist user message before running the workflow.
            # This ensures the user's input is durable even if the process
            # crashes mid-workflow.
            if content.strip():
                session.add_message("user", content)
                self._mark_pending_user_turn(session)
                self.sessions.save(session)

            history = session.get_history(max_messages=0)

            recent_dialog = self._format_recent_dialog(history, max_turns=500)
            node_history_chars = int(self.tools_config.paper.multi_agent_node_history_chars or 0)
            recent_dialog = self._truncate_context(recent_dialog, node_history_chars)

            long_term_memory = self.context.memory.read_memory()
            user_profile = self.context.memory.read_user()
            soul_context = self.context.memory.read_soul()
            session_summary_context = session_summary or ""
            long_term_memory = self._truncate_context(long_term_memory, node_history_chars)
            user_profile = self._truncate_context(user_profile, node_history_chars)
            soul_context = self._truncate_context(soul_context, node_history_chars)
            session_summary_context = self._truncate_context(session_summary_context, node_history_chars)

            result = await self._multi_agent_graph.run(
                user_query=content,
                session_id=session_key,
                progress_callback=on_progress,
                recent_dialog_context=recent_dialog or "(empty)",
                long_term_memory_context=long_term_memory or "(empty)",
                user_profile_context=user_profile or "(empty)",
                soul_context=soul_context or "(empty)",
                session_summary_context=session_summary_context or "(empty)",
            )

            # ------------------------------------------------------------------
            # Pause check: if workflow stopped at "select" phase for user input
            # ------------------------------------------------------------------
            if (
                result.get("research_phase") == "select"
                and result.get("papers_for_selection")
            ):
                # Save minimal state so the next turn can resume
                savable = self._extract_savable_state(result)
                session.metadata["multi_agent_paused_state"] = savable
                self.sessions.save(session)

                logger.info(
                    "process_with_multi_agent: paused at select phase, "
                    "saved state with {} papers for selection",
                    len(result["papers_for_selection"]),
                )

                if on_progress:
                    await on_progress("⏸️ 工作流暂停，等待用户选择论文")

                # The draft_answer from the workflow IS the selection UI
                selection_ui = result.get("final_answer", "") or result.get("draft_answer", "")
                if not selection_ui:
                    selection_ui = (
                        "📚 Paper search completed. Please choose which papers "
                        "to ingest (reply with paper IDs, `skip`, or `all`)."
                    )

                # Persist the selection UI as an assistant message so the
                # conversation history stays coherent and the pending user
                # turn is properly closed — avoids the "Task interrupted
                # before a response was generated" error on the next turn.
                if selection_ui.strip():
                    session.add_message("assistant", selection_ui)
                    self._clear_pending_user_turn(session)
                    self.sessions.save(session)

                return OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content=selection_ui,
                    metadata={
                        "multi_agent": True,
                        "awaiting_selection": True,
                        "routing_decision": result.get("routing_decision", "unknown"),
                    },
                )

            # ------------------------------------------------------------------
            # Normal completion — format and return final answer
            # ------------------------------------------------------------------
            final_answer = result.get("final_answer", "")
            routing_decision = result.get("routing_decision", "unknown")
            iteration_count = result.get("iteration_count", 0)

            metadata = {
                "multi_agent": True,
                "routing_decision": routing_decision,
                "iterations": iteration_count,
                "sources_used": [],
            }

            if result.get("retrieval_results"):
                metadata["sources_used"].append("internal_kb")
            if result.get("external_papers"):
                metadata["sources_used"].append("external_search")

            response_parts = []

            if iteration_count > 0:
                response_parts.append(
                    f"*[Multi-Agent Workflow: {routing_decision} mode, "
                    f"{iteration_count} iteration(s)]*\n\n"
                )

            response_parts.append(final_answer)

            citations = result.get("citations", [])
            if citations:
                response_parts.append("\n\n**References:**")
                for i, citation in enumerate(citations[:10], 1):
                    response_parts.append(f"\n{i}. {citation}")

            full_response = "".join(response_parts)

            # Persist assistant response into session history
            if final_answer.strip():
                session.add_message("assistant", final_answer)
                self._clear_pending_user_turn(session)
                self.sessions.save(session)
                self._schedule_background(
                    self.consolidator.maybe_consolidate_by_tokens(session)
                )
            # self._append_turn_history(content, final_answer)
            # self._schedule_background(self.dream.run())
            # self._schedule_skill_extraction(result, content)

            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=full_response,
                metadata=metadata,
            )

        except Exception as e:
            logger.error("Multi-agent workflow failed: {}", e)
            # Fallback to standard processing
            return await self.process_direct(
                content=content,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                on_progress=on_progress,
            )
    
    async def kb_ingest_local(
        self,
        doc: dict[str, Any],
        local_pdf_path: str,
    ) -> dict[str, Any]:
        """Ingest a local PDF file into the knowledge base.

        Parses with MinerULoader (PDF → markdown, matching PaperIngestTool),
        saves .md alongside the PDF for inspection, extracts the paper title
        from the first ``# `` heading, then runs the full HyDE pipeline
        (semantic chunking → LLM metadata → Chroma upsert).

        Args:
            doc: Paper metadata dict with keys paper_id, title, source, url, year.
            local_pdf_path: Absolute path to the saved PDF file.

        Returns:
            Dict with status, paper_id, chunk_count, question_count, local_md.
        """
        if not self.tools_config.paper.enable or not hasattr(self, "kb"):
            return {"status": "error", "error": "Knowledge base not enabled"}

        from nanobot.agent.tools.paper import (
            MINERU_TOKEN,
            _extract_assets_and_strip,
            _extract_figure_table_blocks,
            _extract_front_matter,
            _generate_chunk_metadata,
            _merge_split_paragraphs,
            _normalize_blank_lines,
            _remove_noisy_blocks,
            _save_asset_kv,
            _split_markdown_semantic,
            _strip_front_matter,
        )

        pdf_path = Path(local_pdf_path)
        text_content: str = ""

        # 1. Parse PDF using MinerULoader (precise markdown output).
        try:
            from langchain_mineru import MinerULoader

            loader = MinerULoader(
                source=str(pdf_path),
                language="en",
                mode="precision",
                token=MINERU_TOKEN,
            )
            parsed = loader.load()
            text_content = parsed[0].page_content if parsed else ""
        except Exception:
            logger.warning(
                "MinerULoader failed for {}, falling back to extract_text",
                pdf_path,
            )

        if not text_content:
            from nanobot.utils.document import extract_text

            text_content = extract_text(pdf_path)
            text_content = text_content if isinstance(text_content, str) else ""

        assets_path = self.workspace / "kb" / "figures.jsonl"
        paper_id = str(doc.get("paper_id", "paper"))

        text_content = _remove_noisy_blocks(text_content) if text_content else ""
        # Extract and remove figure/table blocks (images + caption → KV),
        # then merge paragraphs broken across pages.
        text_content, assets_by_key = _extract_figure_table_blocks(
            text_content, paper_id=paper_id,
        )
        text_content = _merge_split_paragraphs(text_content)
        if assets_by_key:
            _save_asset_kv(assets_path, assets_by_key, paper_id=paper_id)
        if not text_content or not text_content.strip():
            return {"status": "error", "error": "failed_to_parse_content"}

        # Save front matter before stripping (for metadata extraction)
        raw_front_matter = _extract_front_matter(text_content)
        # Strip front matter (title/authors/abstract before Introduction)
        body_text = _strip_front_matter(text_content)
        # Final cleanup: discard any remaining image links / HTML tables
        body_text, _ = _extract_assets_and_strip(body_text, paper_id=paper_id)
        body_text = _normalize_blank_lines(body_text)

        # Extract title from front matter's first ``# `` heading
        if raw_front_matter:
            m = re.search(r"^#\s+(.+)$", raw_front_matter, re.MULTILINE)
            if m:
                doc["title"] = m.group(1).strip()[:200]

        # Enrich metadata from front matter when arXiv data is missing
        if (not doc.get("authors") or not doc.get("abstract")) and raw_front_matter:
            from nanobot.agent.tools.paper import _parse_front_matter_metadata
            fm_meta = await _parse_front_matter_metadata(
                raw_front_matter, self.provider, self.model,
            )
            if fm_meta.get("title"):
                doc["title"] = fm_meta["title"]
            if fm_meta.get("authors"):
                doc["authors"] = fm_meta["authors"]
            if fm_meta.get("abstract"):
                doc["abstract"] = fm_meta["abstract"]
            if fm_meta.get("year"):
                doc["year"] = fm_meta["year"]

        # 2. Save .md (body only, without front matter) for inspection.
        md_path = pdf_path.with_suffix(".md")
        md_path.write_text(body_text, encoding="utf-8")
        pdf_path.unlink(missing_ok=True)

        # 3. Semantic chunking on body text.
        semantic_chunks = _split_markdown_semantic(
            body_text,
            max_chunk_chars=self.kb.config.max_chunk_chars,
            min_chunk_chars=self.kb.config.min_chunk_chars,
        )
        if not semantic_chunks:
            return {"status": "error", "error": "no_chunks_generated"}

        # 4. Generate HyDE metadata for each chunk.
        chunk_metadata: list[dict[str, Any]] = []
        for chunk in semantic_chunks:
            meta = await _generate_chunk_metadata(
                text=chunk.get("text", ""),
                section=chunk.get("section", "content"),
                provider=self.provider,
                model=self.model,
                num_questions=self.kb.config.num_hypothetical_questions,
                summarize=True,
                title=doc.get("title", ""),
                abstract=doc.get("abstract", ""),
                paper_id=paper_id,
                assets_by_key=assets_by_key,
            )
            chunk_metadata.append(meta)

        # 6. Upsert to Chroma.
        result = await self.kb.upsert_semantic_chunks(
            doc=doc,
            semantic_chunks=semantic_chunks,
            chunk_metadata=chunk_metadata,
        )
        result["status"] = "ok"
        result["local_md"] = str(md_path)
        return result

    def _schedule_skill_extraction(self, result: dict[str, Any], user_query: str) -> None:
        """Fire-and-forget background task: extract reusable skills from paper Q&A.
        
        Runs Phase 1 (LLM analysis) + Phase 2 (AgentRunner with file tools)
        to determine if this multi-agent discussion produced domain expertise
        worth preserving as a Skill. If so, creates or updates SKILL.md files
        under workspace/skills/.
        """
        if not self.tools_config.paper.enable:
            return
        try:
            from nanobot.agent.skill_extractor import SkillExtractor

            result["user_query"] = user_query
            extractor = SkillExtractor(
                store=self.context.memory,
                provider=self.provider,
                model=self.model,
                workspace=self.workspace,
            )
            self._schedule_background(extractor.extract(result))
        except Exception:
            logger.exception("Failed to schedule skill extraction")

    def should_use_multi_agent(self, content: str) -> bool:
        """Determine if a query should use the multi-agent workflow.
        
        Heuristics:
        - Query mentions papers, research, arXiv, or academic topics
        - Query is complex (multiple questions or comparisons)
        - Query asks for latest/recent papers
        
        Args:
            content: User query
            
        Returns:
            True if multi-agent should be used
        """
        if not self._multi_agent_graph:
            return False

        return True
        
        content_lower = content.lower()
        
        # Keywords indicating paper-related queries
        paper_keywords = [
            "paper", "论文", "文献", "研究", "research",
            "arxiv", "publication", "survey", "review",
            "method", "algorithm", "model", "architecture",
            "transformer", "mamba", "llm", "gpt", "bert",
            "neural", "deep learning", "machine learning",
        ]
        
        # Check for paper-related keywords
        has_paper_keyword = any(kw in content_lower for kw in paper_keywords)
        
        # Check for complex queries (comparison, analysis, summary)
        complex_indicators = [
            "compare", "对比", "比较", "vs", "versus",
            "summarize", "总结", "综述", "分析",
            "latest", "最新", "recent", "最近",
            "difference", "区别", "差异",
        ]
        has_complex_indicator = any(ind in content_lower for ind in complex_indicators)
        
        # Use multi-agent if it's paper-related or complex
        return has_paper_keyword or has_complex_indicator

    @staticmethod
    def _truncate_context(text: str, max_chars: int) -> str:
        if not text or max_chars <= 0:
            return text
        return text[:max_chars]

    @staticmethod
    def _format_recent_dialog(history: list[dict[str, Any]], max_turns: int = 500) -> str:
        if max_turns <= 0:
            return ""
        collected: list[str] = []
        user_turns = 0
        for message in reversed(history):
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = message.get("content", "")
            if not content:
                continue
            collected.append(f"{role.upper()}: {content}")
            if role == "user":
                user_turns += 1
                if user_turns >= max_turns:
                    break
        if not collected:
            return ""
        collected.reverse()
        return "\n".join(collected)
