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
from nanobot.agent.paper_kb import PaperKbConfig, PaperKnowledgeBase
from nanobot.agent.runner import _MAX_INJECTIONS_PER_TURN, AgentRunner, AgentRunSpec
from nanobot.agent.skill_lifecycle import SkillCandidateManager, SkillUsageStore
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
from nanobot.agent.tools.self import MyTool
from nanobot.agent.tools.shell import ExecTool
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
from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
    external_search_confirmation_prompt,
)

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
_PAPER_TOOL_NAMES = frozenset({
    "paper_search",
    "paper_similarity",
    "paper_rerank",
    "paper_ingest",
    "kb_retrieve",
})
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
        # Tool-call responses from reasoning models can contain only the
        # template separator (usually ``"\n\n"``).  Do not create a visible
        # stream until the model has produced non-whitespace content.  Once a
        # stream has started, whitespace-only deltas are still forwarded so
        # normal word spacing is preserved.
        if (
            incremental
            and new_clean.strip()
            and self._on_stream
        ):
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
            "LLM usage: prompt={} completion={} cached={} finish_reason={} "
            "reasoning_chars={} tool_calls={}",
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            u.get("cached_tokens", 0),
            context.response.finish_reason if context.response else "unknown",
            len(context.response.reasoning_content or "") if context.response else 0,
            len(context.tool_calls),
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
    _MULTI_AGENT_LAST_ROUTING_KEY = "multi_agent_last_routing"
    _MULTI_AGENT_LAST_SOURCES_KEY = "multi_agent_last_sources"
    _MULTI_AGENT_ACTIVE_PAPERS_KEY = "multi_agent_active_papers"
    _MULTI_AGENT_PRESENTED_PAPER_IDS_KEY = "multi_agent_presented_paper_ids"
    _MULTI_AGENT_LAST_SEARCH_TOPIC_KEY = "multi_agent_last_search_topic"
    _MULTI_AGENT_CONTEXT_MESSAGE_COUNT_KEY = "multi_agent_context_message_count"

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
        memory_config: Any | None = None,
        skill_config: Any | None = None,
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
        self.skill_config = skill_config or defaults.skills
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        self.context = ContextBuilder(
            workspace,
            timezone=timezone,
            disabled_skills=disabled_skills,
            context_window_tokens=self.context_window_tokens,
            max_completion_tokens=(
                provider.generation.max_tokens
                if isinstance(provider.generation.max_tokens, int)
                else defaults.max_tokens
            ),
            context_block_limit=self.context_block_limit,
            memory_config=memory_config or defaults.memory,
        )
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self.skill_candidates = SkillCandidateManager(
            workspace,
            auto_promote=self.skill_config.auto_promote,
            max_skill_chars=self.skill_config.max_skill_chars,
            max_skill_lines=self.skill_config.max_skill_lines,
        )
        self.skill_usage = (
            SkillUsageStore(workspace) if self.skill_config.track_usage else None
        )
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
            skill_usage=self.skill_usage,
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
            skill_candidates=self.skill_candidates,
        )

        if self.tools_config.paper.enable:
            paper_cfg = self.tools_config.paper
            self._mineru_api_token = paper_cfg.mineru_api_token
            self.kb = PaperKnowledgeBase(
                workspace=self.workspace,
                config=PaperKbConfig(
                    enabled=paper_cfg.enable,
                    embedding_api_key=paper_cfg.embedding_api_key,
                    embedding_api_base=paper_cfg.embedding_api_base,
                    embedding_model=paper_cfg.embedding_model,
                    embedding_fallback=paper_cfg.embedding_fallback,
                    embedding_batch_size=paper_cfg.embedding_batch_size,
                    rerank_model=paper_cfg.rerank_model,
                    rerank_score_mode=paper_cfg.rerank_score_mode,
                    retrieval_relevance_filter_enabled=(
                        paper_cfg.retrieval_relevance_filter_enabled
                    ),
                    retrieval_min_relevance_score=(
                        paper_cfg.retrieval_min_relevance_score
                    ),
                    retrieval_rerank_candidate_count=(
                        paper_cfg.retrieval_rerank_candidate_count
                    ),
                    retrieval_relevance_fail_closed=(
                        paper_cfg.retrieval_relevance_fail_closed
                    ),
                    metadata_concurrency=paper_cfg.metadata_concurrency,
                    rrf_k=paper_cfg.rrf_k,
                    dense_rrf_weight=paper_cfg.dense_rrf_weight,
                    sparse_rrf_weight=paper_cfg.sparse_rrf_weight,
                    bm25_title_weight=paper_cfg.bm25_title_weight,
                    bm25_keywords_weight=paper_cfg.bm25_keywords_weight,
                    bm25_summary_weight=paper_cfg.bm25_summary_weight,
                    bm25_questions_weight=paper_cfg.bm25_questions_weight,
                    bm25_body_weight=paper_cfg.bm25_body_weight,
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
                        model=self.model,
                        provider_retry_mode=self.provider_retry_mode,
                        context_window_tokens=self.context_window_tokens,
                        max_completion_tokens=self.provider.generation.max_tokens,
                        max_iterations=3,
                        similarity_threshold=paper_cfg.retrieval_min_relevance_score,
                        top_k=paper_cfg.retrieval_top_k,
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
            self.tools.register(
                PaperSearchTool(
                    workspace=self.workspace,
                    kb=self.kb,
                    provider=self.provider,
                    model=self.model,
                )
            )
            self.tools.register(PaperSimilarityTool(workspace=self.workspace, kb=self.kb))
            self.tools.register(PaperRerankTool(workspace=self.workspace, kb=self.kb))
            self.tools.register(
                PaperIngestTool(
                    workspace=self.workspace,
                    kb=self.kb,
                    provider=self.provider,
                    model=self.model,
                    mineru_api_token=self._mineru_api_token,
                    mineru_language=self.tools_config.paper.mineru_language,
                    enable_pdf_ocr=self.tools_config.paper.enable_pdf_ocr,
                    ocr_language=self.tools_config.paper.ocr_language,
                    ocr_max_pages=self.tools_config.paper.ocr_max_pages,
                    max_pdf_text_chars=self.tools_config.paper.max_pdf_text_chars,
                )
            )
            self.tools.register(
                KBRetrieveTool(
                    workspace=self.workspace,
                    kb=self.kb,
                    provider=self.provider,
                    model=self.model,
                )
            )

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

        activated_skills: set[str] = set()
        skill_run_id = self.skill_usage.new_run_id() if self.skill_usage else None

        def _record_skill_activation(path: str) -> None:
            if self.skill_usage is None:
                return
            identified = self.skill_usage.identify(path)
            if identified is None or identified[0] in activated_skills:
                return
            name = self.skill_usage.record_activation(
                path,
                session_key=session.key if session else None,
                run_id=skill_run_id,
            )
            if name:
                activated_skills.add(name)

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
            skill_activation_callback=_record_skill_activation,
        ))
        if self.skill_usage is not None:
            self.skill_usage.record_outcome(
                activated_skills,
                success=(
                    result.stop_reason == "completed"
                    and bool((result.final_content or "").strip())
                ),
                session_key=session.key if session else None,
                run_id=skill_run_id,
            )
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

        # Define the bus-backed progress callback before slash-command
        # dispatch so command handlers such as /multi-agent can relay the same
        # progress events as automatically routed requests.
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

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(
            msg=msg,
            session=session,
            key=key,
            raw=raw,
            loop=self,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            session_summary=pending,
        )
        if result := await self.commands.dispatch(ctx):
            return result

        # Choose the execution path using both the current query and recent
        # paper-session context. A lexical check alone cannot recognize
        # follow-ups such as "该模型的性能呢？".
        use_multi_agent, orchestrator_context = (
            await self._decide_multi_agent_with_orchestrator(msg.content, session)
        )
        if use_multi_agent:
            logger.info(
                "Using multi-agent workflow for query: {} (reason={}, confidence={})",
                preview,
                orchestrator_context.get("orchestrator_reasoning", ""),
                orchestrator_context.get("orchestrator_confidence", 0.0),
            )
            return await self.process_with_multi_agent(
                content=msg.content,
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                media=msg.media if msg.media else None,
                on_progress=on_progress or _bus_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                session_summary=pending,
                orchestrator_context=orchestrator_context,
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
        if self._has_paper_novelty_signal(msg.content):
            presented_ids = list(session.metadata.get(
                self._MULTI_AGENT_PRESENTED_PAPER_IDS_KEY,
                [],
            ) or [])
            if not presented_ids:
                for history_message in reversed(history):
                    if history_message.get("role") != "assistant":
                        continue
                    presented_ids.extend(re.findall(
                        r"\[([A-Za-z0-9][A-Za-z0-9._:/-]{1,127})\](?!\()",
                        str(history_message.get("content") or ""),
                    ))
                    if len(presented_ids) >= 50:
                        break
            presented_ids = list(dict.fromkeys(
                str(paper_id).strip()
                for paper_id in presented_ids
                if str(paper_id).strip()
            ))[-50:]
            if presented_ids:
                novelty_context = (
                    "<paper_novelty_context>\n"
                    "The user asked for other/additional papers. Previously presented "
                    f"paper IDs: {json.dumps(presented_ids, ensure_ascii=False)}.\n"
                    "When calling kb_retrieve or paper_search, pass these IDs via "
                    "exclude_paper_ids unless the user explicitly requested one of them.\n"
                    "</paper_novelty_context>"
                )
                for message in reversed(initial_messages):
                    if message.get("role") != "user":
                        continue
                    content = message.get("content")
                    if isinstance(content, str):
                        message["content"] = f"{content}\n\n{novelty_context}"
                    elif isinstance(content, list):
                        message["content"] = [
                            *content,
                            {"type": "text", "text": novelty_context},
                        ]
                    break

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

        final_content, tools_used, all_msgs, stop_reason, had_injections = await self._run_agent_loop(
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
        paper_tools_used = [name for name in (tools_used or []) if name in _PAPER_TOOL_NAMES]
        if paper_tools_used:
            paper_refs = self._paper_references_from_tool_messages(all_msgs)
            self._remember_multi_agent_context(
                session,
                {
                    "routing_decision": "single_agent_paper_tools",
                    "retrieval_results": paper_refs,
                    "citations": [],
                    "final_answer": final_content,
                    "resolved_topic": msg.content,
                    "discovery_request": self._has_explicit_paper_signal(msg.content),
                },
                paper_tools_used,
            )
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
        # ``empty_final_response`` is synthesized after the model stream has
        # ended, so it was not delivered as deltas.  Leave it unmarked and let
        # ChannelManager send the fallback as a regular message.
        if (
            on_stream is not None
            and stop_reason not in {"error", "empty_final_response"}
        ):
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
            if m.get("_nanobot_transient"):
                continue
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
            "retrieval_results", "retrieval_quality", "embedding_status",
            "rewritten_queries", "rewrite_reasoning",
            "sub_queries_detail", "extracted_entities",
            "referenced_papers", "requires_clarification",
            "explicit_paper_ids", "external_search_requested",
            "external_search_authorized", "awaiting_external_search_confirmation",
            "novelty_required", "discovery_request",
            "presented_paper_ids", "last_search_topic", "resolved_topic",
            "external_search_completed", "post_research_retrieval",
            "critic_triggered_research",
            "research_outcome", "novelty_excluded_count",
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

    async def _pause_multi_agent_result(
        self,
        *,
        session: Session,
        result: dict[str, Any],
        channel: str,
        chat_id: str,
        on_progress: Callable[[str], Awaitable[None]] | None,
        on_stream: Callable[[str], Awaitable[None]] | None,
        on_stream_end: Callable[..., Awaitable[None]] | None,
    ) -> OutboundMessage | None:
        """Persist and deliver either external-search consent or paper selection UI."""
        phase = str(result.get("research_phase") or "")
        if phase == "confirm_search":
            response = str(
                result.get("final_answer")
                or result.get("draft_answer")
                or external_search_confirmation_prompt(
                    str(result.get("user_query") or "")
                )
            ).strip()
            progress_text = "⏸️ 工作流暂停，等待用户确认是否进行外部搜索"
            metadata: dict[str, Any] = {
                "multi_agent": True,
                "awaiting_external_search_confirmation": True,
                "routing_decision": result.get("routing_decision", "unknown"),
            }
            sources = ["internal_kb"] if result.get("retrieval_results") else []
        elif phase == "select" and result.get("papers_for_selection"):
            response = str(
                result.get("final_answer") or result.get("draft_answer") or ""
            ).strip()
            if not response:
                response = (
                    "📚 Paper search completed. Please choose which papers "
                    "to ingest (reply with paper IDs, `skip`, or `all`)."
                )
            progress_text = "⏸️ 工作流暂停，等待用户选择论文"
            metadata = {
                "multi_agent": True,
                "awaiting_selection": True,
                "routing_decision": result.get("routing_decision", "unknown"),
            }
            sources = ["external_search"]
        else:
            return None

        session.metadata["multi_agent_paused_state"] = self._extract_savable_state(result)
        if response:
            session.add_message("assistant", response)
        self._clear_pending_user_turn(session)
        self._remember_multi_agent_context(session, result, sources)
        self.sessions.save(session)

        logger.info(
            "process_with_multi_agent: paused at {} phase",
            phase,
        )
        if on_progress:
            await on_progress(progress_text)
        if await self._deliver_multi_agent_stream(response, on_stream, on_stream_end):
            metadata["_streamed"] = True
        return OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=response,
            metadata=metadata,
        )

    def _build_multi_agent_context_snapshot(
        self,
        *,
        content: str,
        history: list[dict[str, Any]],
        channel: str,
        chat_id: str,
        session_summary: str | None,
        media: list[str] | None,
    ) -> dict[str, Any]:
        """Build one bounded context snapshot shared by all graph nodes.

        The snapshot is taken before the current user message is persisted, so
        ``recent_dialog_context`` never repeats ``user_query``. Legacy memory
        slots carry inclusion markers to avoid duplicating the common system
        context; direct graph callers can still populate those slots normally.
        """
        node_history_chars = int(
            self.tools_config.paper.multi_agent_node_history_chars or 0
        )
        recent_dialog = self._truncate_context(
            self._format_recent_dialog(history, max_turns=500),
            node_history_chars,
        )
        recent_dialog = self.context.context_budget.truncate_text(
            recent_dialog,
            max(256, int(self.context.context_budget.budget.prompt_tokens * 0.12)),
            keep_tail=True,
        )
        router_short = self._truncate_context(
            self._format_recent_dialog(
                history,
                max_turns=self.tools_config.paper.router_short_history_turns,
            ),
            self.tools_config.paper.router_short_memory_chars,
        )

        long_term_memory = self._truncate_context(
            self.context.memory.read_memory(), node_history_chars
        )
        user_profile = self._truncate_context(
            self.context.memory.read_user(), node_history_chars
        )
        soul_context = self._truncate_context(
            self.context.memory.read_soul(), node_history_chars
        )
        session_summary_context = self._truncate_context(
            session_summary or "", node_history_chars
        )
        session_summary_context = self.context.context_budget.truncate_text(
            session_summary_context,
            max(256, int(self.context.context_budget.budget.prompt_tokens * 0.10)),
            keep_tail=True,
        )
        memory_scope = f"{channel}:{chat_id}" if channel and chat_id else None
        shared_system_context = self.context.build_system_prompt(
            channel=channel,
            memory_query=content,
            memory_scope=memory_scope,
        )
        runtime_context = self.context._build_runtime_context(
            channel,
            chat_id,
            self.context.timezone,
            session_summary=session_summary_context,
        )

        media_context: list[dict[str, Any]] = []
        if media:
            built_media = self.context._build_user_content("", media)
            if isinstance(built_media, list):
                media_context = [
                    block
                    for block in built_media
                    if isinstance(block, dict) and block.get("type") != "text"
                ]

        return {
            "shared_system_context": shared_system_context,
            "runtime_context": runtime_context,
            "media_context": media_context,
            "router_memory_short": router_short or "(empty)",
            "router_memory_long": self._truncate_context(
                session_summary_context or long_term_memory,
                self.tools_config.paper.router_long_memory_chars,
            ) or "(empty)",
            "recent_dialog_context": recent_dialog or "(empty)",
            "long_term_memory_context": (
                "(included in shared system context)" if long_term_memory else "(empty)"
            ),
            "user_profile_context": (
                "(included in shared system context)" if user_profile else "(empty)"
            ),
            "soul_context": (
                "(included in shared system context)" if soul_context else "(empty)"
            ),
            "session_summary_context": (
                "(included in runtime context)" if session_summary_context else "(empty)"
            ),
        }

    @staticmethod
    async def _deliver_multi_agent_stream(
        content: str,
        on_stream: Callable[[str], Awaitable[None]] | None,
        on_stream_end: Callable[..., Awaitable[None]] | None,
    ) -> bool:
        """Deliver the reviewed graph result through the normal stream channel.

        Multi-agent synthesis must finish critic review before anything is
        user-visible, so the reviewed response is emitted as one final delta.
        """
        if on_stream is None or not content:
            return False
        delivered = False
        try:
            await on_stream(content)
            delivered = True
            if on_stream_end is not None:
                await on_stream_end(resuming=False)
        except Exception as exc:
            logger.debug("Multi-agent stream delivery failed: {}", exc)
        return delivered

    async def process_with_multi_agent(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        media: list[str] | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        session_summary: str | None = None,
        orchestrator_context: dict[str, Any] | None = None,
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
                media=media,
                on_progress=on_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
            )

        session = self.sessions.get_or_create(session_key)
        await self.consolidator.maybe_consolidate_by_tokens(
            session,
            session_summary=session_summary,
        )
        history = session.get_history(max_messages=0)

        # ------------------------------------------------------------------
        # Check for paused state — resume mode
        # ------------------------------------------------------------------
        paused_state = session.metadata.get("multi_agent_paused_state")
        if paused_state is not None and self._has_paper_novelty_signal(content):
            # A user may decline the selection UI implicitly by asking for
            # more/other papers. Treat that as a fresh discovery turn instead
            # of feeding it to the paper-selection parser.
            session.metadata.pop("multi_agent_paused_state", None)
            self.sessions.save(session)
            paused_state = None
            logger.info(
                "process_with_multi_agent: abandoned paused selection for a fresh novelty query"
            )
        elif paused_state is not None:
            session.metadata.pop("multi_agent_paused_state", None)
        if paused_state is not None:
            logger.info(
                "process_with_multi_agent: found paused state, resuming workflow"
            )
            if on_progress:
                await on_progress("🔄 收到选择，继续处理论文...")

            resume_query = "\n".join(filter(None, [
                str(paused_state.get("user_query") or "").strip(),
                content.strip(),
            ]))
            context_snapshot = self._build_multi_agent_context_snapshot(
                content=resume_query,
                history=history,
                channel=channel,
                chat_id=chat_id,
                session_summary=session_summary,
                media=media,
            )
            paused_state = {**paused_state, **context_snapshot}

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
                progress_callback=on_progress,
            )

            paused_response = await self._pause_multi_agent_result(
                session=session,
                result=result,
                channel=channel,
                chat_id=chat_id,
                on_progress=on_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
            )
            if paused_response is not None:
                return paused_response

            final_answer = str(result.get("final_answer") or "").strip()
            if not final_answer:
                final_answer = EMPTY_FINAL_RESPONSE_MESSAGE
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
            persisted_answer = self._multi_agent_persisted_answer(
                final_answer=final_answer,
                citations=citations,
                full_response=full_response,
            )
            session.add_message("assistant", persisted_answer)
            self._clear_pending_user_turn(session)
            self._remember_multi_agent_context(session, result, metadata["sources_used"])
            self.sessions.save(session)
            self._schedule_background(
                self.consolidator.maybe_consolidate_by_tokens(session)
            )
            self._schedule_skill_extraction(
                result,
                str(result.get("user_query") or paused_state.get("user_query", "")),
            )
            if await self._deliver_multi_agent_stream(
                full_response, on_stream, on_stream_end
            ):
                metadata["_streamed"] = True

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

            context_snapshot = self._build_multi_agent_context_snapshot(
                content=content,
                history=history,
                channel=channel,
                chat_id=chat_id,
                session_summary=session_summary,
                media=media,
            )

            last_routing_decision = str(
                session.metadata.get(self._MULTI_AGENT_LAST_ROUTING_KEY) or "none"
            )
            gate_context = dict(orchestrator_context or {})
            active_papers = session.metadata.get(
                self._MULTI_AGENT_ACTIVE_PAPERS_KEY, []
            )
            presented_paper_ids = session.metadata.get(
                self._MULTI_AGENT_PRESENTED_PAPER_IDS_KEY, []
            )
            last_search_topic = str(
                session.metadata.get(self._MULTI_AGENT_LAST_SEARCH_TOPIC_KEY) or ""
            )

            result = await self._multi_agent_graph.run(
                user_query=content,
                session_id=session_key,
                progress_callback=on_progress,
                **context_snapshot,
                last_routing_decision=last_routing_decision,
                routing_context={
                    "active_papers": active_papers,
                    **gate_context,
                },
                referenced_papers=active_papers,
                presented_paper_ids=presented_paper_ids,
                last_search_topic=last_search_topic,
                orchestrator_decision=gate_context.get("orchestrator_decision", "multi_agent"),
                orchestrator_reasoning=gate_context.get("orchestrator_reasoning", ""),
                orchestrator_confidence=float(
                    gate_context.get("orchestrator_confidence", 0.0) or 0.0
                ),
                retrieval_judge_margin=self.tools_config.paper.multi_agent_retrieval_judge_margin,
            )

            paused_response = await self._pause_multi_agent_result(
                session=session,
                result=result,
                channel=channel,
                chat_id=chat_id,
                on_progress=on_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
            )
            if paused_response is not None:
                return paused_response

            # ------------------------------------------------------------------
            # Normal completion — format and return final answer
            # ------------------------------------------------------------------
            final_answer = str(result.get("final_answer") or "").strip()
            if not final_answer:
                final_answer = EMPTY_FINAL_RESPONSE_MESSAGE
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
            persisted_answer = self._multi_agent_persisted_answer(
                final_answer=final_answer,
                citations=citations,
                full_response=full_response,
            )
            session.add_message("assistant", persisted_answer)
            self._clear_pending_user_turn(session)
            self._remember_multi_agent_context(session, result, metadata["sources_used"])
            self.sessions.save(session)
            self._schedule_background(
                self.consolidator.maybe_consolidate_by_tokens(session)
            )
            self._schedule_skill_extraction(result, content)
            if await self._deliver_multi_agent_stream(
                full_response, on_stream, on_stream_end
            ):
                metadata["_streamed"] = True

            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=full_response,
                metadata=metadata,
            )

        except Exception as e:
            logger.error("Multi-agent workflow failed: {}", e)
            error_response = (
                "The multi-agent workflow failed before it could produce an answer. "
                "Please try again."
            )
            if not session.messages or session.messages[-1].get("role") != "user":
                session.add_message("user", content)
            session.add_message("assistant", error_response)
            self._clear_pending_user_turn(session)
            self.sessions.save(session)
            metadata = {"multi_agent": True, "error": True}
            if await self._deliver_multi_agent_stream(
                error_response, on_stream, on_stream_end
            ):
                metadata["_streamed"] = True
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=error_response,
                metadata=metadata,
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
        pdf_path = Path(local_pdf_path)
        md_path = pdf_path.with_suffix(".md")
        ingest_tool = self.tools.get("paper_ingest")
        if ingest_tool is None or not hasattr(ingest_tool, "ingest_local_pdf"):
            return {"status": "error", "error": "Paper ingest tool not available"}
        result = await ingest_tool.ingest_local_pdf(
            doc,
            pdf_path,
            markdown_path=md_path,
            summarize=True,
        )
        if (
            result.get("status") == "ok"
            and not self.tools_config.paper.retain_uploaded_pdf
        ):
            pdf_path.unlink(missing_ok=True)
        return result

    def _schedule_skill_extraction(self, result: dict[str, Any], user_query: str) -> None:
        """Stage reusable Skill candidates from a completed paper workflow."""
        if (
            not self.tools_config.paper.enable
            or not self.skill_config.auto_extract_from_papers
            or not user_query.strip()
        ):
            return
        try:
            from nanobot.agent.skill_extractor import SkillExtractor

            result_snapshot = {
                key: value
                for key, value in result.items()
                if key not in {
                    "shared_system_context",
                    "runtime_context",
                    "media_context",
                }
            }
            result_snapshot["user_query"] = user_query
            extractor = SkillExtractor(
                store=self.context.memory,
                provider=self.provider,
                model=self.model,
                workspace=self.workspace,
                candidate_manager=self.skill_candidates,
                min_evidence_items=self.skill_config.min_evidence_items,
            )
            self._schedule_background(extractor.extract(result_snapshot))
        except Exception:
            logger.exception("Failed to schedule skill extraction")

    @staticmethod
    def _has_explicit_paper_signal(content: str) -> bool:
        content_lower = content.lower()
        paper_keywords = (
            "paper", "论文", "文献", "文章", "学术", "research", "arxiv",
            "publication", "survey", "review", "citation", "引用",
            "transformer", "mamba", "llm", "gpt", "bert",
            "neural network", "神经网络", "deep learning", "深度学习",
            "machine learning", "机器学习",
        )
        return any(keyword in content_lower for keyword in paper_keywords)

    @staticmethod
    def _has_paper_followup_signal(content: str) -> bool:
        content_lower = content.lower().strip()
        followup_indicators = (
            "该论文", "这篇", "本文", "上述", "前面", "刚才", "第一篇",
            "第二篇", "第三篇", "该模型", "这个模型", "该方法", "这个方法",
            "它的", "其", "继续", "展开", "详细", "性能", "指标", "准确率",
            "实验", "消融", "数据集", "baseline", "基线", "局限", "创新点",
            "算法", "优化", "优化器", "目标函数", "损失函数", "训练策略",
            "收敛", "公式", "架构", "模块", "怎么设计", "如何设计",
            "怎么实现", "具体原理",
            "还有", "其他", "其它", "更多", "另外", "再推荐", "再找",
            "compare it", "this paper", "the paper", "this model", "the model",
            "its performance", "previous one", "above", "former", "latter",
            "method", "algorithm", "optimizer", "objective", "loss function",
            "architecture", "training strategy", "how is it designed",
            "more papers", "other papers", "another paper",
        )
        return any(indicator in content_lower for indicator in followup_indicators)

    @staticmethod
    def _has_paper_novelty_signal(content: str) -> bool:
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in (
            "还有", "其他", "其它", "更多", "另外", "再推荐", "再找",
            "more papers", "other papers", "another paper", "additional papers",
        ))

    @staticmethod
    def _is_simple_acknowledgement(content: str) -> bool:
        normalized = content.strip().lower().rstrip("。.!！?？~～")
        return normalized in {
            "谢谢", "感谢", "好的", "好", "明白了", "知道了", "收到",
            "ok", "okay", "thanks", "thank you", "got it",
        }

    def _has_recent_multi_agent_context(self, session: Session) -> bool:
        marker = session.metadata.get(self._MULTI_AGENT_CONTEXT_MESSAGE_COUNT_KEY)
        try:
            marker_count = int(marker)
        except (TypeError, ValueError):
            return False
        # A turn normally contributes two messages. Keep paper context for the
        # configured short-history window, with a small allowance for tool-free
        # acknowledgements between two paper questions.
        max_delta = self.tools_config.paper.router_short_history_turns * 2 + 2
        return len(session.messages) - marker_count <= max_delta

    def should_use_multi_agent(self, content: str, session: Session | None = None) -> bool:
        """Fast deterministic gate used before the optional LLM orchestrator."""
        if not self._multi_agent_graph:
            return False
        if session is not None and session.metadata.get("multi_agent_paused_state"):
            return True
        if not self.tools_config.paper.multi_agent_orchestrator_enabled:
            return False
        if self._has_explicit_paper_signal(content):
            return True
        return bool(
            session is not None
            and self._has_recent_multi_agent_context(session)
            and self._has_paper_followup_signal(content)
        )

    async def _decide_multi_agent_with_orchestrator(
        self,
        content: str,
        session: Session,
    ) -> tuple[bool, dict[str, Any]]:
        """Choose single vs. paper multi-agent using rules plus recent context.

        High-confidence paper queries and referential follow-ups avoid an extra
        model call. Only ambiguous messages inside a recent paper conversation
        reach the lightweight LLM classifier.
        """
        if not self._multi_agent_graph:
            return False, {
                "orchestrator_decision": "single_agent",
                "orchestrator_reasoning": "multi-agent graph unavailable",
                "orchestrator_confidence": 1.0,
            }
        if session.metadata.get("multi_agent_paused_state"):
            return True, {
                "orchestrator_decision": "multi_agent",
                "orchestrator_reasoning": "resume paused paper workflow",
                "orchestrator_confidence": 1.0,
            }
        if not self.tools_config.paper.multi_agent_orchestrator_enabled:
            return False, {
                "orchestrator_decision": "single_agent",
                "orchestrator_reasoning": "paper orchestrator disabled",
                "orchestrator_confidence": 1.0,
            }
        if self._has_explicit_paper_signal(content):
            return True, {
                "orchestrator_decision": "multi_agent",
                "orchestrator_reasoning": "explicit academic-paper signal",
                "orchestrator_confidence": 1.0,
            }

        if self._is_simple_acknowledgement(content):
            return False, {
                "orchestrator_decision": "single_agent",
                "orchestrator_reasoning": "simple acknowledgement",
                "orchestrator_confidence": 1.0,
            }

        recent_paper_context = self._has_recent_multi_agent_context(session)
        if recent_paper_context and self._has_paper_followup_signal(content):
            return True, {
                "orchestrator_decision": "multi_agent",
                "orchestrator_reasoning": "referential follow-up to recent paper workflow",
                "orchestrator_confidence": 0.98,
            }
        if not recent_paper_context:
            return False, {
                "orchestrator_decision": "single_agent",
                "orchestrator_reasoning": "no paper signal or recent paper context",
                "orchestrator_confidence": 0.98,
            }

        recent_dialog = self._format_recent_dialog(
            session.get_history(max_messages=0),
            max_turns=self.tools_config.paper.router_short_history_turns,
        )
        recent_dialog = self._truncate_context(
            recent_dialog,
            self.tools_config.paper.router_short_memory_chars,
        )
        active_papers = session.metadata.get(self._MULTI_AGENT_ACTIVE_PAPERS_KEY, [])
        prompt = (
            "Classify whether the current message continues an academic-paper "
            "search/analysis conversation and therefore needs the paper multi-agent workflow. "
            "Use single_agent for acknowledgements, general chat, coding, or a clear topic change. "
            "Return JSON only: "
            '{"decision":"multi_agent|single_agent","confidence":0.0,"reasoning":"short"}.\n\n'
            f"Recent dialog:\n{recent_dialog or '(empty)'}\n\n"
            f"Active papers:\n{json.dumps(active_papers, ensure_ascii=False)}\n\n"
            f"Current message:\n{content}"
        )
        route_schema = {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["multi_agent", "single_agent"],
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string"},
            },
            "required": ["decision", "confidence", "reasoning"],
            "additionalProperties": False,
        }
        try:
            response = await self.provider.chat_structured_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a conservative conversation route classifier.",
                    },
                    {"role": "user", "content": prompt},
                ],
                json_schema=route_schema,
                temperature=0.0,
                max_tokens=256,
                disable_thinking=True,
                retry_mode=self.provider_retry_mode,
            )
            if response.finish_reason == "error":
                raise RuntimeError(response.content or "route classifier failed")
            raw = (response.content or "").strip()
            if "```" in raw:
                raw = raw.split("```", 2)[1]
                if raw.lstrip().startswith("json"):
                    raw = raw.lstrip()[4:]
            start = raw.find("{")
            end = raw.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("route classifier returned no JSON object")
            payload = json.loads(raw[start : end + 1])
            decision = str(payload.get("decision", "")).strip().lower()
            if decision not in {"multi_agent", "single_agent"}:
                raise ValueError(f"invalid route decision: {decision!r}")
            confidence = float(payload.get("confidence", 0.0) or 0.0)
            threshold = self.tools_config.paper.multi_agent_orchestrator_confidence_threshold
            # Paper conversations are sticky: only a high-confidence explicit
            # topic switch is allowed to leave the paper workflow. A low-
            # confidence/ambiguous decision keeps the previous multi-agent path.
            use_multi = not (
                decision == "single_agent" and confidence >= threshold
            )
            return use_multi, {
                "orchestrator_decision": decision,
                "orchestrator_reasoning": str(payload.get("reasoning", ""))[:300],
                "orchestrator_confidence": confidence,
            }
        except Exception as exc:
            logger.warning("Paper route orchestrator failed; keeping recent paper route: {}", exc)
            return True, {
                "orchestrator_decision": "multi_agent",
                "orchestrator_reasoning": "orchestrator unavailable; preserve recent paper context",
                "orchestrator_confidence": 0.0,
            }

    def _multi_agent_persisted_answer(
        self,
        *,
        final_answer: str,
        citations: list[Any],
        full_response: str,
    ) -> str:
        mode = self.tools_config.paper.multi_agent_memory_mode
        if mode == "debug_trace":
            return full_response
        if mode != "strict_with_citations" or not citations:
            return final_answer
        references = "\n".join(
            f"{index}. {citation}" for index, citation in enumerate(citations[:10], 1)
        )
        return f"{final_answer}\n\nReferences:\n{references}"

    def _remember_multi_agent_context(
        self,
        session: Session,
        result: dict[str, Any],
        sources: list[str],
    ) -> None:
        session.metadata[self._MULTI_AGENT_LAST_ROUTING_KEY] = str(
            result.get("routing_decision") or "unknown"
        )
        session.metadata[self._MULTI_AGENT_LAST_SOURCES_KEY] = list(dict.fromkeys(sources))

        papers: list[dict[str, str]] = []
        for key in (
            "papers_for_selection",
            "retrieval_results",
            "external_papers",
            "referenced_papers",
        ):
            values = result.get(key, [])
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict):
                    continue
                paper_id = str(value.get("paper_id") or value.get("id") or "").strip()
                title = str(
                    value.get("paper_title") or value.get("title") or ""
                ).strip()
                if paper_id or title:
                    papers.append({"paper_id": paper_id, "title": title})
        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for paper in papers:
            identity = (paper["paper_id"].lower(), paper["title"].lower())
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(paper)
            if len(deduped) >= 8:
                break
        if deduped:
            session.metadata[self._MULTI_AGENT_ACTIVE_PAPERS_KEY] = deduped

        visible_ids: list[str] = []
        if result.get("research_phase") == "select":
            for paper in result.get("papers_for_selection", []):
                if isinstance(paper, dict) and paper.get("paper_id"):
                    visible_ids.append(str(paper["paper_id"]))
        for citation in result.get("citations", []):
            match = re.match(r"\[([^]]+)\]", str(citation or "").strip())
            if match:
                visible_ids.append(match.group(1))
        visible_ids.extend(re.findall(
            r"\[([A-Za-z0-9][A-Za-z0-9._:/-]{1,127})\](?!\()",
            str(result.get("final_answer") or result.get("draft_answer") or ""),
        ))
        answer_text = str(
            result.get("final_answer") or result.get("draft_answer") or ""
        ).casefold()
        for paper in papers:
            paper_id = paper.get("paper_id", "")
            title = paper.get("title", "").strip()
            if paper_id and len(title) >= 8 and title.casefold() in answer_text:
                visible_ids.append(paper_id)

        previous_ids = session.metadata.get(
            self._MULTI_AGENT_PRESENTED_PAPER_IDS_KEY, []
        )
        cumulative_ids: list[str] = []
        seen_ids: set[str] = set()
        for raw_id in [*previous_ids, *visible_ids]:
            paper_id = re.sub(
                r"^arxiv:\s*", "", str(raw_id or "").strip(), flags=re.IGNORECASE
            )
            versioned = re.fullmatch(
                r"(?P<base>(?:\d{4}\.\d{4,5}|[a-z.-]+/\d{7}))v\d+",
                paper_id,
                flags=re.IGNORECASE,
            )
            identity = (versioned.group("base") if versioned else paper_id).casefold()
            if identity and identity not in seen_ids:
                seen_ids.add(identity)
                cumulative_ids.append(paper_id)
        if cumulative_ids:
            session.metadata[self._MULTI_AGENT_PRESENTED_PAPER_IDS_KEY] = cumulative_ids[-200:]

        resolved_topic = str(result.get("resolved_topic") or "").strip()
        if result.get("discovery_request") and resolved_topic:
            session.metadata[self._MULTI_AGENT_LAST_SEARCH_TOPIC_KEY] = resolved_topic
        session.metadata[self._MULTI_AGENT_CONTEXT_MESSAGE_COUNT_KEY] = len(session.messages)

    @staticmethod
    def _paper_references_from_tool_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Extract a bounded set of paper identities from Paper tool results."""
        references: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def visit(value: Any) -> None:
            if len(references) >= 8:
                return
            if isinstance(value, list):
                for item in value:
                    visit(item)
                    if len(references) >= 8:
                        break
                return
            if not isinstance(value, dict):
                return

            paper_id = str(
                value.get("paper_id") or value.get("arxiv_id") or ""
            ).strip()
            title = str(
                value.get("paper_title") or value.get("title") or ""
            ).strip()
            identity = (paper_id.casefold(), title.casefold())
            if (paper_id or title) and identity not in seen:
                seen.add(identity)
                references.append({"paper_id": paper_id, "title": title})
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    visit(nested)
                    if len(references) >= 8:
                        break

        for message in reversed(messages):
            if message.get("role") != "tool":
                continue
            if str(message.get("name") or "") not in _PAPER_TOOL_NAMES:
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                visit(json.loads(content))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if len(references) >= 8:
                break
        return references

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
