"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.context_budget import ContextBudget, ContextBudgetManager
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.utils.helpers import (
    build_assistant_message,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    find_legal_message_start,
    maybe_persist_tool_result,
    strip_think,
    truncate_text,
)
from nanobot.utils.prompt_templates import render_template
from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
    build_finalization_retry_message,
    build_length_recovery_message,
    build_post_tool_continuation_message,
    ensure_nonempty_tool_result,
    external_search_confirmation_prompt,
    is_blank_text,
    is_managed_paper_source_path,
    latest_user_message_text,
    paper_search_is_authorized,
    repeated_external_lookup_error,
)

_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
_PERSISTED_MODEL_ERROR_PLACEHOLDER = "[Assistant reply unavailable due to model error.]"
_MAX_EMPTY_RETRIES = 2
_MAX_LENGTH_RECOVERIES = 3
_TRANSIENT_MESSAGE_KEY = "_nanobot_transient"
_MAX_INJECTIONS_PER_TURN = 3
_MAX_INJECTION_CYCLES = 5
_SNIP_SAFETY_BUFFER = 1024
_MICROCOMPACT_KEEP_RECENT = 10
_MICROCOMPACT_MIN_CHARS = 500
_COMPACTABLE_TOOLS = frozenset({
    "read_file", "exec", "grep", "glob",
    "web_search", "web_fetch", "list_dir",
})
_BACKFILL_CONTENT = "[Tool result unavailable — call was interrupted or lost]"
_DEFAULT_SILENT_RESPONSE_TIMEOUT_S = 60.0
_TOOL_RECOVERY_MAX_TOKENS = 1024
_TOOL_PROTOCOL_TAG_RE = re.compile(
    r"</?(?:think|thought|tool_call)\b|<function\s*=",
    re.IGNORECASE,
)
_PAPER_ID_IN_TEXT_RE = re.compile(
    r'''["']?paper_id["']?\s*:\s*["']([^"'\\\s,}\]]+)''',
    re.IGNORECASE,
)



@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    max_tool_result_chars: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    disable_thinking: bool = False
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    workspace: Path | None = None
    session_key: str | None = None
    context_window_tokens: int | None = None
    context_block_limit: int | None = None
    provider_retry_mode: str = "standard"
    silent_response_timeout_s: float | None = _DEFAULT_SILENT_RESPONSE_TIMEOUT_S
    progress_callback: Any | None = None
    retry_wait_callback: Any | None = None
    checkpoint_callback: Any | None = None
    injection_callback: Any | None = None
    skill_activation_callback: Any | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    had_injections: bool = False


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [
                    item if isinstance(item, dict) else {"type": "text", "text": str(item)}
                    for item in value
                ]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    @classmethod
    def _append_injected_messages(
        cls,
        messages: list[dict[str, Any]],
        injections: list[dict[str, Any]],
    ) -> None:
        """Append injected user messages while preserving role alternation."""
        for injection in injections:
            if (
                messages
                and injection.get("role") == "user"
                and messages[-1].get("role") == "user"
            ):
                merged = dict(messages[-1])
                merged["content"] = cls._merge_message_content(
                    merged.get("content"),
                    injection.get("content"),
                )
                messages[-1] = merged
                continue
            messages.append(injection)

    async def _try_drain_injections(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any] | None,
        injection_cycles: int,
        *,
        phase: str = "after error",
        iteration: int | None = None,
    ) -> tuple[bool, int]:
        """Drain pending injections. Returns (should_continue, updated_cycles).

        If injections are found and we haven't exceeded _MAX_INJECTION_CYCLES,
        append them to *messages* (and emit a checkpoint if *assistant_message*
        and *iteration* are both provided) and return (True, cycles+1) so the
        caller continues the iteration loop.  Otherwise return (False, cycles).
        """
        if injection_cycles >= _MAX_INJECTION_CYCLES:
            return False, injection_cycles
        injections = await self._drain_injections(spec)
        if not injections:
            return False, injection_cycles
        injection_cycles += 1
        if assistant_message is not None:
            messages.append(assistant_message)
            if iteration is not None:
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "final_response",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [],
                    },
                )
        self._append_injected_messages(messages, injections)
        logger.info(
            "Injected {} follow-up message(s) {} ({}/{})",
            len(injections), phase, injection_cycles, _MAX_INJECTION_CYCLES,
        )
        return True, injection_cycles

    async def _drain_injections(self, spec: AgentRunSpec) -> list[dict[str, Any]]:
        """Drain pending user messages via the injection callback.

        Returns normalized user messages (capped by
        ``_MAX_INJECTIONS_PER_TURN``), or an empty list when there is
        nothing to inject. Messages beyond the cap are logged so they
        are not silently lost.
        """
        if spec.injection_callback is None:
            return []
        try:
            signature = inspect.signature(spec.injection_callback)
            accepts_limit = (
                "limit" in signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            if accepts_limit:
                items = await spec.injection_callback(limit=_MAX_INJECTIONS_PER_TURN)
            else:
                items = await spec.injection_callback()
        except Exception:
            logger.exception("injection_callback failed")
            return []
        if not items:
            return []
        injected_messages: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict) and item.get("role") == "user" and "content" in item:
                injected_messages.append(item)
                continue
            text = getattr(item, "content", str(item))
            if text.strip():
                injected_messages.append({"role": "user", "content": text})
        if len(injected_messages) > _MAX_INJECTIONS_PER_TURN:
            dropped = len(injected_messages) - _MAX_INJECTIONS_PER_TURN
            logger.warning(
                "Injection callback returned {} messages, capping to {} ({} dropped)",
                len(injected_messages), _MAX_INJECTIONS_PER_TURN, dropped,
            )
            injected_messages = injected_messages[:_MAX_INJECTIONS_PER_TURN]
        return injected_messages

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        external_lookup_counts: dict[str, int] = {}
        completed_tool_names: set[str] = set()
        empty_content_retries = 0
        length_recovery_count = 0
        length_recovery_parts: list[str] = []
        had_injections = False
        injection_cycles = 0
        post_tool_continuation_message: dict[str, str] | None = None

        for iteration in range(spec.max_iterations):
            try:
                # Keep the persisted conversation untouched. Context governance
                # may repair or compact historical messages for the model, but
                # those synthetic edits must not shift the append boundary used
                # later when the caller saves only the new turn.
                messages_for_model = self._drop_orphan_tool_results(messages)
                messages_for_model = self._backfill_missing_tool_results(messages_for_model)
                messages_for_model = self._microcompact(messages_for_model)
                messages_for_model = self._apply_tool_result_budget(spec, messages_for_model)
                messages_for_model = self._snip_history(spec, messages_for_model)
                # Snipping may have created new orphans; clean them up.
                messages_for_model = self._drop_orphan_tool_results(messages_for_model)
                messages_for_model = self._backfill_missing_tool_results(messages_for_model)
                if post_tool_continuation_message is not None:
                    # This is a request-only recovery hint. Do not append it to
                    # ``messages`` or it would be persisted as if the user had
                    # sent it.
                    messages_for_model = [
                        *messages_for_model,
                        post_tool_continuation_message,
                    ]
                    post_tool_continuation_message = None
            except Exception as exc:
                logger.warning(
                    "Context governance failed on turn {} for {}: {}; applying minimal repair",
                    iteration,
                    spec.session_key or "default",
                    exc,
                )
                try:
                    messages_for_model = self._drop_orphan_tool_results(messages)
                    messages_for_model = self._backfill_missing_tool_results(messages_for_model)
                except Exception:
                    messages_for_model = messages
            context = AgentHookContext(iteration=iteration, messages=messages)
            await hook.before_iteration(context)
            response = await self._request_model(spec, messages_for_model, hook, context)
            raw_usage = self._usage_dict(response.usage)
            context.response = response
            context.usage = dict(raw_usage)
            context.tool_calls = list(response.tool_calls)
            self._accumulate_usage(usage, raw_usage)
            stream_closed_for_repair = False

            logger.debug(
                "Model response on turn {} for {}: finish_reason={} "
                "prompt_tokens={} completion_tokens={} reasoning_chars={} tool_calls={}",
                iteration,
                spec.session_key or "default",
                response.finish_reason,
                raw_usage.get("prompt_tokens", 0),
                raw_usage.get("completion_tokens", 0),
                len(response.reasoning_content or ""),
                len(response.tool_calls),
            )

            if response.has_tool_calls and response.finish_reason in {
                "stop", "tool_calls", "length",
            }:
                preflight_errors = self._preflight_tool_calls(spec, response.tool_calls)
                if response.finish_reason == "length":
                    preflight_errors.append(
                        "tool-call generation ended because the output limit was reached"
                    )
                if preflight_errors:
                    failed_call = response.tool_calls[0]
                    logger.warning(
                        "Rejecting invalid tool call on turn {} for {} "
                        "(tool={}, finish_reason={}, reasoning_chars={}, "
                        "completion_tokens={}): {}",
                        iteration,
                        spec.session_key or "default",
                        failed_call.name or "(missing)",
                        response.finish_reason,
                        len(response.reasoning_content or ""),
                        raw_usage.get("completion_tokens", 0),
                        "; ".join(preflight_errors),
                    )
                    # Close any hidden/partial stream before issuing the direct,
                    # non-streaming repair request. The malformed assistant/tool
                    # pair is deliberately not appended to canonical history.
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=True)
                        stream_closed_for_repair = True
                    repaired = await self._request_tool_call_repair(
                        spec,
                        messages_for_model,
                        failed_call,
                        preflight_errors,
                    )
                    repair_usage = self._usage_dict(repaired.usage)
                    self._accumulate_usage(usage, repair_usage)
                    raw_usage = self._merge_usage(raw_usage, repair_usage)
                    response = repaired
                    context.response = response
                    context.usage = dict(raw_usage)
                    context.tool_calls = list(response.tool_calls)
                    logger.debug(
                        "Tool-call repair on turn {} for {}: finish_reason={} "
                        "completion_tokens={} reasoning_chars={} tool_calls={}",
                        iteration,
                        spec.session_key or "default",
                        response.finish_reason,
                        repair_usage.get("completion_tokens", 0),
                        len(response.reasoning_content or ""),
                        len(response.tool_calls),
                    )

                    repair_errors = self._preflight_tool_calls(
                        spec, response.tool_calls
                    ) if response.has_tool_calls else []
                    if response.finish_reason == "length" and response.has_tool_calls:
                        repair_errors.append(
                            "repaired tool call also reached the output limit"
                        )
                    if repair_errors:
                        logger.error(
                            "Tool-call repair remained invalid for {}: {}",
                            spec.session_key or "default",
                            "; ".join(repair_errors),
                        )
                        response = LLMResponse(
                            content=(
                                "The model could not produce valid tool arguments. "
                                "Please retry the request."
                            ),
                            finish_reason="error",
                            usage=repair_usage,
                            error_kind="invalid_tool_arguments",
                        )
                        context.response = response
                        context.tool_calls = []
                    elif (
                        hook.wants_streaming()
                        and not response.has_tool_calls
                        and not is_blank_text(response.content)
                    ):
                        # Repair calls are deliberately non-streaming. Forward a
                        # direct answer so web clients do not suppress the final
                        # outbound message as if it had already been streamed.
                        await hook.on_stream(context, response.content or "")

            if response.should_execute_tools:
                if hook.wants_streaming() and not stream_closed_for_repair:
                    await hook.on_stream_end(context, resuming=True)

                assistant_message = build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                messages.append(assistant_message)
                tools_used.extend(tc.name for tc in response.tool_calls)
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "awaiting_tools",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [tc.to_openai_tool_call() for tc in response.tool_calls],
                    },
                )

                await hook.before_execute_tools(context)

                results, new_events, fatal_error = await self._execute_tools(
                    spec,
                    response.tool_calls,
                    external_lookup_counts,
                    messages,
                    completed_tool_names,
                )
                tool_events.extend(new_events)
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                completed_tool_results: list[dict[str, Any]] = []
                for tool_call, result in zip(response.tool_calls, results):
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": self._normalize_tool_result(
                            spec,
                            tool_call.id,
                            tool_call.name,
                            result,
                        ),
                    }
                    messages.append(tool_message)
                    completed_tool_results.append(tool_message)
                if fatal_error is not None:
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    final_content = error
                    stop_reason = "tool_error"
                    self._append_final_message(messages, final_content)
                    context.final_content = final_content
                    context.error = error
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    should_continue, injection_cycles = await self._try_drain_injections(
                        spec, messages, None, injection_cycles,
                        phase="after tool error",
                    )
                    if should_continue:
                        had_injections = True
                        continue
                    break
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "tools_completed",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": completed_tool_results,
                        "pending_tool_calls": [],
                    },
                )
                empty_content_retries = 0
                length_recovery_count = 0
                # Checkpoint 1: drain injections after tools, before next LLM call
                _drained, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after tool execution",
                )
                if _drained:
                    had_injections = True
                await hook.after_iteration(context)
                continue

            if response.has_tool_calls:
                logger.warning(
                    "Ignoring tool calls under finish_reason='{}' for {}",
                    response.finish_reason,
                    spec.session_key or "default",
                )

            clean = hook.finalize_content(context, response.content)
            if response.finish_reason != "error" and is_blank_text(clean):
                empty_content_retries += 1
                reasoning_chars = len(response.reasoning_content or "")
                trailing_tool_names = self._trailing_tool_names(messages)
                if trailing_tool_names and empty_content_retries < _MAX_EMPTY_RETRIES:
                    logger.warning(
                        "Empty response after tool result on turn {} for {} "
                        "(finish_reason={}, reasoning_chars={}, tool_calls={}); "
                        "retrying with tools enabled (tools={})",
                        iteration,
                        spec.session_key or "default",
                        response.finish_reason,
                        reasoning_chars,
                        len(response.tool_calls),
                        ",".join(trailing_tool_names),
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=True)
                    post_tool_continuation_message = build_post_tool_continuation_message(
                        trailing_tool_names
                    )
                    await hook.after_iteration(context)
                    continue
                elif empty_content_retries < _MAX_EMPTY_RETRIES:
                    logger.warning(
                        "Empty response on turn {} for {} ({}/{}; finish_reason={}, "
                        "reasoning_chars={}, tool_calls={}); retrying",
                        iteration,
                        spec.session_key or "default",
                        empty_content_retries,
                        _MAX_EMPTY_RETRIES,
                        response.finish_reason,
                        reasoning_chars,
                        len(response.tool_calls),
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=True)
                    await hook.after_iteration(context)
                    continue
                else:
                    logger.warning(
                        "Empty response on turn {} for {} after {} retries "
                        "(finish_reason={}, reasoning_chars={}, tool_calls={}); "
                        "attempting finalization",
                        iteration,
                        spec.session_key or "default",
                        empty_content_retries,
                        response.finish_reason,
                        reasoning_chars,
                        len(response.tool_calls),
                    )
                    response = await self._request_finalization_retry(spec, messages_for_model)
                    retry_usage = self._usage_dict(response.usage)
                    self._accumulate_usage(usage, retry_usage)
                    raw_usage = self._merge_usage(raw_usage, retry_usage)
                    context.response = response
                    context.usage = dict(raw_usage)
                    context.tool_calls = list(response.tool_calls)
                    clean = hook.finalize_content(context, response.content)
                    if hook.wants_streaming() and not is_blank_text(clean):
                        await hook.on_stream(context, clean or "")

            if response.finish_reason == "length" and not is_blank_text(clean):
                length_recovery_count += 1
                length_recovery_parts.append(clean or "")
                if length_recovery_count <= _MAX_LENGTH_RECOVERIES:
                    logger.info(
                        "Output truncated on turn {} for {} ({}/{}); continuing",
                        iteration,
                        spec.session_key or "default",
                        length_recovery_count,
                        _MAX_LENGTH_RECOVERIES,
                    )
                    # A length recovery is still the same assistant reply. Keep
                    # the transport stream open so the next model call appends
                    # to the current UI bubble instead of creating a new one.
                    partial_message = build_assistant_message(
                        clean,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    )
                    partial_message[_TRANSIENT_MESSAGE_KEY] = "length_recovery"
                    messages.append(partial_message)
                    recovery_message = build_length_recovery_message()
                    recovery_message[_TRANSIENT_MESSAGE_KEY] = "length_recovery"
                    messages.append(recovery_message)
                    await hook.after_iteration(context)
                    continue

            # The provider returns each automatic continuation as a separate
            # completion, but to callers and persisted history they constitute
            # one assistant reply. Preserve byte-for-byte boundaries: the
            # model normally emits any required whitespace itself.
            if length_recovery_parts and response.finish_reason != "error":
                if response.finish_reason != "length" and not is_blank_text(clean):
                    length_recovery_parts.append(clean or "")
                clean = "".join(length_recovery_parts)
                length_recovery_parts.clear()

            assistant_message: dict[str, Any] | None = None
            if response.finish_reason != "error" and not is_blank_text(clean):
                assistant_message = build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

            # Check for mid-turn injections BEFORE signaling stream end.
            # If injections are found we keep the stream alive (resuming=True)
            # so streaming channels don't prematurely finalize the card.
            should_continue, injection_cycles = await self._try_drain_injections(
                spec, messages, assistant_message, injection_cycles,
                phase="after final response",
                iteration=iteration,
            )
            if should_continue:
                had_injections = True

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=should_continue)

            if should_continue:
                await hook.after_iteration(context)
                continue

            if response.finish_reason == "error":
                final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                self._append_model_error_placeholder(messages)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after LLM error",
                )
                if should_continue:
                    had_injections = True
                    continue
                break
            if is_blank_text(clean):
                final_content = EMPTY_FINAL_RESPONSE_MESSAGE
                stop_reason = "empty_final_response"
                error = final_content
                self._append_final_message(messages, final_content)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after empty response",
                )
                if should_continue:
                    had_injections = True
                    continue
                break

            messages.append(assistant_message or build_assistant_message(
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            ))
            await self._emit_checkpoint(
                spec,
                {
                    "phase": "final_response",
                    "iteration": iteration,
                    "model": spec.model,
                    "assistant_message": messages[-1],
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                },
            )
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            break
        else:
            stop_reason = "max_iterations"
            if spec.max_iterations_message:
                final_content = spec.max_iterations_message.format(
                    max_iterations=spec.max_iterations,
                )
            else:
                final_content = render_template(
                    "agent/max_iterations_message.md",
                    strip=True,
                    max_iterations=spec.max_iterations,
                )
            self._append_final_message(messages, final_content)
            # Drain any remaining injections so they are appended to the
            # conversation history instead of being re-published as
            # independent inbound messages by _dispatch's finally block.
            # We ignore should_continue here because the for-loop has already
            # exhausted all iterations.
            drained_after_max_iterations, injection_cycles = await self._try_drain_injections(
                spec, messages, None, injection_cycles,
                phase="after max_iterations",
            )
            if drained_after_max_iterations:
                had_injections = True

        return AgentRunResult(
            final_content=final_content,
            # Recovery prompts and partial assistant messages are request-only
            # context. Never expose them as canonical conversation history.
            messages=[m for m in messages if not m.get(_TRANSIENT_MESSAGE_KEY)],
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
            had_injections=had_injections,
        )

    def _build_request_kwargs(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "model": spec.model,
            "retry_mode": spec.provider_retry_mode,
            "on_retry_wait": spec.retry_wait_callback,
        }
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        if spec.max_tokens is not None:
            kwargs["max_tokens"] = spec.max_tokens
        if spec.reasoning_effort is not None:
            kwargs["reasoning_effort"] = spec.reasoning_effort
        return kwargs

    async def _request_model(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        hook: AgentHook,
        context: AgentHookContext,
    ):
        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=spec.tools.get_definitions(),
        )
        if hook.wants_streaming():
            buffered_content = ""
            visible_content_started = asyncio.Event()
            forwarding_content = False

            async def _stream(delta: str) -> None:
                nonlocal buffered_content, forwarding_content
                if forwarding_content:
                    await hook.on_stream(context, delta)
                    return
                buffered_content += delta
                if not strip_think(buffered_content).strip():
                    return
                forwarding_content = True
                visible_content_started.set()
                await hook.on_stream(context, buffered_content)

            request_task = asyncio.create_task(
                self.provider.chat_stream_with_retry(
                    **kwargs,
                    on_content_delta=_stream,
                )
            )
            timeout_s = spec.silent_response_timeout_s
            if timeout_s is None or timeout_s <= 0:
                return await request_task

            visible_task = asyncio.create_task(visible_content_started.wait())
            try:
                done, _pending = await asyncio.wait(
                    {request_task, visible_task},
                    timeout=timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                request_task.cancel()
                visible_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await request_task
                with contextlib.suppress(asyncio.CancelledError):
                    await visible_task
                raise
            if request_task in done:
                visible_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await visible_task
                return await request_task
            if visible_task in done:
                return await request_task

            request_task.cancel()
            visible_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await request_task
            with contextlib.suppress(asyncio.CancelledError):
                await visible_task
            logger.warning(
                "No user-visible model output for {:.1f}s on turn {} for {}; "
                "retrying directly with thinking disabled",
                timeout_s,
                context.iteration,
                spec.session_key or "default",
            )
            await self._emit_progress(
                spec,
                "The model produced no visible output for too long; retrying in fast mode.",
            )
            recovered = await self._request_silent_response_recovery(
                spec, messages
            )
            if not is_blank_text(recovered.content):
                await hook.on_stream(context, recovered.content or "")
            return recovered
        if spec.disable_thinking:
            kwargs["disable_thinking"] = True
        return await self.provider.chat_with_retry(**kwargs)

    @staticmethod
    def _tool_definition_name(definition: dict[str, Any]) -> str:
        function = definition.get("function")
        if isinstance(function, dict):
            return str(function.get("name") or "")
        return str(definition.get("name") or "")

    def _short_request_max_tokens(self, spec: AgentRunSpec) -> int:
        configured = spec.max_tokens
        if not isinstance(configured, int):
            configured = getattr(
                getattr(self.provider, "generation", None),
                "max_tokens",
                None,
            )
        if not isinstance(configured, int) or configured <= 0:
            configured = _TOOL_RECOVERY_MAX_TOKENS
        return min(configured, _TOOL_RECOVERY_MAX_TOKENS)

    async def _request_silent_response_recovery(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ):
        recovery_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "The previous generation produced no user-visible output before "
                    "the timeout. Respond now without hidden reasoning. If a tool is "
                    "needed, issue the native tool call immediately with valid JSON; "
                    "otherwise give a concise direct answer. Resolve references such "
                    "as first/second/that paper from the conversation and preserve the "
                    "exact paper ID in the tool arguments."
                ),
                _TRANSIENT_MESSAGE_KEY: "silent_response_recovery",
            },
        ]
        kwargs = self._build_request_kwargs(
            spec,
            recovery_messages,
            tools=spec.tools.get_definitions(),
        )
        kwargs.update({
            "max_tokens": self._short_request_max_tokens(spec),
            "temperature": 0.0,
            "reasoning_effort": None,
            "disable_thinking": True,
        })
        return await self.provider.chat_with_retry(**kwargs)

    @staticmethod
    async def _emit_progress(spec: AgentRunSpec, message: str) -> None:
        callback = spec.progress_callback
        if not callable(callback):
            return
        try:
            result = callback(message)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Agent progress callback failed")

    @classmethod
    def _protocol_contamination_path(
        cls,
        value: Any,
        path: str = "arguments",
    ) -> str | None:
        if isinstance(value, str):
            return path if _TOOL_PROTOCOL_TAG_RE.search(value) else None
        if isinstance(value, dict):
            for key, child in value.items():
                found = cls._protocol_contamination_path(
                    child,
                    f"{path}.{key}",
                )
                if found:
                    return found
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found = cls._protocol_contamination_path(
                    child,
                    f"{path}[{index}]",
                )
                if found:
                    return found
        return None

    @classmethod
    def _preflight_tool_calls(
        cls,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> list[str]:
        errors: list[str] = []
        prepare_call = getattr(spec.tools, "prepare_call", None)
        for index, tool_call in enumerate(tool_calls):
            label = tool_call.name or f"call[{index}]"
            contaminated_path = cls._protocol_contamination_path(
                tool_call.arguments
            )
            if contaminated_path:
                errors.append(
                    f"{label}: protocol control tag found in {contaminated_path}"
                )
            if not callable(prepare_call):
                continue
            try:
                prepared = prepare_call(tool_call.name, tool_call.arguments)
            except Exception as exc:
                errors.append(f"{label}: validation raised {type(exc).__name__}")
                continue
            if isinstance(prepared, tuple) and len(prepared) == 3 and prepared[2]:
                errors.append(str(prepared[2]))
        return errors

    @staticmethod
    def _safe_repair_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        def _clean(value: Any) -> Any:
            if isinstance(value, str):
                if _TOOL_PROTOCOL_TAG_RE.search(value):
                    return None
                return value[:1000]
            if isinstance(value, list):
                return [cleaned for item in value if (cleaned := _clean(item)) is not None]
            if isinstance(value, dict):
                return {
                    str(key): cleaned
                    for key, item in value.items()
                    if (cleaned := _clean(item)) is not None
                }
            if value is None or isinstance(value, (bool, int, float)):
                return value
            return None

        cleaned = _clean(arguments)
        return cleaned if isinstance(cleaned, dict) else {}

    @staticmethod
    def _recover_paper_ids(arguments: dict[str, Any]) -> list[str]:
        ids: list[str] = []

        def _walk(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                for child_key, child in value.items():
                    _walk(child, str(child_key))
                return
            if isinstance(value, list):
                for child in value:
                    _walk(child, key)
                return
            if not isinstance(value, str):
                return
            if key == "paper_id":
                candidates = [value]
            else:
                normalized = value.replace('\\"', '"').replace("\\'", "'")
                candidates = _PAPER_ID_IN_TEXT_RE.findall(normalized)
            for candidate in candidates:
                paper_id = candidate.strip()
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{1,127}", paper_id):
                    ids.append(paper_id)

        _walk(arguments)
        return list(dict.fromkeys(ids))[:20]

    async def _request_tool_call_repair(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        failed_call: ToolCallRequest,
        errors: list[str],
    ):
        definitions = spec.tools.get_definitions()
        target_definitions = [
            definition
            for definition in definitions
            if self._tool_definition_name(definition) == failed_call.name
        ]
        safe_arguments = self._safe_repair_arguments(failed_call.arguments)
        paper_ids = self._recover_paper_ids(failed_call.arguments)
        details = [
            f"The previous native call to `{failed_call.name}` was rejected before execution.",
            "Return exactly one native tool call with valid JSON matching its schema.",
            "Do not emit prose, hidden reasoning, XML tags, or textual tool-call markup.",
            f"Validation errors: {'; '.join(errors)[:1500]}",
        ]
        if safe_arguments:
            details.append(
                "Preserve these valid arguments when relevant: "
                + json.dumps(safe_arguments, ensure_ascii=False)[:2000]
            )
        if paper_ids:
            details.append(
                "Preserve these recovered target paper IDs: "
                + json.dumps(paper_ids, ensure_ascii=False)
            )
        repair_messages = [
            *messages,
            {
                "role": "user",
                "content": "\n".join(details),
                _TRANSIENT_MESSAGE_KEY: "tool_call_repair",
            },
        ]
        kwargs = self._build_request_kwargs(
            spec,
            repair_messages,
            tools=target_definitions or definitions,
        )
        kwargs.update({
            "max_tokens": self._short_request_max_tokens(spec),
            "temperature": 0.0,
            "reasoning_effort": None,
            "disable_thinking": True,
        })
        if target_definitions:
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": failed_call.name},
            }
        return await self.provider.chat_with_retry(**kwargs)

    async def _request_finalization_retry(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ):
        retry_messages = list(messages)
        retry_messages.append(build_finalization_retry_message())
        kwargs = self._build_request_kwargs(spec, retry_messages, tools=None)
        # Finalization is deliberately short and tool-free. For local Qwen
        # models, disabling thinking prevents another reasoning-only response
        # from consuming the visible-answer phase.
        kwargs["disable_thinking"] = True
        return await self.provider.chat_with_retry(**kwargs)

    @staticmethod
    def _trailing_tool_names(messages: list[dict[str, Any]]) -> list[str]:
        """Return tool names in the latest consecutive tool-result block."""
        names: list[str] = []
        for message in reversed(messages):
            if message.get("role") != "tool":
                break
            name = str(message.get("name") or "").strip()
            if name:
                names.append(name)
        names.reverse()
        return names

    @staticmethod
    def _usage_dict(usage: dict[str, Any] | None) -> dict[str, int]:
        if not usage:
            return {}
        result: dict[str, int] = {}
        for key, value in usage.items():
            try:
                result[key] = int(value or 0)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _accumulate_usage(target: dict[str, int], addition: dict[str, int]) -> None:
        for key, value in addition.items():
            target[key] = target.get(key, 0) + value

    @staticmethod
    def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        merged = dict(left)
        for key, value in right.items():
            merged[key] = merged.get(key, 0) + value
        return merged

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        external_lookup_counts: dict[str, int],
        messages: list[dict[str, Any]] | None = None,
        completed_tool_names: set[str] | None = None,
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None]:
        messages = messages or []
        if completed_tool_names is None:
            completed_tool_names = set()
        batches = self._partition_tool_batches(spec, tool_calls)
        tool_results: list[tuple[Any, dict[str, str], BaseException | None]] = []
        for batch in batches:
            if spec.concurrent_tools and len(batch) > 1:
                tool_results.extend(await asyncio.gather(*(
                    self._run_tool(
                        spec,
                        tool_call,
                        external_lookup_counts,
                        messages,
                        completed_tool_names,
                    )
                    for tool_call in batch
                )))
            else:
                for tool_call in batch:
                    tool_results.append(await self._run_tool(
                        spec,
                        tool_call,
                        external_lookup_counts,
                        messages,
                        completed_tool_names,
                    ))

            for tool_call, (result, event, _) in zip(
                batch,
                tool_results[-len(batch):],
            ):
                if event.get("status") != "ok":
                    continue
                if tool_call.name == "kb_retrieve" and isinstance(result, str):
                    try:
                        kb_payload = json.loads(result)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        kb_payload = None
                    if isinstance(kb_payload, dict) and kb_payload.get("error"):
                        continue
                completed_tool_names.add(tool_call.name)

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        for result, event, error in tool_results:
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        return results, events, fatal_error

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
        external_lookup_counts: dict[str, int],
        messages: list[dict[str, Any]] | None = None,
        completed_tool_names: set[str] | None = None,
    ) -> tuple[Any, dict[str, str], BaseException | None]:
        messages = messages or []
        completed_tool_names = completed_tool_names or set()
        _HINT = "\n\n[Analyze the error above and try a different approach.]"
        lookup_error = repeated_external_lookup_error(
            tool_call.name,
            tool_call.arguments,
            external_lookup_counts,
        )
        if lookup_error:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": "repeated external lookup blocked",
            }
            if spec.fail_on_tool_error:
                return lookup_error + _HINT, event, RuntimeError(lookup_error)
            return lookup_error + _HINT, event, None
        prepare_call = getattr(spec.tools, "prepare_call", None)
        tool, params, prep_error = None, tool_call.arguments, None
        if callable(prepare_call):
            try:
                prepared = prepare_call(tool_call.name, tool_call.arguments)
                if isinstance(prepared, tuple) and len(prepared) == 3:
                    tool, params, prep_error = prepared
            except Exception:
                pass
        if prep_error:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": prep_error.split(": ", 1)[-1][:120],
            }
            return prep_error + _HINT, event, RuntimeError(prep_error) if spec.fail_on_tool_error else None
        if (
            tool_call.name == "read_file"
            and isinstance(params, dict)
            and is_managed_paper_source_path(params.get("path"))
            and "kb_retrieve" not in completed_tool_names
        ):
            result = json.dumps(
                {
                    "status": "kb_retrieve_required",
                    "next_action": "call_kb_retrieve_first",
                    "message": (
                        "This is a managed paper source file. Call kb_retrieve with "
                        "the target paper entity and retrieval_mode='hybrid' first. "
                        "Use read_file only afterward if the retrieved chunks do not "
                        "contain the requested details."
                    ),
                },
                ensure_ascii=False,
            )
            return result, {
                "name": tool_call.name,
                "status": "blocked",
                "detail": "kb_retrieve required before reading a managed paper source",
            }, None
        if tool_call.name == "paper_search" and not paper_search_is_authorized(messages):
            result = json.dumps(
                {
                    "status": "confirmation_required",
                    "next_action": "ask_user_before_external_search",
                    "message": external_search_confirmation_prompt(
                        latest_user_message_text(messages)
                    ),
                },
                ensure_ascii=False,
            )
            return result, {
                "name": tool_call.name,
                "status": "blocked",
                "detail": "external paper search requires user confirmation",
            }, None
        try:
            if tool is not None:
                result = await tool.execute(**params)
            else:
                result = await spec.tools.execute(tool_call.name, params)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            return f"Error: {type(exc).__name__}: {exc}", event, None

        if isinstance(result, str) and result.startswith("Error"):
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": result.replace("\n", " ").strip()[:120],
            }
            if spec.fail_on_tool_error:
                return result + _HINT, event, RuntimeError(result)
            return result + _HINT, event, None

        if tool_call.name == "read_file" and spec.skill_activation_callback is not None:
            path = params.get("path") if isinstance(params, dict) else None
            if path:
                try:
                    callback_result = spec.skill_activation_callback(path)
                    if inspect.isawaitable(callback_result):
                        await callback_result
                except Exception:
                    # Telemetry must never make an otherwise successful tool
                    # call fail or change the model-visible result.
                    logger.exception("skill_activation_callback failed for {}", path)

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        return result, {"name": tool_call.name, "status": "ok", "detail": detail}, None

    async def _emit_checkpoint(
        self,
        spec: AgentRunSpec,
        payload: dict[str, Any],
    ) -> None:
        callback = spec.checkpoint_callback
        if callback is not None:
            await callback(payload)

    @staticmethod
    def _append_final_message(messages: list[dict[str, Any]], content: str | None) -> None:
        if not content:
            return
        if (
            messages
            and messages[-1].get("role") == "assistant"
            and not messages[-1].get("tool_calls")
        ):
            if messages[-1].get("content") == content:
                return
            messages[-1] = build_assistant_message(content)
            return
        messages.append(build_assistant_message(content))

    @staticmethod
    def _append_model_error_placeholder(messages: list[dict[str, Any]]) -> None:
        if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
            return
        messages.append(build_assistant_message(_PERSISTED_MODEL_ERROR_PLACEHOLDER))

    def _normalize_tool_result(
        self,
        spec: AgentRunSpec,
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> Any:
        result = ensure_nonempty_tool_result(tool_name, result)
        try:
            content = maybe_persist_tool_result(
                spec.workspace,
                spec.session_key,
                tool_call_id,
                result,
                max_chars=spec.max_tool_result_chars,
            )
        except Exception as exc:
            logger.warning(
                "Tool result persist failed for {} in {}: {}; using raw result",
                tool_call_id,
                spec.session_key or "default",
                exc,
            )
            content = result
        if isinstance(content, str) and len(content) > spec.max_tool_result_chars:
            return truncate_text(content, spec.max_tool_result_chars)
        return content

    @staticmethod
    def _drop_orphan_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop tool results that have no matching assistant tool_call earlier in the history."""
        declared: set[str] = set()
        updated: list[dict[str, Any]] | None = None
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            if role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    if updated is None:
                        updated = [dict(m) for m in messages[:idx]]
                    continue
            if updated is not None:
                updated.append(dict(msg))

        if updated is None:
            return messages
        return updated

    @staticmethod
    def _backfill_missing_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert synthetic error results for orphaned tool_use blocks."""
        declared: list[tuple[int, str, str]] = []  # (assistant_idx, call_id, name)
        fulfilled: set[str] = set()
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        name = ""
                        func = tc.get("function")
                        if isinstance(func, dict):
                            name = func.get("name", "")
                        declared.append((idx, str(tc["id"]), name))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    fulfilled.add(str(tid))

        missing = [(ai, cid, name) for ai, cid, name in declared if cid not in fulfilled]
        if not missing:
            return messages

        updated = list(messages)
        offset = 0
        for assistant_idx, call_id, name in missing:
            insert_at = assistant_idx + 1 + offset
            while insert_at < len(updated) and updated[insert_at].get("role") == "tool":
                insert_at += 1
            updated.insert(insert_at, {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": _BACKFILL_CONTENT,
            })
            offset += 1
        return updated

    @staticmethod
    def _microcompact(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace old compactable tool results with one-line summaries."""
        compactable_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") == "tool" and msg.get("name") in _COMPACTABLE_TOOLS:
                compactable_indices.append(idx)

        if len(compactable_indices) <= _MICROCOMPACT_KEEP_RECENT:
            return messages

        stale = compactable_indices[: len(compactable_indices) - _MICROCOMPACT_KEEP_RECENT]
        updated: list[dict[str, Any]] | None = None
        for idx in stale:
            msg = messages[idx]
            content = msg.get("content")
            if not isinstance(content, str) or len(content) < _MICROCOMPACT_MIN_CHARS:
                continue
            name = msg.get("name", "tool")
            summary = f"[{name} result omitted from context]"
            if updated is None:
                updated = [dict(m) for m in messages]
            updated[idx]["content"] = summary

        return updated if updated is not None else messages

    def _apply_tool_result_budget(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        updated = messages
        for idx, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            normalized = self._normalize_tool_result(
                spec,
                str(message.get("tool_call_id") or f"tool_{idx}"),
                str(message.get("name") or "tool"),
                message.get("content"),
            )
            if normalized != message.get("content"):
                if updated is messages:
                    updated = [dict(m) for m in messages]
                updated[idx]["content"] = normalized
        return updated

    def _snip_history(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not messages or not spec.context_window_tokens:
            return messages

        provider_max_tokens = getattr(getattr(self.provider, "generation", None), "max_tokens", 4096)
        max_output = spec.max_tokens if isinstance(spec.max_tokens, int) else (
            provider_max_tokens if isinstance(provider_max_tokens, int) else 4096
        )
        budget = ContextBudget(
            context_window_tokens=spec.context_window_tokens,
            output_reserve_tokens=max_output,
            safety_buffer_tokens=_SNIP_SAFETY_BUFFER,
            context_block_limit=spec.context_block_limit,
        ).prompt_tokens
        if budget <= 0:
            return messages

        estimate, _ = estimate_prompt_tokens_chain(
            self.provider,
            spec.model,
            messages,
            spec.tools.get_definitions(),
        )
        if estimate <= budget:
            return messages

        system_messages = [dict(msg) for msg in messages if msg.get("role") == "system"]
        non_system = [dict(msg) for msg in messages if msg.get("role") != "system"]
        if not non_system:
            return messages

        system_tokens = sum(estimate_message_tokens(msg) for msg in system_messages)
        tool_tokens = estimate_prompt_tokens_chain(
            self.provider,
            spec.model,
            [],
            spec.tools.get_definitions(),
        )[0]
        remaining_budget = max(128, budget - system_tokens - tool_tokens)
        kept: list[dict[str, Any]] = []
        kept_tokens = 0
        for message in reversed(non_system):
            msg_tokens = estimate_message_tokens(message)
            if (
                not kept
                and msg_tokens > remaining_budget
                and isinstance(message.get("content"), str)
            ):
                truncated = dict(message)
                truncated["content"] = ContextBudgetManager.truncate_text(
                    str(message["content"]),
                    max(64, remaining_budget - 8),
                    keep_tail=True,
                )
                kept.append(truncated)
                kept_tokens = estimate_message_tokens(truncated)
                break
            if kept and kept_tokens + msg_tokens > remaining_budget:
                break
            kept.append(message)
            kept_tokens += msg_tokens
        kept.reverse()

        if kept:
            for i, message in enumerate(kept):
                if message.get("role") == "user":
                    kept = kept[i:]
                    break
            else:
                # Recover nearest user message from outside the kept window;
                # GLM rejects system→assistant (error 1214).  Budget is
                # intentionally exceeded — oversized beats invalid.
                for idx in range(len(non_system) - 1, -1, -1):
                    if non_system[idx].get("role") == "user":
                        kept = non_system[idx:]
                        break
                # If no user exists at all, _enforce_role_alternation
                # will insert a synthetic one as a safety net.
            start = find_legal_message_start(kept)
            if start:
                kept = kept[start:]
        if not kept:
            kept = non_system[-min(len(non_system), 4) :]
            start = find_legal_message_start(kept)
            if start:
                kept = kept[start:]
        return system_messages + kept

    def _partition_tool_batches(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> list[list[ToolCallRequest]]:
        if not spec.concurrent_tools:
            return [[tool_call] for tool_call in tool_calls]

        batches: list[list[ToolCallRequest]] = []
        current: list[ToolCallRequest] = []
        for tool_call in tool_calls:
            get_tool = getattr(spec.tools, "get", None)
            tool = get_tool(tool_call.name) if callable(get_tool) else None
            can_batch = bool(tool and tool.concurrency_safe)
            if can_batch:
                current.append(tool_call)
                continue
            if current:
                batches.append(current)
                current = []
            batches.append([tool_call])
        if current:
            batches.append(current)
        return batches
