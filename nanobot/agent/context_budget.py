"""Shared token-budget primitives for context construction and pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from nanobot.utils.helpers import estimate_message_tokens, estimate_prompt_tokens


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """One request's prompt budget after output and estimation headroom."""

    context_window_tokens: int
    output_reserve_tokens: int
    safety_buffer_tokens: int = 1024
    context_block_limit: int | None = None

    @property
    def prompt_tokens(self) -> int:
        calculated = max(
            128,
            self.context_window_tokens
            - self.output_reserve_tokens
            - self.safety_buffer_tokens,
        )
        if self.context_block_limit is None:
            return calculated
        return max(128, min(calculated, self.context_block_limit))

    def bounded_share(self, configured_tokens: int, ratio: float) -> int:
        """Cap one optional section by both configuration and prompt share."""
        ratio_cap = max(128, int(self.prompt_tokens * ratio))
        return max(0, min(configured_tokens, ratio_cap))


class ContextBudgetManager:
    """Apply the same budget arithmetic to text sections and message history."""

    def __init__(self, budget: ContextBudget):
        self.budget = budget

    @staticmethod
    def text_tokens(text: str) -> int:
        if not text:
            return 0
        return estimate_prompt_tokens([{"role": "system", "content": text}])

    @staticmethod
    def truncate_text(text: str, max_tokens: int, *, keep_tail: bool = False) -> str:
        """Token-aware truncation with a dependency-free character fallback."""
        if max_tokens <= 0 or not text:
            return ""
        if ContextBudgetManager.text_tokens(text) <= max_tokens:
            return text
        marker = "[... earlier content omitted ...]" if keep_tail else "[... content omitted ...]"
        body_budget = max(
            1,
            max_tokens - ContextBudgetManager.text_tokens(marker) - 2,
        )
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            selected = tokens[-body_budget:] if keep_tail else tokens[:body_budget]
            body = encoding.decode(selected).strip()
        except Exception:
            char_limit = body_budget * 4
            body = (
                text[-char_limit:].lstrip()
                if keep_tail
                else text[:char_limit].rstrip()
            )
        return f"{marker}\n{body}" if keep_tail else f"{body}\n{marker}"

    @classmethod
    def take_recent_texts(
        cls,
        texts: Iterable[str],
        max_tokens: int,
    ) -> list[str]:
        """Keep newest entries that fit, truncating one oversized newest entry."""
        if max_tokens <= 0:
            return []
        kept: list[str] = []
        used = 0
        for text in reversed(list(texts)):
            cost = cls.text_tokens(text)
            if cost + used <= max_tokens:
                kept.append(text)
                used += cost
                continue
            if not kept:
                kept.append(cls.truncate_text(text, max_tokens, keep_tail=True))
            break
        kept.reverse()
        return kept

    def trim_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Keep system messages plus the newest legal user-led suffix."""
        if estimate_prompt_tokens(messages, tools) <= self.budget.prompt_tokens:
            return messages
        system = [dict(message) for message in messages if message.get("role") == "system"]
        other = [dict(message) for message in messages if message.get("role") != "system"]
        system_cost = sum(estimate_message_tokens(message) for message in system)
        tool_cost = estimate_prompt_tokens([], tools)
        remaining = max(128, self.budget.prompt_tokens - system_cost - tool_cost)
        kept: list[dict[str, Any]] = []
        used = 0
        for message in reversed(other):
            cost = estimate_message_tokens(message)
            if not kept and cost > remaining and isinstance(message.get("content"), str):
                truncated = dict(message)
                truncated["content"] = self.truncate_text(
                    str(message["content"]),
                    max(64, remaining - 8),
                    keep_tail=True,
                )
                kept.append(truncated)
                break
            if kept and used + cost > remaining:
                break
            kept.append(message)
            used += cost
        kept.reverse()
        for index, message in enumerate(kept):
            if message.get("role") == "user":
                kept = kept[index:]
                break
        return system + kept
