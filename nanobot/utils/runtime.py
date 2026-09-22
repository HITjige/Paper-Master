"""Runtime-specific helper functions and constants."""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from nanobot.utils.helpers import stringify_text_blocks

_MAX_REPEAT_EXTERNAL_LOOKUPS = 2

EMPTY_FINAL_RESPONSE_MESSAGE = (
    "I completed the tool steps but couldn't produce a final answer. "
    "Please try again or narrow the task."
)

FINALIZATION_RETRY_PROMPT = (
    "Provide the final response to the user now using the conversation and any "
    "successful tool results above. Do not call another tool, emit tool-call markup, "
    "or describe internal steps. Give a direct, evidence-grounded answer."
)

POST_TOOL_CONTINUATION_PROMPT = (
    "The previous tool call completed but your response was empty. Continue the "
    "user's task using the tool result. You still have access to tools: call the "
    "next required tool, or provide the final answer if no more tool is needed."
)

LENGTH_RECOVERY_PROMPT = (
    "Output limit reached. Continue exactly where you left off "
    "— no recap, no apology. Break remaining work into smaller steps if needed."
)


def empty_tool_result_message(tool_name: str) -> str:
    """Short prompt-safe marker for tools that completed without visible output."""
    return f"({tool_name} completed with no output)"


def ensure_nonempty_tool_result(tool_name: str, content: Any) -> Any:
    """Replace semantically empty tool results with a short marker string."""
    if content is None:
        return empty_tool_result_message(tool_name)
    if isinstance(content, str) and not content.strip():
        return empty_tool_result_message(tool_name)
    if isinstance(content, list):
        if not content:
            return empty_tool_result_message(tool_name)
        text_payload = stringify_text_blocks(content)
        if text_payload is not None and not text_payload.strip():
            return empty_tool_result_message(tool_name)
    return content


def is_blank_text(content: str | None) -> bool:
    """True when *content* is missing or only whitespace."""
    return content is None or not content.strip()


def build_finalization_retry_message() -> dict[str, str]:
    """A short no-tools-allowed prompt for final answer recovery."""
    return {"role": "user", "content": FINALIZATION_RETRY_PROMPT}


def build_post_tool_continuation_message(
    tool_names: list[str] | tuple[str, ...],
) -> dict[str, str]:
    """Prompt a tool-capable retry after a reasoning-only/empty response."""
    names = ", ".join(dict.fromkeys(str(name) for name in tool_names if name))
    suffix = f" Latest completed tool(s): {names}." if names else ""
    if "paper_ingest" in tool_names:
        suffix += (
            " If paper_ingest succeeded, call kb_retrieve with the original user's "
            "question before writing an evidence-grounded answer."
        )
    if "kb_retrieve" in tool_names:
        suffix += (
            " Inspect the kb_retrieve quality and next_action fields. If quality is "
            "sufficient, answer from the KB evidence and do not call paper_search unless "
            "the user explicitly requested latest, recent, external, or arXiv results. "
            "If KB quality is insufficient/empty and external search was not explicitly "
            "requested or previously confirmed, ask the user for permission and wait."
        )
    return {"role": "user", "content": POST_TOOL_CONTINUATION_PROMPT + suffix}


def build_length_recovery_message() -> dict[str, str]:
    """Prompt the model to continue after hitting output token limit."""
    return {"role": "user", "content": LENGTH_RECOVERY_PROMPT}


def is_explicit_external_paper_search_request(text: str) -> bool:
    """Return whether the user explicitly requested fresh/external paper search."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    if not normalized:
        return False
    if re.search(
        r"(?:外部|联网|网上|网络搜索|知识库外|arxiv|最新|近期|最近(?:发表|发布|的)?(?:论文|文献|文章))"
        r"|\b(?:external|online|search online|web search|arxiv|latest|newest|recent)\b",
        normalized,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(r"(?:19|20)\d{2}", normalized)
        and re.search(
            r"(?:论文|文献|文章)|\b(?:papers?|literature|articles?)\b",
            normalized,
            re.IGNORECASE,
        )
    )


def parse_external_search_confirmation(text: str) -> bool | None:
    """Parse a short response to an external-paper-search confirmation."""
    normalized = re.sub(r"[\s。.!！?？,，;；]+", "", str(text or "")).casefold()
    if not normalized:
        return None
    affirmative = {
        "可以", "好", "好的", "同意", "允许", "继续", "去搜吧", "搜索吧",
        "可以搜索", "允许搜索", "同意搜索", "继续搜索", "yes", "y", "ok",
        "okay", "sure", "proceed", "goahead", "search", "searchit",
    }
    negative = {
        "不", "不用", "不要", "不需要", "不允许", "不同意", "取消", "算了",
        "只看本地", "只用知识库", "只使用知识库", "no", "n", "nope", "cancel", "stop",
        "localonly", "kbonly",
    }
    if normalized in affirmative:
        return True
    if normalized in negative:
        return False
    if re.search(r"(?:可以|允许|同意|继续|请).{0,6}(?:外部|联网|搜索|arxiv)", normalized):
        return True
    if re.search(r"(?:不要|不用|拒绝|不允许).{0,6}(?:外部|联网|搜索|arxiv)", normalized):
        return False
    return None


def external_search_confirmation_prompt(user_text: str = "") -> str:
    """Stable user-facing prompt shared by ordinary and multi-agent flows."""
    if user_text and not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", user_text):
        return (
            "The local knowledge base does not contain enough evidence for a complete "
            "answer. May I search external arXiv papers for additional sources? "
            "Reply “yes” or “local knowledge base only”."
        )
    return (
        "当前知识库中的证据不足以完整回答。是否允许我搜索外部 arXiv 论文来补充？"
        "请回复“可以”或“只使用知识库”。"
    )


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return stringify_text_blocks(content) or ""


def latest_user_message_text(messages: list[dict[str, Any]]) -> str:
    """Return the latest user-authored text from a model message chain."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message)
    return ""


def paper_search_is_authorized(messages: list[dict[str, Any]]) -> bool:
    """Authorize paper_search from the latest user request and prior confirmation."""
    latest_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )
    if latest_user_index is None:
        return False
    latest_user_text = latest_user_message_text(messages)
    if is_explicit_external_paper_search_request(latest_user_text):
        return True

    confirmation = parse_external_search_confirmation(latest_user_text)
    if confirmation is not True:
        return False

    for message in reversed(messages[max(0, latest_user_index - 8):latest_user_index]):
        text = _message_text(message)
        if message.get("role") == "tool" and message.get("name") == "paper_search":
            try:
                payload = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict) and payload.get("status") == "confirmation_required":
                return True
        if message.get("role") == "assistant" and re.search(
            r"(?:是否|要不要|允许|同意|can i|may i|would you like).{0,30}"
            r"(?:外部|联网|搜索|arxiv|external|online)",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            return True
    return False


def is_managed_paper_source_path(path: Any) -> bool:
    """Recognize raw paper files managed by the local KB."""
    normalized = str(path or "").strip().replace("\\", "/").casefold()
    return bool(re.search(r"(?:^|/)kb/(?:uploads|downloads)(?:/|$)", normalized))


def external_lookup_signature(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Stable signature for repeated external lookups we want to throttle."""
    if tool_name == "web_fetch":
        url = str(arguments.get("url") or "").strip()
        if url:
            return f"web_fetch:{url.lower()}"
    if tool_name == "web_search":
        query = str(arguments.get("query") or arguments.get("search_term") or "").strip()
        if query:
            return f"web_search:{query.lower()}"
    return None


def repeated_external_lookup_error(
    tool_name: str,
    arguments: dict[str, Any],
    seen_counts: dict[str, int],
) -> str | None:
    """Block repeated external lookups after a small retry budget."""
    signature = external_lookup_signature(tool_name, arguments)
    if signature is None:
        return None
    count = seen_counts.get(signature, 0) + 1
    seen_counts[signature] = count
    if count <= _MAX_REPEAT_EXTERNAL_LOOKUPS:
        return None
    logger.warning(
        "Blocking repeated external lookup {} on attempt {}",
        signature[:160],
        count,
    )
    return (
        "Error: repeated external lookup blocked. "
        "Use the results you already have to answer, or try a meaningfully different source."
    )
