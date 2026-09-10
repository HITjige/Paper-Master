"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any

from nanobot.agent.context_budget import ContextBudget, ContextBudgetManager
from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.config.schema import MemoryConfig
from nanobot.utils.helpers import build_assistant_message, current_time_str, detect_image_mime
from nanobot.utils.prompt_templates import render_template


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    _MAX_RECENT_HISTORY = 50
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"

    def __init__(
        self,
        workspace: Path,
        timezone: str | None = None,
        disabled_skills: list[str] | None = None,
        context_window_tokens: int = 65_536,
        max_completion_tokens: int = 8192,
        context_block_limit: int | None = None,
        memory_config: MemoryConfig | None = None,
    ):
        self.workspace = workspace
        self.timezone = timezone
        self.memory = MemoryStore(workspace)
        self.memory_config = memory_config or MemoryConfig()
        self.context_budget = ContextBudgetManager(ContextBudget(
            context_window_tokens=context_window_tokens,
            output_reserve_tokens=max_completion_tokens,
            context_block_limit=context_block_limit,
        ))
        self.skills = SkillsLoader(workspace, disabled_skills=set(disabled_skills) if disabled_skills else None)

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        memory_query: str = "",
        memory_scope: str | None = None,
    ) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        system_cap = max(128, int(
            self.context_budget.budget.prompt_tokens
            * self.memory_config.system_prompt_max_ratio
        ))
        identity = self.context_budget.truncate_text(
            self._get_identity(channel=channel),
            system_cap,
        )
        parts = [identity] if identity else []
        current_tokens = self.context_budget.text_tokens(identity)
        remaining = max(0, system_cap - current_tokens)

        def _append_optional(
            title: str | None,
            content: str,
            configured_cap: int | None = None,
        ) -> None:
            """Append one section without exceeding the shared system budget."""
            nonlocal remaining
            if not content or remaining <= 0:
                return
            prefix = f"# {title}\n\n" if title else ""
            separator = "\n\n---\n\n" if parts else ""
            overhead = self.context_budget.text_tokens(separator + prefix)
            allowed = remaining - overhead
            if configured_cap is not None:
                allowed = min(configured_cap, allowed)
            if allowed <= 0:
                return
            bounded = self.context_budget.truncate_text(content, allowed)
            if not bounded:
                return
            section = prefix + bounded
            before = self.context_budget.text_tokens("\n\n---\n\n".join(parts))
            candidate = "\n\n---\n\n".join([*parts, section])
            delta = max(0, self.context_budget.text_tokens(candidate) - before)
            if delta > remaining:
                bounded = self.context_budget.truncate_text(
                    content,
                    max(0, allowed - (delta - remaining) - 4),
                )
                if not bounded:
                    return
                section = prefix + bounded
                candidate = "\n\n---\n\n".join([*parts, section])
                delta = max(0, self.context_budget.text_tokens(candidate) - before)
                if delta > remaining:
                    return
            parts.append(section)
            remaining = max(0, remaining - delta)

        bootstrap = self._load_bootstrap_files()
        _append_optional(None, bootstrap)

        always_skills = self.skills.get_always_skills()
        if always_skills:
            _append_optional(
                "Active Skills",
                self.skills.load_skills_for_context(always_skills),
            )

        skills_summary = self.skills.build_skills_summary(exclude=set(always_skills))
        if skills_summary:
            _append_optional(
                None,
                render_template("agent/skills_section.md", skills_summary=skills_summary),
            )

        scopes = self._memory_scopes(memory_scope)
        if self.memory_config.structured_enabled and memory_query.strip():
            records = self.memory.search_memory_records(
                memory_query,
                scopes=scopes,
                top_k=self.memory_config.structured_top_k,
                min_confidence=self.memory_config.min_confidence,
            )
            _append_optional(
                "Relevant Memory",
                self.memory.render_memory_records(records),
                self.memory_config.retrieved_memory_max_tokens,
            )

        memory = self.memory.get_memory_context()
        if memory and not self._is_template_content(self.memory.read_memory(), "memory/MEMORY.md"):
            _append_optional(
                "Memory",
                memory,
                self.memory_config.pinned_memory_max_tokens,
            )

        entries = self.memory.read_unprocessed_history(
            since_cursor=self.memory.get_last_dream_cursor(),
            scopes=scopes,
        )
        if entries:
            capped = entries[-self._MAX_RECENT_HISTORY:]
            recent_lines = [
                f"- [{entry['timestamp']}] {entry['content']}"
                for entry in capped
            ]
            recent = "\n".join(self.context_budget.take_recent_texts(
                recent_lines,
                min(self.memory_config.recent_history_max_tokens, remaining),
            ))
            _append_optional(
                "Recent History",
                recent,
                self.memory_config.recent_history_max_tokens,
            )

        return "\n\n---\n\n".join(parts)

    def _memory_scopes(self, memory_scope: str | None) -> set[str] | None:
        if self.memory_config.scope_mode == "workspace":
            return None
        return {memory_scope} if memory_scope else set()

    def _get_identity(self, channel: str | None = None) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )

    @staticmethod
    def _build_runtime_context(
        channel: str | None, chat_id: str | None, timezone: str | None = None,
        session_summary: str | None = None,
    ) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        lines = [f"Current Time: {current_time_str(timezone)}"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        if session_summary:
            lines += ["", "[Resumed Session]", session_summary]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n" + ContextBuilder._RUNTIME_CONTEXT_END

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """Check if *content* is identical to the bundled template (user hasn't customized it)."""
        try:
            tpl = pkg_files("nanobot") / "templates" / template_path
            if tpl.is_file():
                return content.strip() == tpl.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return False

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        session_summary: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        runtime_ctx = self._build_runtime_context(channel, chat_id, self.timezone, session_summary=session_summary)
        user_content = self._build_user_content(current_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content
        memory_scope = (
            f"{channel}:{chat_id}"
            if channel and chat_id
            else None
        )
        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    skill_names,
                    channel=channel,
                    memory_query=current_message,
                    memory_scope=memory_scope,
                ),
            },
            *history,
        ]
        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            return messages
        messages.append({"role": current_role, "content": merged})
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: Any,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages
