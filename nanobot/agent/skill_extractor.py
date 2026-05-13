"""Skill Extractor — background agent that distills paper discussions into reusable skills.

Runs asynchronously after multi-agent workflows, following the same pattern as Dream:
Phase 1: Analyze conversation + paper content → determine if skill-worthy
Phase 2: Delegate to AgentRunner to create/edit SKILL.md files
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.agent.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.agent.memory import MemoryStore
    from nanobot.providers.base import LLMProvider


class SkillExtractor:
    """Background agent that creates/updates skills from paper-driven discussions.

    Reuses the same two-phase pattern as Dream:
    1. Plain LLM call analyzes the conversation for reusable insights
    2. AgentRunner with file tools creates or edits SKILL.md files

    Usage:
        extractor = SkillExtractor(store, provider, model, workspace)
        await extractor.extract(multi_agent_result)
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        workspace: Path,
        *,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.workspace = workspace
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self._runner = AgentRunner(provider)
        self._tools = self._build_tools()

    # -- tool registry -------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        """Build a minimal tool registry for the Skill Extractor agent.

        Same tool set as Dream: read_file for reference, edit_file for
        updating existing skills, write_file (restricted to skills/) for
        creating new ones.
        """
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        tools = ToolRegistry()
        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        tools.register(
            ReadFileTool(
                workspace=self.workspace,
                allowed_dir=self.workspace,
                extra_allowed_dirs=extra_read,
            )
        )
        tools.register(EditFileTool(workspace=self.workspace, allowed_dir=self.workspace))
        skills_dir = self.workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=skills_dir))
        return tools

    # -- skill listing --------------------------------------------------------

    def _list_existing_skills(self) -> list[str]:
        """List existing skills as 'name — description' for dedup context."""
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        _DESC_RE = re.compile(r"^description:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
        entries: dict[str, str] = {}
        for base in (self.workspace / "skills", BUILTIN_SKILLS_DIR):
            if not base.exists():
                continue
            for d in base.iterdir():
                if not d.is_dir():
                    continue
                skill_md = d / "SKILL.md"
                if not skill_md.exists():
                    continue
                if d.name in entries and base == BUILTIN_SKILLS_DIR:
                    continue
                content = skill_md.read_text(encoding="utf-8")[:500]
                m = _DESC_RE.search(content)
                desc = m.group(1).strip() if m else "(no description)"
                entries[d.name] = desc
        return [f"{name} — {desc}" for name, desc in sorted(entries.items())]

    # -- main entry ----------------------------------------------------------

    async def extract(self, multi_agent_result: dict[str, Any]) -> bool:
        """Analyze a multi-agent conversation and create/update skills if warranted.

        Args:
            multi_agent_result: Dict from MultiAgentGraph.run() containing at least:
                - user_query: original user question
                - final_answer: synthesized response
                - retrieval_results: KB chunks (optional)
                - external_papers: external paper metadata (optional)

        Returns:
            True if a skill was created/updated, False otherwise.
        """
        user_query = str(multi_agent_result.get("user_query", ""))
        final_answer = str(multi_agent_result.get("final_answer", ""))
        if not user_query or not final_answer:
            return False

        current_memory = self.store.read_memory() or "(empty)"
        current_date = datetime.now().strftime("%Y-%m-%d")
        paper_context = self._format_paper_context(multi_agent_result)

        # ---- Phase 1: Analyze -----------------------------------------------
        phase1_prompt = (
            f"## Current Date\n{current_date}\n\n"
            f"## Current MEMORY.md ({len(current_memory)} chars)\n{current_memory[:8000]}\n\n"
            f"## User Query\n{user_query}\n\n"
        )
        if paper_context:
            phase1_prompt += f"## Retrieved Paper Context\n{paper_context}\n\n"
        phase1_prompt += f"## Synthesized Answer\n{final_answer[:4000]}"

        try:
            phase1_response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template(
                            "agent/skill_extract_phase1.md",
                            strip=True,
                        ),
                    },
                    {"role": "user", "content": phase1_prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            analysis = phase1_response.content or ""
            logger.debug(
                "SkillExtractor Phase 1 ({} chars): {}",
                len(analysis),
                analysis[:500],
            )
        except Exception:
            logger.exception("SkillExtractor Phase 1 failed")
            return False

        if "[SKIP]" in analysis[:500] and "[SKILL]" not in analysis[:500]:
            logger.info("SkillExtractor: nothing skill-worthy, skipping")
            return False

        # ---- Phase 2: Delegate to AgentRunner -------------------------------
        existing_skills = self._list_existing_skills()
        skills_section = (
            "\n".join(f"- {s}" for s in existing_skills)
            if existing_skills
            else "(no existing skills)"
        )

        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        skill_creator_path = str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md")
        memory_preview = current_memory[:4000]

        phase2_prompt = (
            f"## Analysis\n{analysis}\n\n"
            f"## Existing Skills\n{skills_section}\n\n"
            f"## Current MEMORY.md\n{memory_preview}"
        )

        try:
            result = await self._runner.run(
                AgentRunSpec(
                    initial_messages=[
                        {
                            "role": "system",
                            "content": render_template(
                                "agent/skill_extract_phase2.md",
                                strip=True,
                                skill_creator_path=skill_creator_path,
                            ),
                        },
                        {"role": "user", "content": phase2_prompt},
                    ],
                    tools=self._tools,
                    model=self.model,
                    max_iterations=self.max_iterations,
                    max_tool_result_chars=self.max_tool_result_chars,
                    error_message="Skill extraction encountered an error.",
                    workspace=self.workspace,
                )
            )
            tools_used = result.tools_used or []
            logger.info(
                "SkillExtractor Phase 2 complete: {} tools called ({})",
                len(tools_used),
                ", ".join(tools_used[:10]),
            )
            return len(tools_used) > 0
        except Exception:
            logger.exception("SkillExtractor Phase 2 failed")
            return False

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _format_paper_context(result: dict[str, Any]) -> str:
        """Format retrieved paper chunks and external papers for the prompt."""
        parts: list[str] = []

        retrieval_results = result.get("retrieval_results", []) or []
        if retrieval_results:
            parts.append("### Internal KB Chunks")
            for i, r in enumerate(retrieval_results[:8], 1):
                title = r.get("paper_title") or r.get("title", "unknown")
                pid = r.get("paper_id", "")
                text = str(r.get("text", ""))[:600]
                parts.append(f"{i}. [{pid}] {title}\n   {text}")

        external_papers = result.get("external_papers", []) or []
        if external_papers:
            parts.append("### External Papers (from arXiv)")
            for i, p in enumerate(external_papers[:8], 1):
                title = p.get("title", "unknown")
                pid = p.get("paper_id", "")
                abstract = str(p.get("abstract", ""))[:400]
                parts.append(f"{i}. [{pid}] {title}\n   {abstract}")

        return "\n\n".join(parts) if parts else ""
