"""Distil successful paper workflows into validated Skill candidates."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.skill_lifecycle import SkillCandidateManager
from nanobot.agent.skills import SkillsLoader
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.agent.memory import MemoryStore
    from nanobot.providers.base import LLMProvider


class SkillExtractor:
    """Create reviewable Skill proposals from a completed paper-agent trace.

    The extractor deliberately has no filesystem tools. The model produces a
    structured proposal; deterministic code renders, validates and stages it.
    This prevents an extraction prompt from editing arbitrary workspace files
    or immediately changing the active Skill catalog.
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        workspace: Path,
        *,
        candidate_manager: SkillCandidateManager | None = None,
        min_evidence_items: int = 2,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
    ):
        # Kept for compatibility with older callers. Extraction is now one
        # structured LLM call without filesystem tools.
        del max_iterations, max_tool_result_chars
        self.store = store
        self.provider = provider
        self.model = model
        self.workspace = workspace.resolve()
        self.candidate_manager = candidate_manager or SkillCandidateManager(self.workspace)
        self.min_evidence_items = max(1, min_evidence_items)
        self.last_candidate: dict[str, Any] | None = None

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any] | None:
        raw = content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _list_existing_skills(self) -> list[str]:
        loader = SkillsLoader(self.workspace)
        entries: list[str] = []
        for skill in loader.list_skills(filter_unavailable=False):
            name = skill["name"]
            metadata = loader.get_skill_metadata(name) or {}
            description = str(metadata.get("description") or name).strip()
            entries.append(f"{name} — {description}")
        for candidate in self.candidate_manager.list_candidates(status="draft"):
            entries.append(
                f"{candidate.get('name', '')} [draft] — "
                f"{candidate.get('description', '')}"
            )
        return sorted(entries)

    @staticmethod
    def _evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(kind: str, identifier: Any, title: Any = "") -> None:
            value = str(identifier or "").strip()
            key = value.casefold()
            if not value or key in seen:
                return
            seen.add(key)
            item = {"kind": kind, "id": value}
            if str(title or "").strip():
                item["title"] = str(title).strip()[:300]
            evidence.append(item)

        for chunk in result.get("retrieval_results", []) or []:
            if not isinstance(chunk, dict):
                continue
            add(
                "paper",
                chunk.get("paper_id") or chunk.get("chunk_id"),
                chunk.get("paper_title") or chunk.get("title"),
            )
        for paper in result.get("external_papers", []) or []:
            if not isinstance(paper, dict):
                continue
            add("paper", paper.get("paper_id") or paper.get("id"), paper.get("title"))
        for citation in result.get("citations", []) or []:
            citation_text = str(citation or "").strip()
            match = re.match(r"^\[([^\]]+)\]", citation_text)
            add("paper" if match else "citation", match.group(1) if match else citation_text)
        return evidence[:40]

    @staticmethod
    def _format_trace(result: dict[str, Any]) -> str:
        """Format state transitions, outcomes and bounded evidence for analysis."""
        trace = {
            "routing": {
                "decision": result.get("routing_decision"),
                "reasoning": result.get("routing_reasoning"),
            },
            "query_processing": {
                "rewritten_queries": result.get("rewritten_queries", []),
                "sub_queries": result.get("sub_queries_detail", []),
                "entities": result.get("extracted_entities", []),
                "fallback_used": result.get("rewrite_fallback_used", False),
            },
            "retrieval": {
                "quality": result.get("retrieval_quality"),
                "internal_result_count": len(result.get("retrieval_results", []) or []),
                "external_paper_count": len(result.get("external_papers", []) or []),
                "ingested_papers": result.get("ingested_papers", []),
            },
            "critic": {
                "verdict": result.get("critic_verdict"),
                "issues": result.get("critic_issues", []),
                "suggestion": result.get("critic_suggestion", ""),
                "feedback": result.get("critic_feedback", ""),
            },
            "outcome": {
                "complete": result.get("is_complete"),
                "iterations": result.get("iteration_count", 0),
                "invalid_citations": result.get("invalid_citations", []),
                "citations": result.get("citations", []),
            },
        }
        return json.dumps(trace, ensure_ascii=False, indent=2, default=str)[:16_000]

    @staticmethod
    def _format_paper_context(result: dict[str, Any]) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for chunk in result.get("retrieval_results", []) or []:
            if not isinstance(chunk, dict):
                continue
            if len(parts) >= 8:
                break
            paper_id = chunk.get("paper_id") or chunk.get("chunk_id", "")
            title = chunk.get("paper_title") or chunk.get("title", "unknown")
            text = str(chunk.get("text", ""))[:600]
            parts.append(f"{len(parts) + 1}. [{paper_id}] {title}\n{text}")
            if paper_id:
                seen.add(str(paper_id).casefold())
        for paper in result.get("external_papers", []) or []:
            if not isinstance(paper, dict) or len(parts) >= 8:
                continue
            paper_id = paper.get("paper_id") or paper.get("id", "")
            if paper_id and str(paper_id).casefold() in seen:
                continue
            title = paper.get("title", "unknown")
            abstract = str(paper.get("abstract", ""))[:600]
            parts.append(f"{len(parts) + 1}. [{paper_id}] {title}\n{abstract}")
            if paper_id:
                seen.add(str(paper_id).casefold())
        return "\n\n".join(parts)

    @staticmethod
    def _successful_trace(result: dict[str, Any]) -> bool:
        if result.get("is_complete") is not True:
            return False
        verdict = str(result.get("critic_verdict", "")).strip().lower()
        if verdict != "passed":
            return False
        if result.get("invalid_citations"):
            return False
        return bool(str(result.get("final_answer", "")).strip())

    async def extract(self, multi_agent_result: dict[str, Any]) -> bool:
        """Stage a candidate when a successful trace contains reusable procedure."""
        user_query = str(multi_agent_result.get("user_query", "")).strip()
        final_answer = str(multi_agent_result.get("final_answer", "")).strip()
        if not user_query or not final_answer or not self._successful_trace(multi_agent_result):
            return False

        evidence = self._evidence(multi_agent_result)
        if len(evidence) < self.min_evidence_items:
            logger.info(
                "SkillExtractor: only {} evidence item(s), need {}; skipping",
                len(evidence),
                self.min_evidence_items,
            )
            return False

        existing = self._list_existing_skills()
        prompt = (
            f"## Current Date\n{datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"## Existing Skill Catalog\n"
            + ("\n".join(f"- {item}" for item in existing) or "(empty)")
            + f"\n\n## User Query\n{user_query[:4000]}\n\n"
            f"## Successful Workflow Trace\n{self._format_trace(multi_agent_result)}\n\n"
            f"## Grounded Paper Excerpts\n{self._format_paper_context(multi_agent_result)}\n\n"
            f"## Final Answer Excerpt\n{final_answer[:5000]}"
        )
        try:
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template("agent/skill_extract_phase1.md", strip=True),
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            if response.finish_reason == "error":
                raise RuntimeError(f"Skill extraction returned an error: {response.content}")
            payload = self._parse_json_object(response.content or "")
        except Exception:
            logger.exception("SkillExtractor analysis failed")
            return False

        if not payload or str(payload.get("decision", "")).lower() != "candidate":
            logger.info("SkillExtractor: trace did not justify a reusable Skill")
            return False
        proposal = payload.get("proposal")
        if not isinstance(proposal, dict):
            logger.warning("SkillExtractor returned candidate without a proposal")
            return False

        try:
            self.last_candidate = self.candidate_manager.stage(
                proposal,
                source="paper_multi_agent",
                evidence=evidence,
            )
        except (OSError, ValueError):
            logger.exception("SkillExtractor candidate validation/staging failed")
            return False
        logger.info(
            "SkillExtractor staged {} candidate {} ({})",
            self.last_candidate["status"],
            self.last_candidate["candidate_id"],
            self.last_candidate["name"],
        )
        return True
