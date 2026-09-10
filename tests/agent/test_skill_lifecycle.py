from __future__ import annotations

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.agent.skill_extractor import SkillExtractor
from nanobot.agent.skill_lifecycle import SkillCandidateManager, SkillUsageStore
from nanobot.agent.skills import SkillsLoader
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _proposal(name: str = "paper-comparison") -> dict:
    return {
        "action": "create",
        "name": name,
        "description": "Use when comparing multiple research methods with grounded evidence",
        "when_to_use": ["A query asks for a method comparison across papers"],
        "steps": ["Define comparison dimensions", "Retrieve evidence for every method"],
        "completion_criteria": ["Every comparison claim maps to supplied evidence"],
        "failure_recovery": ["State missing evidence and narrow the comparison"],
        "examples": ["Compare method A with method B"],
    }


def test_candidate_lifecycle_stages_then_promotes_and_versions(tmp_path):
    manager = SkillCandidateManager(tmp_path)

    first = manager.stage(_proposal(), source="test", evidence=[{"id": "p1"}])
    assert first["status"] == "draft"
    assert not (tmp_path / "skills" / "paper-comparison" / "SKILL.md").exists()

    promoted = manager.promote(first["candidate_id"])
    assert promoted["status"] == "promoted"
    active = tmp_path / "skills" / "paper-comparison" / "SKILL.md"
    assert "## Completion Criteria" in active.read_text(encoding="utf-8")

    updated = _proposal()
    updated["action"] = "update"
    updated["steps"].append("Report trade-offs")
    second = manager.stage(updated, source="test", evidence=[{"id": "p2"}])
    manager.promote(second["candidate_id"])

    revisions = list((tmp_path / "skills" / ".history" / "paper-comparison").iterdir())
    assert len(revisions) == 1
    assert "Report trade-offs" in active.read_text(encoding="utf-8")


def test_drafts_are_hidden_from_model_catalog_until_promoted(tmp_path):
    manager = SkillCandidateManager(tmp_path)
    draft = manager.stage(_proposal(), source="test")
    loader = SkillsLoader(tmp_path)

    assert "paper-comparison" not in {
        item["name"] for item in loader.list_skills(filter_unavailable=False)
    }

    manager.promote(draft["candidate_id"])
    assert "paper-comparison" in {
        item["name"] for item in loader.list_skills(filter_unavailable=False)
    }


def test_auto_promote_never_overwrites_an_existing_skill(tmp_path):
    manager = SkillCandidateManager(tmp_path, auto_promote=True)
    created = manager.stage(_proposal(), source="test")
    assert created["status"] == "promoted"

    update = _proposal()
    update["action"] = "update"
    update["steps"] = ["Potential replacement", "Verify replacement behavior"]
    staged = manager.stage(update, source="test")

    assert staged["status"] == "draft"
    assert "Potential replacement" not in (
        tmp_path / "skills" / "paper-comparison" / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_candidate_rejects_builtin_name(tmp_path):
    manager = SkillCandidateManager(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        manager.stage(_proposal("paper-expert"), source="test")


def test_candidate_rejects_duplicate_open_draft(tmp_path):
    manager = SkillCandidateManager(tmp_path)
    manager.stage(_proposal(), source="test")

    with pytest.raises(ValueError, match="draft candidate already exists"):
        manager.stage(_proposal(), source="test")


def test_usage_store_records_activation_once_per_callback_and_outcome(tmp_path):
    skill = tmp_path / "skills" / "local-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: local-skill\ndescription: Local\n---\n", encoding="utf-8")
    store = SkillUsageStore(tmp_path)
    run_id = store.new_run_id()

    assert store.record_activation(
        skill,
        session_key="cli:test",
        run_id=run_id,
    ) == "local-skill"
    store.record_outcome(
        {"local-skill"},
        success=True,
        session_key="cli:test",
        run_id=run_id,
    )

    stats = store.stats()
    assert stats[0]["activation_count"] == 1
    assert stats[0]["success_count"] == 1
    assert stats[0]["failure_count"] == 0
    with sqlite3.connect(store.database) as connection:
        outcome = connection.execute(
            "SELECT run_id, completed FROM skill_outcome_events"
        ).fetchone()
    assert outcome == (run_id, 1)


@pytest.mark.asyncio
async def test_runner_emits_activation_only_after_successful_skill_read(tmp_path):
    skill = tmp_path / "skills" / "local-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("skill body", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(ReadFileTool(workspace=tmp_path, allowed_dir=tmp_path))
    paths: list[str] = []
    spec = AgentRunSpec(
        initial_messages=[],
        tools=tools,
        model="test",
        max_iterations=1,
        max_tool_result_chars=1000,
        skill_activation_callback=paths.append,
    )

    result, event, error = await AgentRunner(MagicMock())._run_tool(
        spec,
        ToolCallRequest(
            id="read-1",
            name="read_file",
            arguments={"path": "skills/local-skill/SKILL.md"},
        ),
        {},
    )

    assert "skill body" in result
    assert event["status"] == "ok"
    assert error is None
    assert paths == ["skills/local-skill/SKILL.md"]


@pytest.mark.asyncio
async def test_paper_extractor_stages_structured_candidate(tmp_path):
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=json.dumps({"decision": "candidate", "proposal": _proposal()}),
    ))
    manager = SkillCandidateManager(tmp_path)
    extractor = SkillExtractor(
        MemoryStore(tmp_path),
        provider,
        "test",
        tmp_path,
        candidate_manager=manager,
        min_evidence_items=2,
    )
    trace = {
        "user_query": "Compare two paper methods",
        "final_answer": "Grounded comparison",
        "is_complete": True,
        "critic_verdict": "passed",
        "retrieval_results": [
            {"paper_id": "p1", "paper_title": "Paper One", "text": "method one"},
            {"paper_id": "p2", "paper_title": "Paper Two", "text": "method two"},
        ],
        "citations": ["[p1] Paper One", "[p2] Paper Two"],
    }

    assert await extractor.extract(trace) is True
    assert extractor.last_candidate is not None
    assert manager.list_candidates(status="draft")[0]["source"] == "paper_multi_agent"
    assert not (tmp_path / "skills" / "paper-comparison" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_paper_extractor_rejects_failed_trace_without_llm_call(tmp_path):
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock()
    extractor = SkillExtractor(
        MemoryStore(tmp_path),
        provider,
        "test",
        tmp_path,
        min_evidence_items=1,
    )

    accepted = await extractor.extract({
        "user_query": "question",
        "final_answer": "draft",
        "is_complete": False,
        "critic_verdict": "needs_revision",
        "retrieval_results": [{"paper_id": "p1"}],
    })

    assert accepted is False
    provider.chat_with_retry.assert_not_called()
