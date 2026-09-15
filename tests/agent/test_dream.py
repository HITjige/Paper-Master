"""Tests for the Dream class — two-phase memory consolidation via AgentRunner."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.memory import Dream, MemoryStore
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.skill_lifecycle import SkillCandidateManager
from nanobot.utils.gitstore import LineAge


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path)
    s.write_soul("# Soul\n- Helpful")
    s.write_user("# User\n- Developer")
    s.write_memory("# Memory\n- Project X active")
    return s


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()

    async def structured_proxy(**kwargs):
        forwarded = dict(kwargs)
        forwarded.pop("json_schema", None)
        return await p.chat_with_retry(**forwarded)

    p.chat_structured_with_retry = AsyncMock(side_effect=structured_proxy)
    return p


@pytest.fixture
def mock_runner():
    return MagicMock()


@pytest.fixture
def dream(store, mock_provider, mock_runner):
    d = Dream(
        store=store,
        provider=mock_provider,
        model="test-model",
        max_batch_size=5,
        skill_candidates=SkillCandidateManager(store.workspace),
    )
    d._runner = mock_runner
    return d


def _make_run_result(
    stop_reason="completed",
    final_content=None,
    tool_events=None,
    usage=None,
):
    return AgentRunResult(
        final_content=final_content or stop_reason,
        stop_reason=stop_reason,
        messages=[],
        tools_used=[],
        usage={},
        tool_events=tool_events or [],
    )


def _phase1_analysis(*, proposals=None, skills=None) -> str:
    return json.dumps(
        {
            "proposals": proposals or [],
            "skills": skills or [],
        }
    )


def _user_proposal(content: str = "The user prefers dark mode") -> dict:
    return {
        "action": "upsert",
        "target": "USER",
        "kind": "preference",
        "subject": "display preference",
        "content": content,
        "old_content": "",
        "confidence": 0.95,
        "valid_from": None,
        "expires_at": None,
        "reason": "The user stated this preference",
    }


class TestDreamRun:
    async def test_noop_when_no_unprocessed_history(self, dream, mock_provider, mock_runner, store):
        """Dream should not call LLM when there's nothing to process."""
        result = await dream.run()
        assert result is False
        mock_provider.chat_with_retry.assert_not_called()
        mock_runner.run.assert_not_called()

    async def test_calls_runner_for_unprocessed_entries(self, dream, mock_provider, mock_runner, store):
        """Dream should call AgentRunner when there are unprocessed history entries."""
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content=_phase1_analysis(proposals=[_user_proposal()])
        )
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))
        result = await dream.run()
        assert result is True
        mock_runner.run.assert_called_once()
        spec = mock_runner.run.call_args[0][0]
        assert spec.max_iterations == 10
        assert spec.fail_on_tool_error is False

    async def test_advances_dream_cursor(self, dream, mock_provider, mock_runner, store):
        """Dream should advance the cursor after processing."""
        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content=_phase1_analysis()
        )
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        assert store.get_last_dream_cursor() == 2

    async def test_compacts_processed_history(self, dream, mock_provider, mock_runner, store):
        """Dream should compact history after processing."""
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content=_phase1_analysis()
        )
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        # After Dream, cursor is advanced and 3, compact keeps last max_history_entries
        entries = store.read_unprocessed_history(since_cursor=0)
        assert all(e["cursor"] > 0 for e in entries)

    async def test_skill_proposal_is_staged_not_published(self, dream, mock_provider, mock_runner, store):
        """Dream should stage validated candidates without changing active Skills."""
        store.append_history("Repeated workflow one")
        store.append_history("Repeated workflow two")
        mock_provider.chat_with_retry.return_value = MagicMock(content="""{
          "proposals": [],
          "skills": [{
            "action": "create",
            "name": "test-skill",
            "description": "Use for a repeated test workflow",
            "when_to_use": ["The repeated test workflow is requested"],
            "steps": ["Prepare the test input", "Run the validated test step"],
            "completion_criteria": ["The test result is verified"]
          }]
        }""")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        mock_runner.run.assert_not_called()
        candidates = dream.skill_candidates.list_candidates(status="draft")
        assert [candidate["name"] for candidate in candidates] == ["test-skill"]
        assert not (store.workspace / "skills" / "test-skill" / "SKILL.md").exists()

    async def test_phase1_uses_constrained_json_and_disables_thinking(
        self, dream, mock_provider, mock_runner, store,
    ):
        store.append_history("nothing durable")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content=_phase1_analysis(),
            finish_reason="stop",
        )

        assert await dream.run() is True

        kwargs = mock_provider.chat_structured_with_retry.call_args.kwargs
        assert kwargs["json_schema"]["required"] == ["proposals", "skills"]
        assert kwargs["max_tokens"] == 4096
        assert kwargs["disable_thinking"] is True
        mock_runner.run.assert_not_called()

    async def test_invalid_phase1_json_retains_cursor_without_editing(
        self, dream, mock_provider, mock_runner, store,
    ):
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Let me analyze the conversation first...",
            finish_reason="stop",
        )

        assert await dream.run() is False
        assert store.get_last_dream_cursor() == 0
        mock_runner.run.assert_not_called()

    async def test_truncated_phase1_retains_cursor_without_editing(
        self, dream, mock_provider, mock_runner, store,
    ):
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content=_phase1_analysis(proposals=[_user_proposal()]),
            finish_reason="length",
        )

        assert await dream.run() is False
        assert store.get_last_dream_cursor() == 0
        mock_runner.run.assert_not_called()

    async def test_phase2_restores_snapshot_before_retry(
        self, dream, mock_provider, mock_runner, store,
    ):
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content=_phase1_analysis(proposals=[_user_proposal()])
        )
        original_user = store.read_user()
        attempts = 0

        async def run_phase2(_spec):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                store.write_user("# User\n- partial edit")
                return _make_run_result(tool_events=[{
                    "name": "edit_file",
                    "status": "error",
                    "detail": "wrong path",
                }])
            assert store.read_user() == original_user
            store.write_user("# User\n- Prefers dark mode")
            return _make_run_result(tool_events=[{
                "name": "edit_file",
                "status": "ok",
                "detail": "USER.md",
            }])

        mock_runner.run = AsyncMock(side_effect=run_phase2)

        assert await dream.run() is True
        assert mock_runner.run.await_count == 2
        assert store.read_user() == "# User\n- Prefers dark mode"
        retry_spec = mock_runner.run.await_args_list[1].args[0]
        assert retry_spec.disable_thinking is True
        assert "Previous attempt errors" in retry_spec.initial_messages[1]["content"]

    async def test_memory_edit_alias_is_normalized(self, dream, store):
        edit_tool = dream._tools.get("edit_file")
        assert edit_tool is not None

        result = await edit_tool.execute(
            path="MEMORY.md",
            old_text="- Project X active",
            new_text="- Project X completed",
        )

        assert "Successfully edited" in result
        assert "- Project X completed" in store.read_memory()

    async def test_disallowed_edit_reports_attempted_path(self, dream):
        edit_tool = dream._tools.get("edit_file")
        assert edit_tool is not None

        result = await edit_tool.execute(
            path="skills/generated/SKILL.md",
            old_text="old",
            new_text="new",
        )

        assert "skills/generated/SKILL.md" in result
        assert "is not allowed" in result

    async def test_dream_has_no_skill_write_tool(self, dream):
        assert dream._tools.get("write_file") is None

    async def test_phase1_prompt_includes_line_age_annotations(self, dream, mock_provider, mock_runner, store):
        """Phase 1 prompt should have per-line age suffixes in MEMORY.md when git is available."""
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        # Init git so line_ages works
        store.git.init()
        store.git.auto_commit("initial memory state")

        await dream.run()

        # The MEMORY.md section should not crash and should contain the memory content
        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        assert "## Current MEMORY.md" in user_msg

    async def test_phase1_annotates_only_memory_not_soul_or_user(self, dream, mock_provider, mock_runner, store):
        """SOUL.md and USER.md should never have age annotations — they are permanent."""
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        store.git.init()
        store.git.auto_commit("initial state")

        await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        # The ← suffix should only appear in MEMORY.md section
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        soul_section = user_msg.split("## Current SOUL.md")[1].split("## Current USER.md")[0]
        user_section = user_msg.split("## Current USER.md")[1]
        # SOUL and USER should not contain age arrows
        assert "\u2190" not in soul_section
        assert "\u2190" not in user_section

    async def test_phase1_prompt_works_without_git(self, dream, mock_provider, mock_runner, store):
        """Phase 1 should work fine even if git is not initialized (no age annotations)."""
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        # Should still succeed — just without age annotations
        mock_provider.chat_with_retry.assert_called_once()
        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        assert "## Current MEMORY.md" in user_msg

    async def test_phase1_prompt_carries_age_suffix_for_stale_lines(
        self, dream, mock_provider, mock_runner, store,
    ):
        """End-to-end: ages >14d must appear verbatim in the LLM prompt, ages ≤14d must not."""
        # MEMORY.md fixture has 2 non-blank lines ("# Memory" and "- Project X active").
        # Inject four ages to cover threshold boundaries: >14 suffix, ==14 no suffix, <14 no suffix.
        store.write_memory("# Memory\n- Project X active\n- fresh item\n- edge case line")
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        fake_ages = [
            LineAge(age_days=30),   # "# Memory"        → should get ← 30d
            LineAge(age_days=20),   # "- Project X..."  → should get ← 20d
            LineAge(age_days=14),   # "- fresh item"    → ==14, threshold is strictly >14, no suffix
            LineAge(age_days=5),    # "- edge case..."  → no suffix
        ]
        with patch.object(store.git, "line_ages", return_value=fake_ages):
            await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        assert "\u2190 30d" in memory_section
        assert "\u2190 20d" in memory_section
        assert "\u2190 14d" not in memory_section
        assert "\u2190 5d" not in memory_section

    async def test_phase1_skips_annotation_when_disabled(
        self, dream, mock_provider, mock_runner, store,
    ):
        """`annotate_line_ages=False` must bypass the git lookup entirely and keep MEMORY.md raw."""
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        dream.annotate_line_ages = False
        # line_ages must be bypassed entirely — verify with a spy rather than a
        # raising side_effect, because _annotate_with_ages catches Exception
        # (which swallows AssertionError) and would hide an accidental call.
        with patch.object(store.git, "line_ages") as mock_line_ages:
            await dream.run()
            mock_line_ages.assert_not_called()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        assert "\u2190" not in user_msg

    async def test_phase1_skips_annotation_on_line_ages_length_mismatch(
        self, dream, mock_provider, mock_runner, store,
    ):
        """If ages length != lines length (dirty working tree), skip annotation instead of mis-tagging."""
        # MEMORY.md has 2 non-blank lines but we hand back only 1 age → mismatch.
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        with patch.object(store.git, "line_ages", return_value=[LineAge(age_days=999)]):
            await dream.run()

        call_args = mock_provider.chat_with_retry.call_args
        user_msg = call_args.kwargs.get("messages", call_args[1].get("messages"))[1]["content"]
        memory_section = user_msg.split("## Current MEMORY.md")[1].split("## Current SOUL.md")[0]
        # No age arrow at all — we refused to annotate rather than tag the wrong line.
        assert "\u2190" not in memory_section

    async def test_phase1_prompt_uses_threshold_from_template_var(
        self, dream, mock_provider, mock_runner, store,
    ):
        """System prompt should reference the stale-threshold constant, not a hardcoded 14."""
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        await dream.run()

        system_msg = mock_provider.chat_with_retry.call_args.kwargs["messages"][0]["content"]
        # The template renders with stale_threshold_days=14 → LLM must see "N>14"
        assert "N>14" in system_msg
