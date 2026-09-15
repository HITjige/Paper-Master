from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.autocompact import AutoCompact
from nanobot.agent.memory import Consolidator, Dream, MemoryStore
from nanobot.agent.runner import AgentRunResult
from nanobot.session.manager import SessionManager


def _run_result(stop_reason: str) -> AgentRunResult:
    return AgentRunResult(
        final_content=stop_reason,
        messages=[],
        stop_reason=stop_reason,
    )


def test_dream_failure_restores_files_and_retains_cursor(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_memory("# Memory\n- trusted state")
    store.append_history("candidate update", session_key="cli:test")
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=MagicMock(
        content="[MEMORY] uncommitted candidate",
        finish_reason="stop",
    ))
    provider.chat_structured_with_retry = provider.chat_with_retry
    dream = Dream(store=store, provider=provider, model="test")

    async def _failed_run(_spec):
        store.write_memory("# Memory\n- partial bad edit")
        return _run_result("max_iterations")

    dream._runner.run = AsyncMock(side_effect=_failed_run)

    assert asyncio.run(dream.run()) is False
    assert store.get_last_dream_cursor() == 0
    assert store.read_memory() == "# Memory\n- trusted state"
    assert len(store.read_unprocessed_history(0)) == 1


def test_concurrent_history_appends_keep_unique_ordered_cursors(tmp_path):
    store = MemoryStore(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        cursors = list(executor.map(
            lambda index: store.append_history(
                f"event {index}",
                session_key="cli:alice",
            ),
            range(40),
        ))

    assert sorted(cursors) == list(range(1, 41))
    entries = store.read_unprocessed_history(0)
    assert [entry["cursor"] for entry in entries] == list(range(1, 41))


def test_dream_cursor_never_moves_backwards(tmp_path):
    store = MemoryStore(tmp_path)

    store.set_last_dream_cursor(8)
    store.set_last_dream_cursor(3)

    assert store.get_last_dream_cursor() == 8


def test_successful_dream_commits_scoped_structured_memory(tmp_path):
    store = MemoryStore(tmp_path)
    store.append_history("Project Atlas decision", session_key="cli:alice")
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=MagicMock(
        content="[MEMORY] Project Atlas stores memory in SQLite",
        finish_reason="stop",
    ))
    provider.chat_structured_with_retry = provider.chat_with_retry
    dream = Dream(store=store, provider=provider, model="test")
    dream._runner.run = AsyncMock(return_value=_run_result("completed"))

    assert asyncio.run(dream.run()) is True
    assert store.get_last_dream_cursor() == 1
    records = store.search_memory_records(
        "Atlas SQLite",
        scopes={"cli:alice"},
    )
    assert records[0]["content"] == "Project Atlas stores memory in SQLite"
    assert records[0]["source_cursor"] == 1


def test_structured_subject_correction_supersedes_previous_value(tmp_path):
    store = MemoryStore(tmp_path)
    first = store.upsert_memory_record(
        scope="cli:alice",
        kind="preference",
        subject="editor_theme",
        content="Editor theme is dark",
        confidence=0.8,
    )
    second = store.upsert_memory_record(
        scope="cli:alice",
        kind="preference",
        subject="editor_theme",
        content="Editor theme is light",
        confidence=0.9,
    )

    assert first != second
    dark_query_results = store.search_memory_records(
        "dark editor theme",
        scopes={"cli:alice"},
    )
    assert all(
        result["content"] != "Editor theme is dark"
        for result in dark_query_results
    )
    current = store.search_memory_records(
        "light editor theme",
        scopes={"cli:alice"},
    )
    assert current[0]["content"] == "Editor theme is light"
    assert current[0]["supersedes"] == first


def test_structured_remove_invalidates_matching_subject(tmp_path):
    store = MemoryStore(tmp_path)
    store.upsert_memory_record(
        scope="cli:alice",
        kind="fact",
        subject="active_project",
        content="The active project is Atlas",
    )

    changed = store.apply_memory_proposals(
        [{
            "action": "remove",
            "kind": "fact",
            "subject": "active_project",
            "old_content": "The active project is Atlas",
        }],
        scope="cli:alice",
    )

    assert len(changed) == 1
    assert store.search_memory_records(
        "active project Atlas",
        scopes={"cli:alice"},
    ) == []


def test_structured_proposal_batch_rolls_back_on_error(tmp_path):
    store = MemoryStore(tmp_path)

    with pytest.raises(ValueError):
        store.apply_memory_proposals(
            [
                {
                    "action": "upsert",
                    "kind": "fact",
                    "content": "This must be rolled back",
                    "confidence": 0.9,
                },
                {
                    "action": "upsert",
                    "kind": "fact",
                    "content": "Invalid confidence",
                    "confidence": "not-a-number",
                },
            ],
            scope="cli:alice",
        )

    assert store.search_memory_records(
        "rolled back",
        scopes={"cli:alice"},
    ) == []


def test_consolidator_failure_creates_bounded_checkpoint_and_raw_artifact(tmp_path):
    store = MemoryStore(tmp_path)
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("offline"))
    sessions = MagicMock()
    consolidator = Consolidator(
        store=store,
        provider=provider,
        model="test",
        sessions=sessions,
        context_window_tokens=4096,
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
    )

    checkpoint = asyncio.run(consolidator.archive(
        [
            {"role": "user", "content": "critical request"},
            {"role": "assistant", "content": "unfinished response"},
        ],
        session_key="cli:test",
    ))

    assert checkpoint and "DEGRADED CHECKPOINT" in checkpoint
    entry = store.read_unprocessed_history(0)[0]
    assert entry["scope"] == "cli:test"
    assert (tmp_path / entry["evidence"]["artifact"]).exists()
    assert len(entry["content"]) < 3000


def test_auto_compact_does_not_evict_without_checkpoint(tmp_path):
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("cli:test")
    for index in range(8):
        session.add_message("user", f"question {index}")
        session.add_message("assistant", f"answer {index}")
    sessions.save(session)
    consolidator = MagicMock()
    consolidator.archive = AsyncMock(return_value=None)
    compact = AutoCompact(sessions, consolidator, session_ttl_minutes=1)

    asyncio.run(compact._archive("cli:test"))

    sessions.invalidate("cli:test")
    assert len(sessions.get_or_create("cli:test").messages) == 16
