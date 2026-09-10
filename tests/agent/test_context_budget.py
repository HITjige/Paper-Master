from nanobot.agent.context_budget import ContextBudget, ContextBudgetManager
from nanobot.utils.helpers import estimate_message_tokens


def test_prompt_budget_respects_output_safety_and_explicit_block_limit():
    normal = ContextBudget(
        context_window_tokens=10_000,
        output_reserve_tokens=2_000,
        safety_buffer_tokens=1_000,
    )
    limited = ContextBudget(
        context_window_tokens=10_000,
        output_reserve_tokens=2_000,
        safety_buffer_tokens=1_000,
        context_block_limit=5_000,
    )

    assert normal.prompt_tokens == 7_000
    assert limited.prompt_tokens == 5_000


def test_token_truncation_keeps_newest_content_within_budget():
    text = "old-prefix " * 1000 + "LATEST_USER_REQUIREMENT"
    truncated = ContextBudgetManager.truncate_text(
        text,
        80,
        keep_tail=True,
    )

    assert truncated.count("old-prefix") < 100
    assert "LATEST_USER_REQUIREMENT" in truncated
    assert ContextBudgetManager.text_tokens(truncated) <= 80


def test_trim_messages_preserves_system_and_bounded_latest_user():
    manager = ContextBudgetManager(ContextBudget(
        context_window_tokens=500,
        output_reserve_tokens=100,
        safety_buffer_tokens=100,
    ))
    messages = [
        {"role": "system", "content": "system policy"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "x " * 2000 + "LATEST_REQUIREMENT"},
    ]

    trimmed = manager.trim_messages(messages)

    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["role"] == "user"
    assert "LATEST_REQUIREMENT" in trimmed[-1]["content"]
    assert estimate_message_tokens(trimmed[-1]) < estimate_message_tokens(messages[-1])
