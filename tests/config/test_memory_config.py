from pydantic import ValidationError
import pytest

from nanobot.config.schema import AgentDefaults, MemoryConfig


def test_memory_config_defaults_are_serialized_with_camel_case() -> None:
    dumped = AgentDefaults().model_dump(by_alias=True)["memory"]

    assert dumped["scopeMode"] == "workspace"
    assert dumped["structuredEnabled"] is True
    assert dumped["structuredTopK"] == 8
    assert dumped["systemPromptMaxRatio"] == 0.35


def test_memory_config_accepts_session_scope_and_validates_bounds() -> None:
    config = MemoryConfig.model_validate({
        "scopeMode": "session",
        "structuredTopK": 12,
        "minConfidence": 0.7,
    })

    assert config.scope_mode == "session"
    assert config.structured_top_k == 12
    assert config.min_confidence == 0.7

    with pytest.raises(ValidationError):
        MemoryConfig(system_prompt_max_ratio=0.95)
