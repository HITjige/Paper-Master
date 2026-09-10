import pytest
from pydantic import ValidationError

from nanobot.config.schema import AgentDefaults, SkillConfig


def test_skill_config_defaults_are_serialized_with_camel_case() -> None:
    dumped = AgentDefaults().model_dump(by_alias=True)["skills"]

    assert dumped["trackUsage"] is True
    assert dumped["autoExtractFromPapers"] is True
    assert dumped["autoPromote"] is False
    assert dumped["minEvidenceItems"] == 2


def test_skill_config_validates_bounds() -> None:
    config = SkillConfig.model_validate({"minEvidenceItems": 3, "maxSkillLines": 200})
    assert config.min_evidence_items == 3
    assert config.max_skill_lines == 200

    with pytest.raises(ValidationError):
        SkillConfig(min_evidence_items=0)
