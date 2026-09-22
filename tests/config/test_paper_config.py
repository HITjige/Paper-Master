from nanobot.config.schema import PaperToolsConfig


def test_removed_auto_context_settings_are_not_part_of_paper_config() -> None:
    fields = PaperToolsConfig.model_fields

    assert "auto_context_retrieve" not in fields
    assert "auto_context_top_k" not in fields
