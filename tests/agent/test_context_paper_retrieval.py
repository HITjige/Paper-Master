import json
from pathlib import Path

from nanobot.agent.context import ContextBuilder
from nanobot.agent.paper_kb import PaperKbConfig


def test_context_does_not_inject_papers_by_default(tmp_path: Path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "documents.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "p-1",
                "title": "Diffusion models for EEG synthesis",
                "url": "https://arxiv.org/abs/0000.00001",
                "source": "arxiv",
                "year": 2026,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (kb_dir / "chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "p-1:0",
                "paper_id": "p-1",
                "chunk_index": 0,
                "text": "This work studies EEG synthesis with diffusion models and guidance.",
                "embedding": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    ctx = ContextBuilder(
        tmp_path,
    )
    messages = ctx.build_messages(
        history=[],
        current_message="recent diffusion models for eeg synthesis",
        channel="cli",
        chat_id="direct",
    )
    user_msg = messages[-1]["content"]
    assert isinstance(user_msg, str)
    assert "# Retrieved Papers" not in user_msg


def test_context_injects_retrieved_papers_when_enabled(tmp_path: Path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "documents.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "p-1",
                "title": "Diffusion models for EEG synthesis",
                "url": "https://arxiv.org/abs/0000.00001",
                "source": "arxiv",
                "year": 2026,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (kb_dir / "chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "p-1:0",
                "paper_id": "p-1",
                "chunk_index": 0,
                "text": "This work studies EEG synthesis with diffusion models and guidance.",
                "embedding": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    ctx = ContextBuilder(
        tmp_path,
    )
    messages = ctx.build_messages(
        history=[],
        current_message="recent diffusion models for eeg synthesis",
        channel="cli",
        chat_id="direct",
    )
    user_msg = messages[-1]["content"]
    assert isinstance(user_msg, str)
    assert "# Retrieved Papers" in user_msg
    assert "Diffusion models for EEG synthesis" in user_msg
