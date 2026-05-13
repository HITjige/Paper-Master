"""Tests for paper semantic chunking and hypothetical question retrieval."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.agent.tools.paper import (
    _split_markdown_semantic,
    _generate_chunk_metadata,
    _remove_noisy_blocks,
)
from nanobot.agent.paper_kb import (
    PaperKbConfig,
    PaperKnowledgeBase,
)


# Sample markdown text for testing
SAMPLE_MARKDOWN = """
# Abstract

This paper presents a novel approach to semantic chunking of scientific documents. We propose using Markdown headers to identify natural boundaries in the document structure, enabling more precise retrieval of relevant content.

## Introduction

The problem of document chunking is fundamental to many information retrieval systems. Traditional approaches use fixed-size chunks or simple paragraph boundaries, which often fail to capture the semantic structure of scientific papers.

### Motivation

Scientific papers have a well-defined structure with sections like Introduction, Methods, Results, and Conclusion. Each section serves a specific purpose and contains related information. By preserving this structure during chunking, we can improve retrieval accuracy.

### Related Work

Previous work on document chunking includes various approaches such as sliding windows, sentence boundaries, and topic modeling. However, these methods do not explicitly leverage the document structure.

## Method

Our approach uses Markdown header levels to split documents. We process each header level (#, ##, ###) as a potential boundary.

### Algorithm

1. Parse Markdown headers
2. Create chunks at each header boundary
3. Preserve header context in each chunk
4. Generate embeddings for retrieval

## Results

We evaluated our method on a corpus of 1000 scientific papers. The results show significant improvement over baseline methods.

| Method | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Fixed-size | 0.65 | 0.72 | 0.68 |
| Our method | 0.85 | 0.88 | 0.86 |

## Conclusion

We presented a novel Markdown-based chunking method that improves retrieval accuracy for scientific documents. Future work will explore dynamic chunk sizing and multi-document chunking.

## References

[1] Smith et al., 2023
[2] Johnson et al., 2024

## Acknowledgements

We thank the anonymous reviewers for their helpful comments.
"""


class TestMarkdownSemanticChunking:
    """Tests for _split_markdown_semantic function."""

    def test_basic_chunking(self):
        """Test basic Markdown header-based chunking."""
        chunks = _split_markdown_semantic(SAMPLE_MARKDOWN)
        
        assert len(chunks) > 0
        # Should have chunks for Abstract, Introduction, Method, Results, Conclusion
        sections = [c["section"] for c in chunks]
        assert "abstract" in sections or "Abstract" in sections
        
        # Each chunk should have required fields
        for chunk in chunks:
            assert "section" in chunk
            assert "text" in chunk
            assert "heading_level" in chunk
            assert "heading_path" in chunk
            assert len(chunk["text"]) > 0

    def test_noisy_blocks_removed(self):
        """Test that references and acknowledgements are removed."""
        clean_text = _remove_noisy_blocks(SAMPLE_MARKDOWN)
        
        assert "References" not in clean_text
        assert "Acknowledgements" not in clean_text
        assert "Smith et al." not in clean_text
        
        # Should retain main content
        assert "Abstract" in clean_text
        assert "Introduction" in clean_text

    def test_empty_text(self):
        """Test handling of empty text."""
        chunks = _split_markdown_semantic("")
        assert chunks == []

    def test_no_headers(self):
        """Test handling of text without headers."""
        text = "This is plain text without any headers.\n\nAnother paragraph."
        chunks = _split_markdown_semantic(text, min_chunk_chars=10)
        
        # Should return as single chunk
        assert len(chunks) >= 1
        assert chunks[0]["section"] == "content"

    def test_chunk_size_limits(self):
        """Test that chunks respect size limits."""
        # Long text that should be split
        long_text = "# Section\n\n" + ("Very long content paragraph. " * 100)
        chunks = _split_markdown_semantic(long_text, max_chunk_chars=500, min_chunk_chars=10)
        
        for chunk in chunks:
            assert len(chunk["text"]) <= 500 + 50  # Allow some flexibility

    def test_heading_path_preserved(self):
        """Test that heading path is correctly captured."""
        chunks = _split_markdown_semantic(SAMPLE_MARKDOWN)
        
        # Find the Motivation chunk (under Introduction)
        motivation_chunks = [c for c in chunks if "motivation" in c["section"].lower()]
        if motivation_chunks:
            # Should have heading_path showing hierarchy
            assert "Introduction" in motivation_chunks[0]["heading_path"] or \
                   "introduction" in motivation_chunks[0]["heading_path"].lower()


class TestChunkMetadataGeneration:
    """Tests for _generate_chunk_metadata function."""

    @pytest.mark.asyncio
    async def test_fallback_without_provider(self):
        """Test fallback behavior when no LLM provider available."""
        result = await _generate_chunk_metadata(
            text="This is a test chunk about machine learning.",
            section="Introduction",
            provider=None,
            model=None,
            num_questions=3,
        )
        
        assert "summary" in result
        assert "hypothetical_questions" in result
        assert "keywords" in result
        assert len(result["hypothetical_questions"]) <= 3

    @pytest.mark.asyncio
    async def test_with_mock_provider(self):
        """Test LLM-based metadata generation with mock."""
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "summary": "This discusses machine learning approaches.",
            "hypothetical_questions": [
                "What machine learning methods are discussed?",
                "How does this relate to deep learning?",
                "What are the key findings?",
            ],
            "keywords": ["machine learning", "deep learning", "AI"],
        })
        mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)
        
        result = await _generate_chunk_metadata(
            text="This is a test chunk about machine learning and deep learning approaches.",
            section="Method",
            provider=mock_provider,
            model="test-model",
            num_questions=3,
        )
        
        assert result["summary"] == "This discusses machine learning approaches."
        assert len(result["hypothetical_questions"]) == 3
        assert "machine learning" in result["keywords"]

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        """Test fallback when LLM returns invalid JSON."""
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Not valid JSON at all"
        mock_provider.chat_with_retry = AsyncMock(return_value=mock_response)
        
        result = await _generate_chunk_metadata(
            text="Test content for fallback.",
            section="Results",
            provider=mock_provider,
            model="test-model",
            num_questions=3,
        )
        
        # Should have fallback values
        assert "summary" in result
        assert "hypothetical_questions" in result
        assert len(result["hypothetical_questions"]) >= 1


class TestPaperKnowledgeBaseWithChroma:
    """Tests for PaperKnowledgeBase Chroma integration."""

    @pytest.fixture
    def kb_config(self, tmp_path: Path):
        """Create test KB config."""
        return PaperKbConfig(
            enabled=True,
            embedding_model="",  # Will use hash embeddings
            num_hypothetical_questions=3,
            enable_hypothetical_retrieval=True,
            chroma_persist_dir=str(tmp_path / "chroma"),
        )

    @pytest.fixture
    def paper_kb(self, tmp_path: Path, kb_config: PaperKbConfig):
        """Create test PaperKnowledgeBase."""
        return PaperKnowledgeBase(tmp_path, kb_config)

    def test_chroma_initialization(self, paper_kb: PaperKnowledgeBase):
        """Test that Chroma collections are initialized."""
        assert paper_kb._chroma_client is not None
        assert paper_kb._summary_collection is not None
        assert paper_kb._question_collection is not None
        assert paper_kb._chunk_collection is not None

    @pytest.mark.asyncio
    async def test_upsert_semantic_chunks(self, paper_kb: PaperKnowledgeBase):
        """Test upserting semantic chunks to Chroma."""
        doc = {
            "paper_id": "test-paper-001",
            "title": "Test Paper",
            "url": "https://example.com/test.pdf",
            "source": "arxiv",
            "year": 2024,
        }
        
        semantic_chunks = [
            {
                "section": "Introduction",
                "heading_level": 1,
                "text": "This paper introduces a new method for document chunking.",
                "heading_path": "Introduction",
            },
            {
                "section": "Method",
                "heading_level": 1,
                "text": "We use Markdown headers to split documents into semantic chunks.",
                "heading_path": "Method",
            },
        ]
        
        chunk_metadata = [
            {
                "summary": "Introduces document chunking method.",
                "hypothetical_questions": [
                    "What is this paper about?",
                    "What problem does it solve?",
                    "What method is proposed?",
                ],
                "keywords": ["chunking", "document", "method"],
            },
            {
                "summary": "Describes Markdown-based splitting approach.",
                "hypothetical_questions": [
                    "How does the method work?",
                    "What is the chunking algorithm?",
                    "What format is used?",
                ],
                "keywords": ["markdown", "headers", "splitting"],
            },
        ]
        
        result = await paper_kb.upsert_semantic_chunks(doc, semantic_chunks, chunk_metadata)
        
        assert result["paper_id"] == "test-paper-001"
        assert result["chunk_count"] == 2
        assert result["question_count"] == 6  # 3 questions per chunk
        assert result["success"] is True
        
        # Verify Chroma collections have data
        assert paper_kb._chunk_collection.count() >= 2
        assert paper_kb._summary_collection.count() >= 2
        assert paper_kb._question_collection.count() >= 6

    @pytest.mark.asyncio
    async def test_retrieve_by_hypothetical_questions(self, paper_kb: PaperKnowledgeBase):
        """Test retrieval via hypothetical questions."""
        # First insert some test data
        doc = {
            "paper_id": "test-paper-002",
            "title": "Semantic Chunking Paper",
            "source": "arxiv",
            "year": 2024,
        }
        
        semantic_chunks = [
            {
                "section": "Method",
                "heading_level": 2,
                "text": "Our method uses Markdown headers to identify document boundaries and create semantic chunks.",
                "heading_path": "Method",
            },
        ]
        
        chunk_metadata = [
            {
                "summary": "Markdown-based semantic chunking method.",
                "hypothetical_questions": [
                    "How do you split documents into chunks?",
                    "What method is used for chunking?",
                    "How does Markdown header chunking work?",
                ],
                "keywords": ["markdown", "chunking", "semantic"],
            },
        ]
        
        await paper_kb.upsert_semantic_chunks(doc, semantic_chunks, chunk_metadata)
        
        # Now test retrieval with a similar question
        results = await paper_kb.retrieve_by_hypothetical_questions(
            query="How does the document chunking method work?",
            top_k=5,
            search_mode="hybrid",
        )
        
        assert len(results) >= 1
        assert "text" in results[0]
        assert "chunk_id" in results[0]
        assert "matched_by" in results[0]  # Should show whether matched by question or summary

    @pytest.mark.asyncio
    async def test_delete_paper_from_chroma(self, paper_kb: PaperKnowledgeBase):
        """Test deletion of paper data from Chroma."""
        # Insert test data
        doc = {
            "paper_id": "test-paper-003",
            "title": "Test Paper for Deletion",
            "source": "arxiv",
            "year": 2024,
        }
        
        semantic_chunks = [
            {
                "section": "content",
                "heading_level": 1,
                "text": "Test content for deletion test.",
                "heading_path": "content",
            },
        ]
        
        chunk_metadata = [{
            "summary": "Test summary.",
            "hypothetical_questions": ["Test question?"],
            "keywords": ["test"],
        }]
        
        await paper_kb.upsert_semantic_chunks(doc, semantic_chunks, chunk_metadata)
        
        # Verify data exists
        assert paper_kb._chunk_collection.count() >= 1
        
        # Delete
        paper_kb._delete_paper_from_chroma("test-paper-003")
        
        # Verify deletion
        remaining = paper_kb._chunk_collection.get(
            where={"paper_id": "test-paper-003"},
        ).get("ids", [])
        assert len(remaining) == 0


class TestKBRetrieveToolModes:
    """Tests for KBRetrieveTool retrieval modes."""

    @pytest.fixture
    def mock_kb(self):
        """Create mock PaperKnowledgeBase."""
        kb = MagicMock(spec=PaperKnowledgeBase)
        kb.config = PaperKbConfig(enable_hypothetical_retrieval=True)
        kb.retrieve = AsyncMock(return_value=[
            {"chunk_id": "test:0", "text": "Traditional result", "score": 0.8}
        ])
        kb.retrieve_by_hypothetical_questions = AsyncMock(return_value=[
            {"chunk_id": "test:0", "text": "Hypothetical result", "score": 0.9, "matched_by": "question"}
        ])
        return kb

    # Note: Full KBRetrieveTool tests would require more setup
    # These are placeholder tests for the retrieval mode logic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])