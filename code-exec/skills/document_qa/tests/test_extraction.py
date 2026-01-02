"""
Tests for Document Extraction Components.

Module: code-exec/skills/document_qa/tests/test_extraction.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from haystack import Document

from ..utils.source_tracker import SourceTracker, SourceType, SourceEntry
from ..utils.image_filter import ImageFilter, ImageData
from ..components.semantic_chunker import SemanticChunker, chunk_text


class TestSourceTracker:
    """Tests for SourceTracker class."""

    def test_add_source_basic(self) -> None:
        """Test adding a basic source."""
        tracker = SourceTracker()
        source_id = tracker.add_source(
            source_type=SourceType.DOCUMENT,
            file_path="report.pdf",
            page=3,
        )

        assert source_id == "src_001"
        assert tracker.count == 1

        source = tracker.get_source(source_id)
        assert source is not None
        assert source.file_path == "report.pdf"
        assert source.page == 3

    def test_add_source_with_section(self) -> None:
        """Test adding a source with section."""
        tracker = SourceTracker()
        source_id = tracker.add_source(
            source_type=SourceType.DOCUMENT,
            file_path="report.pdf",
            page=5,
            section="Revenue Analysis",
        )

        source = tracker.get_source(source_id)
        assert source.section == "Revenue Analysis"

    def test_source_deduplication(self) -> None:
        """Test that duplicate content gets the same source ID."""
        tracker = SourceTracker()

        content = "This is the same content"

        id1 = tracker.add_source(
            source_type=SourceType.DOCUMENT,
            file_path="doc1.pdf",
            content=content,
        )

        id2 = tracker.add_source(
            source_type=SourceType.DOCUMENT,
            file_path="doc2.pdf",
            content=content,
        )

        # Same content should return same ID
        assert id1 == id2
        assert tracker.count == 1

    def test_to_citation(self) -> None:
        """Test citation formatting."""
        tracker = SourceTracker()
        tracker.add_source(
            source_type=SourceType.DOCUMENT,
            file_path="report.pdf",
            page=3,
            section="Revenue Analysis",
        )

        source = tracker.get_source("src_001")
        citation = source.to_citation()

        assert "report.pdf" in citation
        assert "Page 3" in citation
        assert "Revenue Analysis" in citation

    def test_format_citations_block(self) -> None:
        """Test formatting all citations as a block."""
        tracker = SourceTracker()
        tracker.add_source(
            source_type=SourceType.DOCUMENT,
            file_path="report.pdf",
            page=3,
        )
        tracker.add_source(
            source_type=SourceType.TABLE,
            file_path="data.xlsx",
            sheet="Revenue",
        )

        block = tracker.format_citations()
        assert "---" in block
        assert "Sources:" in block
        assert "[src_001]" in block
        assert "[src_002]" in block

    def test_clear(self) -> None:
        """Test clearing all sources."""
        tracker = SourceTracker()
        tracker.add_source(source_type=SourceType.DOCUMENT, file_path="test.pdf")
        tracker.add_source(source_type=SourceType.DOCUMENT, file_path="test2.pdf")

        assert tracker.count == 2
        tracker.clear()
        assert tracker.count == 0


class TestImageFilter:
    """Tests for ImageFilter class."""

    def test_filter_small_image(self) -> None:
        """Test that small images are filtered out."""
        filter_ = ImageFilter(min_size_kb=50)

        # Small image (10KB)
        small_image = ImageData(
            image_bytes=b"x" * 10240,  # 10KB
            file_path="icon.png",
            source_file="doc.pptx",
            image_format="png",
        )

        should_process, reason = filter_.should_process(small_image)
        assert not should_process
        assert "too_small" in reason

    def test_filter_accepts_large_image(self) -> None:
        """Test that large images are accepted."""
        filter_ = ImageFilter(min_size_kb=50)

        # Large image (100KB)
        large_image = ImageData(
            image_bytes=b"x" * 102400,  # 100KB
            file_path="chart.png",
            source_file="doc.pptx",
            image_format="png",
            size=(800, 600),
        )

        should_process, reason = filter_.should_process(large_image)
        assert should_process
        assert reason == "acceptable"

    def test_filter_narrow_image(self) -> None:
        """Test that very narrow images are filtered."""
        filter_ = ImageFilter(min_size_kb=10)

        narrow_image = ImageData(
            image_bytes=b"x" * 51200,  # 50KB
            file_path="banner.png",
            source_file="doc.pptx",
            image_format="png",
            size=(1000, 50),  # Very narrow
        )

        should_process, reason = filter_.should_process(narrow_image)
        assert not should_process
        assert "too_wide" in reason

    def test_filter_skip_patterns(self) -> None:
        """Test that images with skip patterns in alt text are filtered."""
        filter_ = ImageFilter(min_size_kb=10)

        logo_image = ImageData(
            image_bytes=b"x" * 51200,
            file_path="company_logo.png",
            source_file="doc.pptx",
            image_format="png",
            size=(200, 200),
            alt_text="Company Logo",
        )

        should_process, reason = filter_.should_process(logo_image)
        assert not should_process
        assert "matches_skip_pattern" in reason

    def test_filter_duplicates(self) -> None:
        """Test that duplicate images are filtered."""
        filter_ = ImageFilter(min_size_kb=10)

        image1 = ImageData(
            image_bytes=b"same content" * 1000,
            file_path="image1.png",
            source_file="doc.pptx",
            image_format="png",
            size=(400, 300),
        )

        image2 = ImageData(
            image_bytes=b"same content" * 1000,
            file_path="image2.png",
            source_file="doc.pptx",
            image_format="png",
            size=(400, 300),
        )

        # First should pass
        should_process1, _ = filter_.should_process(image1)
        assert should_process1

        # Second (duplicate) should fail
        should_process2, reason = filter_.should_process(image2)
        assert not should_process2
        assert "duplicate" in reason

    def test_filter_images_list(self) -> None:
        """Test filtering a list of images."""
        filter_ = ImageFilter(min_size_kb=10, max_images=2)

        # Each image needs unique bytes to avoid duplicate detection
        images = [
            ImageData(
                image_bytes=f"unique_content_{i}_".encode() * 5120,  # ~50KB each, unique
                file_path=f"image{i}.png",
                source_file="doc.pptx",
                image_format="png",
                size=(400, 300),
            )
            for i in range(5)
        ]

        filtered = filter_.filter_images(images)
        assert len(filtered) == 2  # max_images limit


class TestSemanticChunker:
    """Tests for SemanticChunker class."""

    def test_chunk_short_document(self) -> None:
        """Test that short documents create single chunk."""
        chunker = SemanticChunker(chunk_size=1000)

        doc = Document(
            content="This is a short document.",
            meta={"file_name": "test.pdf"},
        )

        result = chunker.run([doc])
        chunks = result["chunks"]

        assert len(chunks) == 1
        assert "short document" in chunks[0].content

    def test_chunk_preserves_metadata(self) -> None:
        """Test that metadata is preserved in chunks."""
        chunker = SemanticChunker(chunk_size=100)

        doc = Document(
            content="This is content. " * 50,
            meta={"file_name": "test.pdf", "page": 1},
        )

        result = chunker.run([doc])
        chunks = result["chunks"]

        for chunk in chunks:
            assert chunk.meta["file_name"] == "test.pdf"

    def test_chunk_by_headings(self) -> None:
        """Test chunking by heading boundaries."""
        chunker = SemanticChunker(chunk_size=500)

        content = """# Introduction

This is the introduction section with some content.

# Methods

This is the methods section with different content.

# Results

This is the results section."""

        doc = Document(content=content, meta={"file_name": "paper.pdf"})
        result = chunker.run([doc])
        chunks = result["chunks"]

        # Should have chunks for each section
        assert len(chunks) >= 3

    def test_chunk_text_convenience(self) -> None:
        """Test the chunk_text convenience function."""
        text = "This is sentence one. This is sentence two. This is sentence three."

        chunks = chunk_text(text, chunk_size=50, overlap=10)

        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)


class TestMarkItDownConverter:
    """Tests for MarkItDownConverter component."""

    def test_converter_initialization(self) -> None:
        """Test converter initializes correctly."""
        from ..components.markitdown_converter import MarkItDownConverter

        converter = MarkItDownConverter(extract_images=True)
        assert converter.extract_images is True

    def test_supported_extensions(self) -> None:
        """Test supported file extensions."""
        from ..components.markitdown_converter import MarkItDownConverter

        converter = MarkItDownConverter()

        assert ".pdf" in converter.SUPPORTED_EXTENSIONS
        assert ".docx" in converter.SUPPORTED_EXTENSIONS
        assert ".xlsx" in converter.SUPPORTED_EXTENSIONS
        assert ".pptx" in converter.SUPPORTED_EXTENSIONS
        assert ".html" in converter.SUPPORTED_EXTENSIONS
        assert ".csv" in converter.SUPPORTED_EXTENSIONS


class TestContextAssembler:
    """Tests for ContextAssembler component."""

    def test_assemble_single_document(self) -> None:
        """Test assembling context from single document."""
        from ..components.context_assembler import ContextAssembler

        assembler = ContextAssembler(max_tokens=1000)

        doc = Document(
            content="This is the content of the document.",
            meta={"source_id": "src_001", "file_name": "test.pdf"},
        )

        result = assembler.run([doc])

        assert "context" in result
        assert "sources" in result
        assert "[src_001]" in result["context"]

    def test_position_strategy(self) -> None:
        """Test edge positioning strategy."""
        from ..components.context_assembler import ContextAssembler

        assembler = ContextAssembler(max_tokens=10000, position_strategy="edges")

        docs = [
            Document(content=f"Document {i} content.", meta={"source_id": f"src_{i:03d}"})
            for i in range(1, 6)
        ]

        result = assembler.run(docs)
        context = result["context"]

        # First doc should be at the start
        assert context.startswith("[src_001]")


class TestLLMReranker:
    """Tests for LLMReranker component."""

    def test_reranker_initialization(self) -> None:
        """Test reranker initializes correctly."""
        from ..components.llm_reranker import LLMReranker

        reranker = LLMReranker(top_k=5, min_score=3.0)
        assert reranker.top_k == 5
        assert reranker.min_score == 3.0

    def test_parse_score_response(self) -> None:
        """Test parsing score response."""
        from ..components.llm_reranker import LLMReranker

        reranker = LLMReranker()

        response = '{"score": 8, "reason": "Highly relevant"}'
        score, reason = reranker._parse_score_response(response)

        assert score == 8.0
        assert "Highly relevant" in reason

    def test_parse_score_fallback(self) -> None:
        """Test parsing score when JSON is invalid."""
        from ..components.llm_reranker import LLMReranker

        reranker = LLMReranker()

        response = "The relevance score is 7 out of 10."
        score, reason = reranker._parse_score_response(response)

        assert score == 7.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
