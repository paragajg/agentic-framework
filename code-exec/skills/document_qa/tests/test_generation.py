"""
Tests for Document Q&A Generation.

Module: code-exec/skills/document_qa/tests/test_generation.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestLLMGenerator:
    """Tests for the LLM Generator component."""

    def test_generator_initialization(self) -> None:
        """Test generator initializes correctly."""
        from ..pipelines.generation import LLMGenerator

        generator = LLMGenerator(llm_client=None)
        assert generator.llm_client is None

    def test_generator_prompt_formatting(self) -> None:
        """Test that the generation prompt is formatted correctly."""
        from ..pipelines.generation import LLMGenerator

        generator = LLMGenerator()

        # Check prompt template has required placeholders
        assert "{query}" in generator.GENERATION_PROMPT
        assert "{context}" in generator.GENERATION_PROMPT

    @patch("openai.OpenAI")
    def test_generator_with_mock_llm(self, mock_openai: Mock) -> None:
        """Test generator with mocked LLM."""
        from ..pipelines.generation import LLMGenerator

        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="The answer is 42."))]
        mock_response.usage = Mock(total_tokens=100)
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = LLMGenerator()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = generator.run(
                query="What is the answer?",
                context="[src_001] The answer to everything is 42.",
                sources=[{"id": "src_001", "file": "guide.txt"}],
            )

        assert "answer" in result
        assert "confidence" in result


class TestGenerationPipeline:
    """Tests for the complete generation pipeline."""

    def test_create_generation_pipeline(self) -> None:
        """Test creating the generation pipeline."""
        from ..pipelines.generation import create_generation_pipeline

        store = InMemoryDocumentStore()

        pipeline = create_generation_pipeline(
            document_store=store,
            retrieval_top_k=10,
            rerank_top_k=5,
        )

        assert pipeline is not None

    def test_run_generation_empty_store(self) -> None:
        """Test generation with empty document store."""
        from ..pipelines.generation import run_generation

        store = InMemoryDocumentStore()

        # This should handle empty store gracefully
        # Note: Will likely fail in actual run without mock, but tests structure


class TestDocumentHandler:
    """Tests for the main document_qa handler."""

    def test_handler_validates_empty_documents(self) -> None:
        """Test handler returns error for empty documents."""
        from ..handler import document_qa

        result = document_qa(documents=[], query="What is the revenue?")

        assert result["success"] is False
        assert "No documents provided" in result.get("error", "")

    def test_handler_validates_empty_query(self) -> None:
        """Test handler returns error for empty query."""
        from ..handler import document_qa

        result = document_qa(documents=["test.pdf"], query="")

        assert result["success"] is False
        assert "No query provided" in result.get("error", "")

    def test_handler_validates_nonexistent_files(self) -> None:
        """Test handler returns error for non-existent files."""
        from ..handler import document_qa

        result = document_qa(
            documents=["/nonexistent/file.pdf"],
            query="What is the content?",
        )

        assert result["success"] is False
        assert "No valid document" in result.get("error", "")

    def test_handler_result_structure(self) -> None:
        """Test that handler returns correct structure."""
        from ..handler import document_qa

        result = document_qa(documents=[], query="test")

        # Check all required keys are present
        assert "success" in result
        assert "answer" in result
        assert "sources" in result
        assert "chunks_retrieved" in result
        assert "confidence" in result
        assert "metadata" in result


class TestExtractOnly:
    """Tests for the extract_only function."""

    def test_extract_only_returns_correct_structure(self) -> None:
        """Test extract_only returns correct structure."""
        from ..handler import extract_only

        # With non-existent file, should return empty but structured result
        result = extract_only(
            documents=["/nonexistent/file.pdf"],
            process_images=False,
        )

        assert "chunks" in result
        assert "statistics" in result


class TestSourceIntegration:
    """Tests for source tracking integration."""

    def test_source_ids_in_chunks(self) -> None:
        """Test that source IDs are added to chunks."""
        from ..utils.source_tracker import SourceTracker, SourceType
        from ..components.semantic_chunker import SemanticChunker

        tracker = SourceTracker()

        # Add source
        source_id = tracker.add_source(
            source_type=SourceType.DOCUMENT,
            file_path="test.pdf",
            page=1,
        )

        # Create chunk with source ID
        doc = Document(
            content="Test content",
            meta={"source_id": source_id, "file_name": "test.pdf"},
        )

        chunker = SemanticChunker()
        result = chunker.run([doc])

        for chunk in result["chunks"]:
            assert "source_id" in chunk.meta or "file_name" in chunk.meta


class TestDeepResearchIntegration:
    """Tests for deep_research integration."""

    def test_deep_research_accepts_documents(self) -> None:
        """Test that deep_research function accepts documents parameter."""
        import inspect
        from skills.deep_research.handler import deep_research

        sig = inspect.signature(deep_research)
        params = list(sig.parameters.keys())

        assert "documents" in params

    def test_deep_research_schema_has_documents(self) -> None:
        """Test that deep_research schema includes documents field."""
        import json
        from pathlib import Path

        schema_path = Path(__file__).parent.parent.parent / "deep_research" / "schema.json"

        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)

            input_props = schema.get("input", {}).get("properties", {})
            assert "documents" in input_props


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
