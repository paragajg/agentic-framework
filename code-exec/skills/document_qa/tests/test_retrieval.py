"""
Tests for Document Retrieval and Indexing.

Module: code-exec/skills/document_qa/tests/test_retrieval.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from ..pipelines.indexing import run_indexing, create_query_embedding, SimpleEmbedder


class TestIndexingPipeline:
    """Tests for the indexing pipeline."""

    def test_run_indexing_empty(self) -> None:
        """Test indexing with no documents."""
        result = run_indexing(documents=[])

        assert result["documents_written"] == 0
        assert "document_store" in result

    def test_run_indexing_single_document(self) -> None:
        """Test indexing a single document."""
        doc = Document(
            content="This is test content for indexing.",
            meta={"source_id": "src_001"},
        )

        result = run_indexing(documents=[doc])

        assert result["documents_written"] == 1
        assert result["statistics"]["total_documents"] == 1

    def test_run_indexing_multiple_documents(self) -> None:
        """Test indexing multiple documents."""
        docs = [
            Document(content=f"Content {i}", meta={"source_id": f"src_{i:03d}"})
            for i in range(5)
        ]

        result = run_indexing(documents=docs)

        assert result["documents_written"] == 5

    def test_run_indexing_with_existing_store(self) -> None:
        """Test indexing with pre-existing document store."""
        store = InMemoryDocumentStore()

        docs = [Document(content="Test content", meta={"source_id": "src_001"})]

        result = run_indexing(documents=docs, document_store=store)

        assert result["document_store"] is store


class TestSimpleEmbedder:
    """Tests for the fallback embedder."""

    def test_embedder_generates_embeddings(self) -> None:
        """Test that simple embedder generates embeddings."""
        embedder = SimpleEmbedder(dimension=384)

        docs = [
            Document(content="Test content 1"),
            Document(content="Test content 2"),
        ]

        result = embedder.run(documents=docs)

        assert "documents" in result
        assert len(result["documents"]) == 2

        for doc in result["documents"]:
            assert doc.embedding is not None
            assert len(doc.embedding) == 384

    def test_embedder_deterministic(self) -> None:
        """Test that embedder produces same output for same input."""
        embedder = SimpleEmbedder(dimension=384)

        doc1 = Document(content="Same content")
        doc2 = Document(content="Same content")

        result1 = embedder.run(documents=[doc1])
        result2 = embedder.run(documents=[doc2])

        # Same content should produce same embedding
        assert result1["documents"][0].embedding == result2["documents"][0].embedding


class TestQueryEmbedding:
    """Tests for query embedding generation."""

    def test_create_query_embedding_fallback(self) -> None:
        """Test query embedding generation.

        Uses the configured embedding model from .env (OPENAI_EMBEDDING_MODEL)
        or falls back to sentence-transformers.
        """
        embedding = create_query_embedding("test query")

        assert isinstance(embedding, list)
        # Embedding dimension depends on configured model:
        # - text-embedding-3-large: 3072
        # - text-embedding-3-small: 1536
        # - sentence-transformers/all-MiniLM-L6-v2: 384
        # - fallback (hash-based): 384
        assert len(embedding) > 0, "Embedding should not be empty"
        assert all(isinstance(x, (int, float)) for x in embedding), "All values should be numeric"

    def test_query_embedding_deterministic(self) -> None:
        """Test that query embedding is deterministic."""
        query = "What is the revenue?"

        embedding1 = create_query_embedding(query)
        embedding2 = create_query_embedding(query)

        assert embedding1 == embedding2


class TestDocumentStore:
    """Tests for document store operations."""

    def test_document_store_write_and_read(self) -> None:
        """Test writing and reading documents."""
        store = InMemoryDocumentStore()

        docs = [
            Document(
                content="Revenue was $1M",
                embedding=[0.1] * 384,
                meta={"source_id": "src_001"},
            ),
            Document(
                content="Expenses were $500K",
                embedding=[0.2] * 384,
                meta={"source_id": "src_002"},
            ),
        ]

        store.write_documents(docs)

        # Count documents
        assert store.count_documents() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
