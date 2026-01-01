"""
Tests for SummarizeCompactionStrategy.

Module: memory-service/tests/test_summarization.py
"""

from typing import Any, Dict
from datetime import datetime
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from service.main import _call_code_executor_summarize, _group_artifacts_by_similarity
from service.models import ArtifactRecord, SafetyClass


@pytest.mark.asyncio
async def test_call_code_executor_summarize_success() -> None:
    """Test successful call to code executor summarize skill."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "result": {"summary": "This is a test summary"},
        "logs": [],
        "hash": "abc123",
    }

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        summary = await _call_code_executor_summarize(
            "This is a long text that needs summarization", summary_length="medium"
        )

        assert summary == "This is a test summary"
        mock_client.return_value.__aenter__.return_value.post.assert_called_once()


@pytest.mark.asyncio
async def test_call_code_executor_summarize_connection_error() -> None:
    """Test handling of connection error to code executor."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        summary = await _call_code_executor_summarize("Test text")

        assert summary is None


@pytest.mark.asyncio
async def test_call_code_executor_summarize_timeout() -> None:
    """Test handling of timeout when calling code executor."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.TimeoutException("Request timeout")
        )

        summary = await _call_code_executor_summarize("Test text")

        assert summary is None


@pytest.mark.asyncio
async def test_call_code_executor_summarize_non_200_response() -> None:
    """Test handling of non-200 response from code executor."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        summary = await _call_code_executor_summarize("Test text")

        assert summary is None


@pytest.mark.asyncio
async def test_group_artifacts_by_similarity() -> None:
    """Test grouping artifacts by semantic similarity."""
    # Create mock artifacts
    artifact_with_embedding = MagicMock()
    artifact_with_embedding.embedding_ref = "artifacts/123"
    artifact_with_embedding.id = "123"

    artifact_without_embedding = MagicMock()
    artifact_without_embedding.embedding_ref = None
    artifact_without_embedding.id = "456"

    artifacts = [artifact_with_embedding, artifact_without_embedding]

    # Mock the global adapters
    with patch("service.main.vector_adapter") as mock_vector, patch(
        "service.main.embedding_generator"
    ) as mock_embeddings:
        mock_vector.__bool__.return_value = True
        mock_embeddings.__bool__.return_value = True

        groups = await _group_artifacts_by_similarity(artifacts)

        # Should have 2 groups: with embeddings and without
        assert len(groups) == 2
        assert artifact_with_embedding in groups[0]
        assert artifact_without_embedding in groups[1]


@pytest.mark.asyncio
async def test_group_artifacts_by_similarity_no_adapters() -> None:
    """Test grouping artifacts when adapters are not available."""
    artifact1 = MagicMock()
    artifact2 = MagicMock()
    artifacts = [artifact1, artifact2]

    # Mock the global adapters as None
    with patch("service.main.vector_adapter", None), patch(
        "service.main.embedding_generator", None
    ):
        groups = await _group_artifacts_by_similarity(artifacts)

        # Should return all artifacts in a single group
        assert len(groups) == 1
        assert len(groups[0]) == 2


@pytest.mark.asyncio
async def test_summarization_integration() -> None:
    """
    Integration test for the complete summarization flow.

    This test verifies:
    1. Artifacts are grouped
    2. Text is extracted and combined
    3. Code executor is called
    4. Summary artifact is created
    5. Original artifacts are deleted
    6. Token counts are updated
    """
    # This would be a full integration test with mocked dependencies
    # Left as a placeholder for future implementation
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
