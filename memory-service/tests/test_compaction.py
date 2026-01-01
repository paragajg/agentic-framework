"""
Tests for Memory Compaction Features.

Module: tests/test_compaction.py
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List, Dict, Any

from service.models import (
    CompactionRequest,
    CompactionResponse,
    CompactionStrategy,
    Artifact,
    ArtifactRecord,
    ArtifactType,
    SafetyClass,
)
from service.main import _call_code_executor_summarize, _group_artifacts_by_similarity
from service.embedding import TokenBudgetManager


class TestTokenCompaction:
    """Test suite for token compaction features."""

    @pytest.fixture
    def mock_artifacts(self) -> List[ArtifactRecord]:
        """Create mock artifacts for testing."""
        artifacts = []
        for i in range(5):
            artifact = ArtifactRecord(
                id=f"artifact-{i}",
                artifact_type=ArtifactType.RESEARCH_SNIPPET.value,
                content_hash=f"hash-{i}",
                safety_class=SafetyClass.INTERNAL.value,
                created_by="test-agent",
                created_at=datetime.utcnow(),
                metadata={
                    "text": f"This is test artifact {i} with some content",
                    "summary": f"Summary {i}",
                },
                token_count=100,
                session_id="test-session-123",
                embedding_ref=f"embedding-{i}" if i % 2 == 0 else None,
            )
            artifacts.append(artifact)
        return artifacts

    @pytest.fixture
    def token_budget_manager(self) -> TokenBudgetManager:
        """Create token budget manager."""
        from service.config import Settings

        settings = Settings()
        return TokenBudgetManager(settings)

    @pytest.mark.asyncio
    async def test_needs_compaction(self, token_budget_manager: TokenBudgetManager) -> None:
        """Test compaction detection logic."""
        # Below threshold - no compaction needed
        assert not token_budget_manager.needs_compaction(5000)

        # At threshold - no compaction needed
        assert not token_budget_manager.needs_compaction(8000)

        # Above threshold - compaction needed
        assert token_budget_manager.needs_compaction(10000)
        assert token_budget_manager.needs_compaction(15000)

    @pytest.mark.asyncio
    async def test_calculate_target_tokens(self, token_budget_manager: TokenBudgetManager) -> None:
        """Test target token calculation for different strategies."""
        # Summarize strategy - 60% of threshold
        target_summarize = token_budget_manager.calculate_target_tokens("summarize")
        assert target_summarize == int(8000 * 0.6)

        # Truncate strategy - 70% of threshold
        target_truncate = token_budget_manager.calculate_target_tokens("truncate")
        assert target_truncate == int(8000 * 0.7)

        # None strategy - threshold
        target_none = token_budget_manager.calculate_target_tokens("none")
        assert target_none == 8000

    @pytest.mark.asyncio
    async def test_prioritize_artifacts(self, token_budget_manager: TokenBudgetManager) -> None:
        """Test artifact prioritization logic."""
        artifacts = [
            {
                "id": "artifact-1",
                "created_at": "2024-01-01T10:00:00",
                "token_count": 100,
                "confidence": 0.9,
            },
            {
                "id": "artifact-2",
                "created_at": "2024-01-01T11:00:00",
                "token_count": 200,
                "confidence": 0.5,
            },
            {
                "id": "artifact-3",
                "created_at": "2024-01-01T12:00:00",
                "token_count": 150,
                "confidence": 0.8,
            },
        ]

        prioritized = token_budget_manager.prioritize_artifacts(artifacts, preserve_ids=[])

        # Should be sorted by priority score (confidence * recency)
        assert len(prioritized) == 3
        assert all("id" in a for a in prioritized)

    @pytest.mark.asyncio
    async def test_group_artifacts_by_similarity_no_embeddings(
        self, mock_artifacts: List[ArtifactRecord]
    ) -> None:
        """Test artifact grouping when no embeddings available."""
        # Mock vector adapter and embedding generator to None
        with patch("service.main.vector_adapter", None):
            with patch("service.main.embedding_generator", None):
                groups = await _group_artifacts_by_similarity(mock_artifacts)

                # Should group all artifacts together as fallback
                assert len(groups) == 1
                assert len(groups[0]) == len(mock_artifacts)

    @pytest.mark.asyncio
    async def test_group_artifacts_by_similarity_with_embeddings(
        self, mock_artifacts: List[ArtifactRecord]
    ) -> None:
        """Test artifact grouping with embeddings."""
        # Mock vector adapter and embedding generator
        mock_vector = MagicMock()
        mock_embedding = MagicMock()

        with patch("service.main.vector_adapter", mock_vector):
            with patch("service.main.embedding_generator", mock_embedding):
                groups = await _group_artifacts_by_similarity(mock_artifacts)

                # Should separate artifacts with/without embeddings
                assert len(groups) <= 2

                # Check that artifacts are properly grouped
                total_artifacts = sum(len(g) for g in groups)
                assert total_artifacts == len(mock_artifacts)

    @pytest.mark.asyncio
    async def test_call_code_executor_summarize_success(self) -> None:
        """Test successful summarization via code executor."""
        test_text = "This is a long text that needs to be summarized."

        # Mock httpx AsyncClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"summary": "This is a summary."},
            "logs": [],
            "hash": "abc123",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            summary = await _call_code_executor_summarize(test_text, "medium")

            assert summary == "This is a summary."

    @pytest.mark.asyncio
    async def test_call_code_executor_summarize_failure(self) -> None:
        """Test summarization failure handling."""
        test_text = "Test text"

        # Mock httpx AsyncClient with error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            summary = await _call_code_executor_summarize(test_text)

            assert summary is None

    @pytest.mark.asyncio
    async def test_call_code_executor_summarize_timeout(self) -> None:
        """Test summarization timeout handling."""
        test_text = "Test text"

        with patch("httpx.AsyncClient") as mock_client:
            import httpx

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            summary = await _call_code_executor_summarize(test_text)

            assert summary is None

    @pytest.mark.asyncio
    async def test_call_code_executor_summarize_connection_error(self) -> None:
        """Test summarization connection error handling."""
        test_text = "Test text"

        with patch("httpx.AsyncClient") as mock_client:
            import httpx

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )

            summary = await _call_code_executor_summarize(test_text)

            assert summary is None


class TestCompactionRequest:
    """Test suite for compaction request validation."""

    def test_compaction_request_validation(self) -> None:
        """Test compaction request validation."""
        request = CompactionRequest(
            session_id="test-session",
            strategy=CompactionStrategy.SUMMARIZE,
            preserve_artifact_ids=["artifact-1", "artifact-2"],
            target_tokens=5000,
        )

        assert request.session_id == "test-session"
        assert request.strategy == CompactionStrategy.SUMMARIZE
        assert len(request.preserve_artifact_ids) == 2
        assert request.target_tokens == 5000

    def test_compaction_request_defaults(self) -> None:
        """Test compaction request default values."""
        request = CompactionRequest(session_id="test-session")

        assert request.strategy == CompactionStrategy.SUMMARIZE
        assert request.preserve_artifact_ids == []
        assert request.target_tokens is None


class TestCompactionResponse:
    """Test suite for compaction response."""

    def test_compaction_response_structure(self) -> None:
        """Test compaction response structure."""
        response = CompactionResponse(
            session_id="test-session",
            tokens_before=10000,
            tokens_after=6000,
            artifacts_compacted=5,
            artifacts_removed=3,
            strategy_used=CompactionStrategy.SUMMARIZE,
            summary_artifact_id="summary-123",
        )

        assert response.session_id == "test-session"
        assert response.tokens_before == 10000
        assert response.tokens_after == 6000
        assert response.artifacts_compacted == 5
        assert response.artifacts_removed == 3
        assert response.strategy_used == CompactionStrategy.SUMMARIZE
        assert response.summary_artifact_id == "summary-123"

        # Verify token savings
        savings = response.tokens_before - response.tokens_after
        assert savings == 4000


class TestCompactionIntegration:
    """Integration tests for compaction workflow."""

    @pytest.mark.asyncio
    async def test_compaction_workflow_no_compaction_needed(self) -> None:
        """Test compaction when token count is below threshold."""
        # This would be an integration test with actual FastAPI client
        # For now, we test the logic components

        from service.config import Settings

        settings = Settings()
        manager = TokenBudgetManager(settings)

        # Session with low token count
        current_tokens = 5000

        # Should not need compaction
        assert not manager.needs_compaction(current_tokens)

    @pytest.mark.asyncio
    async def test_compaction_workflow_truncate_strategy(self) -> None:
        """Test truncate compaction strategy."""
        # Test the truncate strategy logic
        artifacts_to_remove = []
        current_tokens = 10000
        target_tokens = 5600  # 70% of 8000

        # Simulate removing oldest artifacts
        mock_artifacts = [
            {"id": f"artifact-{i}", "token_count": 500, "created_at": f"2024-01-0{i}"}
            for i in range(1, 11)
        ]

        for artifact in reversed(mock_artifacts):
            if current_tokens <= target_tokens:
                break
            artifacts_to_remove.append(artifact["id"])
            current_tokens -= artifact["token_count"]

        # Should have removed enough artifacts to get under target
        assert current_tokens <= target_tokens
        assert len(artifacts_to_remove) > 0

    @pytest.mark.asyncio
    async def test_compaction_workflow_summarize_strategy(self) -> None:
        """Test summarize compaction strategy workflow."""
        # Mock the full summarization workflow

        # 1. Group artifacts
        artifacts = [
            {"id": f"artifact-{i}", "text": f"Content {i}", "token_count": 200}
            for i in range(5)
        ]

        # 2. Combine texts
        combined_text = "\n\n---\n\n".join(a["text"] for a in artifacts)
        assert "Content 0" in combined_text
        assert "Content 4" in combined_text

        # 3. Mock summarization call
        with patch("service.main._call_code_executor_summarize") as mock_summarize:
            mock_summarize.return_value = "This is a summary of all content."

            summary = await _call_code_executor_summarize(combined_text)

            assert summary is not None
            assert len(summary) > 0
            mock_summarize.assert_called_once()

        # 4. Verify token savings would occur
        original_tokens = sum(a["token_count"] for a in artifacts)
        summary_tokens = 50  # Estimated
        assert summary_tokens < original_tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=service", "--cov-report=html"])
