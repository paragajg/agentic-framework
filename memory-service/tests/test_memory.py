"""
Memory Service Tests.

Module: memory-service/tests/test_memory.py
"""

from typing import Any, Dict
from datetime import datetime
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memory_service.service.models import (
    Artifact,
    ArtifactType,
    SafetyClass,
    CommitRequest,
    QueryRequest,
    CompactionRequest,
    CompactionStrategy,
)
from memory_service.service.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(
        redis_url="redis://localhost:6379/0",
        postgres_url="postgresql://test:test@localhost:5432/test",
        vector_db_type="chroma",
        chroma_path="./test_data/chroma",
        minio_endpoint="localhost:9000",
        minio_bucket="test-artifacts",
    )


@pytest.fixture
def sample_artifact() -> Artifact:
    """Create sample artifact for testing."""
    return Artifact(
        artifact_type=ArtifactType.RESEARCH_SNIPPET,
        content={
            "text": "Sample research finding about agent frameworks",
            "summary": "Research on agent frameworks",
            "source": {"url": "https://example.com/research"},
        },
        safety_class=SafetyClass.INTERNAL,
        created_by="test-agent",
        session_id="test-session-123",
        tags=["research", "agents"],
    )


@pytest.fixture
def sample_commit_request(sample_artifact: Artifact) -> CommitRequest:
    """Create sample commit request."""
    return CommitRequest(
        artifact=sample_artifact,
        actor_id="test-actor",
        actor_type="subagent",
        tool_ids=["web_search", "summarize"],
        generate_embedding=True,
        store_in_cold=False,
    )


class TestRedisAdapter:
    """Test Redis storage adapter."""

    @pytest.mark.asyncio
    async def test_session_data_storage(self, test_settings: Settings) -> None:
        """Test storing and retrieving session data."""
        from memory_service.service.storage.redis import RedisAdapter

        adapter = RedisAdapter(test_settings)

        # Mock Redis client
        mock_client = AsyncMock()
        adapter.client = mock_client

        session_data = {"user_id": "test-user", "context": "testing"}

        # Test set
        mock_client.setex = AsyncMock(return_value=True)
        result = await adapter.set_session_data("session-123", session_data)
        assert result is True

        # Test get
        mock_client.get = AsyncMock(return_value='{"user_id": "test-user", "context": "testing"}')
        retrieved = await adapter.get_session_data("session-123")
        assert retrieved == session_data

    @pytest.mark.asyncio
    async def test_artifact_caching(self, test_settings: Settings) -> None:
        """Test artifact caching."""
        from memory_service.service.storage.redis import RedisAdapter

        adapter = RedisAdapter(test_settings)
        mock_client = AsyncMock()
        adapter.client = mock_client

        artifact_content = {"text": "test content"}

        mock_client.setex = AsyncMock(return_value=True)
        result = await adapter.cache_artifact("artifact-123", artifact_content)
        assert result is True


class TestPostgresAdapter:
    """Test Postgres storage adapter."""

    @pytest.mark.asyncio
    async def test_create_artifact_record(self, test_settings: Settings, sample_artifact: Artifact) -> None:
        """Test creating artifact record."""
        from memory_service.service.storage.postgres import PostgresAdapter

        # This would require actual database setup in integration tests
        # For unit tests, we mock the database operations
        adapter = PostgresAdapter(test_settings)

        # Mock the engine and session
        with patch("memory_service.service.storage.postgres.AsyncSession"):
            artifact_id = await adapter.create_artifact_record(
                artifact=sample_artifact,
                content_hash="abc123",
                embedding_ref="embeddings/artifact-123",
                token_count=100,
            )
            assert artifact_id is not None

    @pytest.mark.asyncio
    async def test_provenance_logging(self, test_settings: Settings) -> None:
        """Test provenance log creation."""
        from memory_service.service.storage.postgres import PostgresAdapter

        adapter = PostgresAdapter(test_settings)

        with patch("memory_service.service.storage.postgres.AsyncSession"):
            provenance_id = await adapter.create_provenance_log(
                artifact_id="artifact-123",
                actor_id="actor-1",
                actor_type="subagent",
                inputs_hash="input-hash",
                outputs_hash="output-hash",
                tool_ids=["tool1", "tool2"],
                parent_artifact_ids=["parent-1"],
            )
            assert provenance_id is not None


class TestEmbeddingGenerator:
    """Test embedding generation."""

    @pytest.mark.asyncio
    async def test_extract_searchable_text(self, test_settings: Settings) -> None:
        """Test extracting searchable text from artifact."""
        from memory_service.service.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator(test_settings)

        content = {
            "text": "Main content here",
            "summary": "Brief summary",
            "tags": ["tag1", "tag2"],
            "source": {"url": "https://example.com"},
        }

        searchable = generator.extract_searchable_text(content)
        assert "Main content here" in searchable
        assert "Brief summary" in searchable
        assert "tag1" in searchable

    @pytest.mark.asyncio
    async def test_token_counting(self, test_settings: Settings) -> None:
        """Test token counting."""
        from memory_service.service.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator(test_settings)

        text = "This is a test sentence with multiple words."
        token_count = generator.count_tokens(text)
        assert token_count > 0


class TestTokenBudgetManager:
    """Test token budget management."""

    def test_needs_compaction(self, test_settings: Settings) -> None:
        """Test compaction detection."""
        from memory_service.service.embedding import TokenBudgetManager

        manager = TokenBudgetManager(test_settings)

        # Below threshold
        assert not manager.needs_compaction(5000)

        # Above threshold
        assert manager.needs_compaction(10000)

    def test_calculate_target_tokens(self, test_settings: Settings) -> None:
        """Test target token calculation."""
        from memory_service.service.embedding import TokenBudgetManager

        manager = TokenBudgetManager(test_settings)

        target_summarize = manager.calculate_target_tokens("summarize")
        assert target_summarize < test_settings.memory_compaction_threshold_tokens

        target_truncate = manager.calculate_target_tokens("truncate")
        assert target_truncate < target_summarize

    def test_prioritize_artifacts(self, test_settings: Settings) -> None:
        """Test artifact prioritization."""
        from memory_service.service.embedding import TokenBudgetManager

        manager = TokenBudgetManager(test_settings)

        artifacts = [
            {"id": "1", "created_at": "2024-01-01", "confidence": 0.9},
            {"id": "2", "created_at": "2024-01-02", "confidence": 0.5},
            {"id": "3", "created_at": "2024-01-03", "confidence": 0.8},
        ]

        prioritized = manager.prioritize_artifacts(artifacts, preserve_ids=["1"])

        # First artifact should be preserved
        assert prioritized[0]["id"] == "1"


class TestMemoryServiceAPI:
    """Test FastAPI endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        from memory_service.service.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_commit_artifact_validation(self, sample_commit_request: CommitRequest) -> None:
        """Test commit request validation."""
        # Ensure required fields are present
        assert sample_commit_request.artifact.content is not None
        assert sample_commit_request.actor_id is not None
        assert sample_commit_request.actor_type is not None

    @pytest.mark.asyncio
    async def test_query_request_validation(self) -> None:
        """Test query request validation."""
        # Valid with query_text
        query1 = QueryRequest(query_text="test query", top_k=5)
        assert query1.query_text == "test query"

        # Valid with query_embedding
        query2 = QueryRequest(query_embedding=[0.1, 0.2, 0.3], top_k=5)
        assert len(query2.query_embedding) == 3

    @pytest.mark.asyncio
    async def test_compaction_request_validation(self) -> None:
        """Test compaction request validation."""
        request = CompactionRequest(
            session_id="test-session",
            strategy=CompactionStrategy.SUMMARIZE,
            target_tokens=5000,
            preserve_artifact_ids=["artifact-1"],
        )
        assert request.strategy == CompactionStrategy.SUMMARIZE
        assert request.target_tokens == 5000


# Integration test markers
@pytest.mark.integration
class TestMemoryServiceIntegration:
    """Integration tests requiring actual services."""

    @pytest.mark.asyncio
    async def test_end_to_end_commit_and_query(self) -> None:
        """Test committing and querying artifacts end-to-end."""
        # This would require actual Redis, Postgres, and vector DB running
        # Skipped in unit tests, run separately in CI/CD
        pytest.skip("Integration test - requires running services")

    @pytest.mark.asyncio
    async def test_provenance_chain_tracking(self) -> None:
        """Test complete provenance chain tracking."""
        pytest.skip("Integration test - requires running services")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
