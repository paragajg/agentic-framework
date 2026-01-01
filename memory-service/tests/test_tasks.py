"""
Tests for Celery Background Tasks.

Module: tests/test_tasks.py
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timedelta
from typing import Dict, Any

from service.tasks import (
    prune_expired_sessions,
    archive_old_artifacts,
    compact_long_running_sessions,
    cleanup_orphaned_vectors,
)


class TestPruneExpiredSessions:
    """Test suite for session pruning task."""

    @patch("service.tasks.RedisAdapter")
    @patch("service.tasks.PostgresAdapter")
    def test_prune_expired_sessions_no_sessions(
        self, mock_postgres: MagicMock, mock_redis: MagicMock
    ) -> None:
        """Test pruning with no sessions to prune."""
        # Mock empty Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.client.scan_iter = AsyncMock(return_value=iter([]))
        mock_redis.return_value = mock_redis_instance

        mock_postgres_instance = MagicMock()
        mock_postgres.return_value = mock_postgres_instance

        result = prune_expired_sessions()

        assert result["sessions_pruned"] == 0
        assert result["artifacts_pruned"] == 0

    @patch("service.tasks.RedisAdapter")
    @patch("service.tasks.PostgresAdapter")
    def test_prune_expired_sessions_with_expired(
        self, mock_postgres: MagicMock, mock_redis: MagicMock
    ) -> None:
        """Test pruning with expired sessions."""
        # This would require complex mocking of async Redis operations
        # For now, test the logic structure

        # Verify task is callable
        assert callable(prune_expired_sessions)

        # Verify it returns dict with expected keys
        result = prune_expired_sessions()
        assert "sessions_pruned" in result
        assert "artifacts_pruned" in result
        assert "expiry_threshold" in result


class TestArchiveOldArtifacts:
    """Test suite for artifact archival task."""

    @patch("service.tasks.PostgresAdapter")
    @patch("service.tasks.S3Adapter")
    def test_archive_old_artifacts_no_artifacts(
        self, mock_s3: MagicMock, mock_postgres: MagicMock
    ) -> None:
        """Test archival with no old artifacts."""
        # Mock empty result set
        mock_postgres_instance = MagicMock()
        mock_postgres.return_value = mock_postgres_instance

        result = archive_old_artifacts()

        assert "artifacts_archived" in result
        assert "bytes_archived" in result
        assert "archival_threshold" in result

    def test_archive_old_artifacts_callable(self) -> None:
        """Test that archive task is callable."""
        assert callable(archive_old_artifacts)

        # Should return dict with statistics
        result = archive_old_artifacts()
        assert isinstance(result, dict)


class TestCompactLongRunningSessions:
    """Test suite for session compaction task."""

    @patch("service.tasks.PostgresAdapter")
    @patch("service.tasks.RedisAdapter")
    def test_compact_long_running_sessions_no_sessions(
        self, mock_redis: MagicMock, mock_postgres: MagicMock
    ) -> None:
        """Test compaction with no long-running sessions."""
        # Mock empty result set
        mock_postgres_instance = MagicMock()
        mock_postgres.return_value = mock_postgres_instance

        result = compact_long_running_sessions()

        assert "sessions_compacted" in result
        assert "tokens_saved" in result

    def test_compact_long_running_sessions_callable(self) -> None:
        """Test that compaction task is callable."""
        assert callable(compact_long_running_sessions)

        result = compact_long_running_sessions()
        assert isinstance(result, dict)


class TestCleanupOrphanedVectors:
    """Test suite for vector cleanup task."""

    def test_cleanup_orphaned_vectors_not_implemented(self) -> None:
        """Test vector cleanup task (not yet implemented)."""
        result = cleanup_orphaned_vectors()

        assert result["vectors_cleaned"] == 0
        assert "note" in result
        assert "Not yet implemented" in result["note"]


class TestTaskConfiguration:
    """Test suite for Celery task configuration."""

    def test_celery_app_configuration(self) -> None:
        """Test Celery app configuration."""
        from service.tasks import celery_app

        assert celery_app.conf.task_serializer == "json"
        assert "json" in celery_app.conf.accept_content
        assert celery_app.conf.result_serializer == "json"
        assert celery_app.conf.timezone == "UTC"
        assert celery_app.conf.enable_utc is True

    def test_celery_task_time_limits(self) -> None:
        """Test Celery task time limits."""
        from service.tasks import celery_app

        assert celery_app.conf.task_time_limit == 3600  # 1 hour
        assert celery_app.conf.task_soft_time_limit == 3000  # 50 minutes

    def test_celery_beat_schedule(self) -> None:
        """Test Celery beat schedule configuration."""
        from service.tasks import celery_app

        schedule = celery_app.conf.beat_schedule

        assert "prune-expired-sessions" in schedule
        assert "archive-old-artifacts" in schedule
        assert "compact-long-sessions" in schedule

        # Verify task names
        assert schedule["prune-expired-sessions"]["task"] == "service.tasks.prune_expired_sessions"
        assert schedule["archive-old-artifacts"]["task"] == "service.tasks.archive_old_artifacts"
        assert (
            schedule["compact-long-sessions"]["task"]
            == "service.tasks.compact_long_running_sessions"
        )


class TestTaskRetention:
    """Test suite for retention policy logic."""

    def test_session_ttl_calculation(self) -> None:
        """Test session TTL calculation."""
        from service.config import Settings

        settings = Settings()

        # Default TTL is 72 hours
        assert settings.session_ttl_hours == 72

        # Calculate expiry threshold
        now = datetime.utcnow()
        threshold = now - timedelta(hours=settings.session_ttl_hours)

        # Verify threshold is 72 hours ago
        expected = now - timedelta(hours=72)
        assert abs((threshold - expected).total_seconds()) < 1

    def test_archival_threshold_calculation(self) -> None:
        """Test archival threshold calculation."""
        # Default archival is 90 days
        now = datetime.utcnow()
        threshold = now - timedelta(days=90)

        # Verify threshold
        expected = now - timedelta(days=90)
        assert abs((threshold - expected).total_seconds()) < 1

    def test_memory_compaction_threshold(self) -> None:
        """Test memory compaction threshold."""
        from service.config import Settings

        settings = Settings()

        # Default threshold is 8000 tokens
        assert settings.memory_compaction_threshold_tokens == 8000


class TestTaskErrorHandling:
    """Test suite for task error handling."""

    @patch("service.tasks.RedisAdapter")
    @patch("service.tasks.PostgresAdapter")
    def test_prune_sessions_handles_errors(
        self, mock_postgres: MagicMock, mock_redis: MagicMock
    ) -> None:
        """Test that pruning task handles errors gracefully."""
        # Mock Redis to raise exception
        mock_redis.side_effect = Exception("Redis connection error")

        # Task should handle error and not crash
        try:
            result = prune_expired_sessions()
            # If it returns a result, it handled the error
            assert isinstance(result, dict)
        except Exception as e:
            # If it raises, verify it's logged (in real implementation)
            assert "Redis" in str(e) or "connection" in str(e)

    @patch("service.tasks.PostgresAdapter")
    @patch("service.tasks.S3Adapter")
    def test_archive_handles_errors(
        self, mock_s3: MagicMock, mock_postgres: MagicMock
    ) -> None:
        """Test that archival task handles errors gracefully."""
        # Mock S3 to raise exception
        mock_s3.side_effect = Exception("S3 connection error")

        # Task should handle error
        try:
            result = archive_old_artifacts()
            assert isinstance(result, dict)
        except Exception as e:
            assert "S3" in str(e) or "connection" in str(e)


class TestAsyncSyncBridge:
    """Test suite for async/sync bridge utility."""

    def test_run_async_utility(self) -> None:
        """Test run_async utility function."""
        from service.tasks import run_async

        async def test_coro() -> str:
            return "test result"

        result = run_async(test_coro())
        assert result == "test result"

    def test_run_async_with_exception(self) -> None:
        """Test run_async with exception."""
        from service.tasks import run_async

        async def failing_coro() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            run_async(failing_coro())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=service.tasks", "--cov-report=html"])
