"""
Tests for SessionMemory - In-session learning and context.

Module: tests/test_agent/test_session_memory.py
"""

import pytest

from kautilya.agent.session_memory import MemoryEntry, MemoryType, SessionMemory, TaskMemory


class TestMemoryEntry:
    """Tests for MemoryEntry."""

    def test_entry_creation(self) -> None:
        """Test creating a memory entry."""
        entry = MemoryEntry(
            id="test_001",
            type=MemoryType.TASK,
            content={"action": "test"},
        )

        assert entry.id == "test_001"
        assert entry.type == MemoryType.TASK
        assert entry.relevance_score == 1.0

    def test_entry_to_dict(self) -> None:
        """Test converting entry to dictionary."""
        entry = MemoryEntry(
            id="test_001",
            type=MemoryType.RESULT,
            content={"data": "result"},
            metadata={"source": "test"},
        )

        d = entry.to_dict()
        assert d["id"] == "test_001"
        assert d["type"] == "result"
        assert "timestamp" in d


class TestSessionMemory:
    """Tests for SessionMemory."""

    @pytest.fixture
    def memory(self) -> SessionMemory:
        """Create a SessionMemory instance."""
        return SessionMemory()

    def test_add_entry(self, memory: SessionMemory) -> None:
        """Test adding a memory entry."""
        entry_id = memory.add(
            MemoryType.TASK,
            {"description": "Test task"},
        )

        assert entry_id is not None
        assert len(memory.entries) == 1

    def test_get_recent(self, memory: SessionMemory) -> None:
        """Test getting recent entries."""
        for i in range(5):
            memory.add(MemoryType.TASK, {"num": i})

        recent = memory.get_recent(count=3)
        assert len(recent) == 3

    def test_get_recent_by_type(self, memory: SessionMemory) -> None:
        """Test getting recent entries by type."""
        memory.add(MemoryType.TASK, {"task": "1"})
        memory.add(MemoryType.ERROR, {"error": "1"})
        memory.add(MemoryType.TASK, {"task": "2"})

        tasks = memory.get_recent(count=10, type_filter=MemoryType.TASK)
        assert len(tasks) == 2
        assert all(e.type == MemoryType.TASK for e in tasks)

    def test_search(self, memory: SessionMemory) -> None:
        """Test searching memory."""
        memory.add(MemoryType.TASK, {"description": "Extract ESG metrics"})
        memory.add(MemoryType.TASK, {"description": "Analyze revenue data"})
        memory.add(MemoryType.TASK, {"description": "Generate report"})

        results = memory.search("ESG")
        assert len(results) >= 1
        assert any("ESG" in str(e.content) for e in results)

    def test_start_task(self, memory: SessionMemory) -> None:
        """Test starting a task."""
        memory.start_task("task_001", "Test task", {"input": "data"})

        assert "task_001" in memory.task_memory
        task = memory.task_memory["task_001"]
        assert task.status == "in_progress"
        assert task.inputs == {"input": "data"}

    def test_complete_task(self, memory: SessionMemory) -> None:
        """Test completing a task."""
        memory.start_task("task_001", "Test task")
        memory.complete_task("task_001", {"result": "success"})

        task = memory.task_memory["task_001"]
        assert task.status == "completed"
        assert task.outputs == {"result": "success"}
        assert task.end_time is not None

    def test_fail_task(self, memory: SessionMemory) -> None:
        """Test failing a task."""
        memory.start_task("task_001", "Test task")
        memory.fail_task("task_001", "Error occurred")

        task = memory.task_memory["task_001"]
        assert task.status == "failed"
        assert "Error occurred" in task.errors

    def test_record_recovery(self, memory: SessionMemory) -> None:
        """Test recording a recovery attempt."""
        memory.start_task("task_001", "Test task")
        memory.record_recovery("task_001", "retry_same", True)

        task = memory.task_memory["task_001"]
        assert len(task.recovery_attempts) == 1
        assert task.recovery_attempts[0]["success"] is True

    def test_cache_file_resolution(self, memory: SessionMemory) -> None:
        """Test caching file resolutions."""
        memory.cache_file_resolution("@report.pdf", "/full/path/report.pdf")

        cached = memory.get_cached_file("@report.pdf")
        assert cached == "/full/path/report.pdf"

    def test_get_cached_file_not_found(self, memory: SessionMemory) -> None:
        """Test getting uncached file."""
        cached = memory.get_cached_file("nonexistent")
        assert cached is None

    def test_record_capability_usage(self, memory: SessionMemory) -> None:
        """Test recording capability usage."""
        memory.record_capability_usage("document_qa", True, 1.5)
        memory.record_capability_usage("document_qa", True, 2.0)
        memory.record_capability_usage("document_qa", False, 0.5, "Error")

        stats = memory.capability_stats["document_qa"]
        assert stats["total_calls"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1

    def test_get_capability_success_rate(self, memory: SessionMemory) -> None:
        """Test getting capability success rate."""
        memory.record_capability_usage("test_skill", True, 1.0)
        memory.record_capability_usage("test_skill", True, 1.0)
        memory.record_capability_usage("test_skill", False, 1.0)

        rate = memory.get_capability_success_rate("test_skill")
        assert rate == pytest.approx(0.666, rel=0.01)

    def test_get_capability_success_rate_unknown(self, memory: SessionMemory) -> None:
        """Test success rate for unknown capability."""
        rate = memory.get_capability_success_rate("unknown_skill")
        assert rate == 0.5  # Default for unknown

    def test_record_user_correction(self, memory: SessionMemory) -> None:
        """Test recording user corrections."""
        memory.record_user_correction(
            original_output="Wrong answer",
            correction="The correct answer is...",
            context={"topic": "math"},
        )

        assert len(memory.user_corrections) == 1
        assert memory.user_corrections[0]["correction"] == "The correct answer is..."

    def test_get_similar_corrections(self, memory: SessionMemory) -> None:
        """Test finding similar corrections."""
        memory.record_user_correction("Math calculation error", "Use formula X")
        memory.record_user_correction("Date parsing issue", "Use ISO format")

        similar = memory.get_similar_corrections("math error in calculation")
        assert len(similar) >= 1

    def test_working_context(self, memory: SessionMemory) -> None:
        """Test working context management."""
        memory.set_context("current_file", "report.pdf")
        memory.set_context("mode", "analysis")

        assert memory.get_context("current_file") == "report.pdf"
        assert memory.get_context("mode") == "analysis"
        assert memory.get_context("nonexistent") is None

    def test_clear_context(self, memory: SessionMemory) -> None:
        """Test clearing working context."""
        memory.set_context("key1", "value1")
        memory.set_context("key2", "value2")

        memory.clear_context()
        assert memory.get_context("key1") is None

    def test_build_context_prompt(self, memory: SessionMemory) -> None:
        """Test building context prompt."""
        memory.start_task("task_1", "Previous task")
        memory.complete_task("task_1", {})
        memory.set_context("topic", "ESG analysis")

        prompt = memory.build_context_prompt("Current task about ESG")

        # Should include recent tasks and context
        assert isinstance(prompt, str)

    def test_get_summary(self, memory: SessionMemory) -> None:
        """Test getting session summary."""
        memory.start_task("task_1", "Task 1")
        memory.complete_task("task_1", {})
        memory.cache_file_resolution("file.pdf", "/path/file.pdf")

        summary = memory.get_summary()

        assert "total_entries" in summary
        assert "tasks_completed" in summary
        assert summary["tasks_completed"] >= 1
        assert summary["files_cached"] >= 1

    def test_export(self, memory: SessionMemory) -> None:
        """Test exporting session memory."""
        memory.add(MemoryType.TASK, {"test": "data"})
        memory.start_task("task_1", "Task 1")

        exported = memory.export()

        assert "entries" in exported
        assert "task_memory" in exported
        assert "file_cache" in exported

    def test_decay_relevance(self, memory: SessionMemory) -> None:
        """Test that relevance decays for older entries."""
        memory.add(MemoryType.TASK, {"num": 1})
        first_entry = memory.entries[0]
        initial_relevance = first_entry.relevance_score

        # Add more entries to trigger decay
        for i in range(5):
            memory.add(MemoryType.TASK, {"num": i + 2})

        # First entry should have decayed relevance
        assert first_entry.relevance_score < initial_relevance

    def test_max_entries_limit(self) -> None:
        """Test that memory respects max entries limit."""
        memory = SessionMemory(max_entries=10)

        for i in range(15):
            memory.add(MemoryType.TASK, {"num": i})

        assert len(memory.entries) <= 10
