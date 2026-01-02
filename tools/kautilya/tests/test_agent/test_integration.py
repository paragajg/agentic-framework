"""
Integration tests for Kautilya Agent components.

Module: tests/test_agent/test_integration.py

Tests the full agentic workflow including:
- File resolution
- Capability discovery
- Task planning
- Error recovery
- Session memory
- Output validation
"""

import os
import tempfile
from pathlib import Path

import pytest

from kautilya.agent import (
    AgentCore,
    CapabilityRegistry,
    ErrorCategory,
    ErrorRecoveryEngine,
    ExecutionPlan,
    FileResolver,
    OutputValidator,
    RecoveryStrategy,
    SessionMemory,
    TaskPlanner,
    ValidationLevel,
)


class TestAgentIntegration:
    """Integration tests for agent components working together."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory with sample files."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create project structure
            (Path(tmp) / "reports").mkdir()
            (Path(tmp) / "reports" / "sample.pdf").write_text("PDF content")
            (Path(tmp) / "data").mkdir()
            (Path(tmp) / "data" / "metrics.csv").write_text("a,b,c\n1,2,3")
            (Path(tmp) / "src").mkdir()
            (Path(tmp) / "src" / "main.py").write_text("def main(): pass")

            original_cwd = os.getcwd()
            os.chdir(tmp)
            yield tmp
            os.chdir(original_cwd)

    def test_file_resolver_with_session_memory(self, temp_project: str) -> None:
        """Test that file resolution results are cached in session memory."""
        resolver = FileResolver()
        memory = SessionMemory()

        # Resolve a file
        match = resolver.resolve("reports/sample.pdf")
        assert match.path.exists()

        # Cache the resolution
        memory.cache_file_resolution("@reports/sample.pdf", str(match.path))

        # Retrieve from cache
        cached = memory.get_cached_file("@reports/sample.pdf")
        assert cached == str(match.path)

    def test_task_planner_with_capability_registry(self) -> None:
        """Test task planning with capability discovery."""
        planner = TaskPlanner()
        registry = CapabilityRegistry()

        # Analyze a document task with @file reference
        analysis = planner.analyze_task("Extract ESG metrics from @reports/annual_report.pdf")

        assert analysis["task_type"] == "document_extraction"
        assert analysis["has_file_reference"] is True
        assert "document_processing" in analysis["required_capabilities"]

        # Find matching capabilities
        matches = registry.match_capabilities("extract data from PDF document")
        # Should find document-related capabilities
        assert any("document" in m.name.lower() or "pdf" in m.name.lower() for m in matches[:5])

    def test_error_recovery_with_session_memory(self) -> None:
        """Test that error recovery learns from session history."""
        engine = ErrorRecoveryEngine()
        memory = SessionMemory()

        # Simulate a file not found error
        error = FileNotFoundError("File not found: report.pdf")
        memory.start_task("task_001", "Read report.pdf")

        # Analyze error
        analysis = engine.analyze_error(error, task="Read report.pdf")
        assert analysis.category == ErrorCategory.FILE_NOT_FOUND
        assert analysis.is_recoverable is True

        # Record a successful recovery
        engine.record_recovery_attempt(
            error=error,
            strategy=RecoveryStrategy.REQUEST_USER_INPUT,
            success=True,
        )

        # Next time, confidence should be adjusted
        analysis2 = engine.analyze_error(error, task="Read another report")
        request_input_action = next(
            (a for a in analysis2.recovery_actions if a.strategy == RecoveryStrategy.REQUEST_USER_INPUT),
            None
        )
        assert request_input_action is not None

        # Record in session memory
        memory.record_recovery("task_001", "request_user_input", True)
        assert len(memory.task_memory["task_001"].recovery_attempts) == 1

    def test_output_validation_with_schema(self) -> None:
        """Test output validation against JSON schema."""
        validator = OutputValidator()

        schema = {
            "type": "object",
            "required": ["metrics", "summary"],
            "properties": {
                "metrics": {"type": "array"},
                "summary": {"type": "string"},
            },
        }

        # Valid output
        valid_output = '{"metrics": [1, 2, 3], "summary": "Analysis complete"}'
        result = validator.validate(valid_output, expected_schema=schema)
        assert result.is_valid is True

        # Invalid output (missing required field)
        invalid_output = '{"metrics": [1, 2, 3]}'
        result = validator.validate(invalid_output, expected_schema=schema)
        assert any(i.message for i in result.issues if "required" in i.message.lower() or "missing" in i.message.lower())

    def test_full_task_workflow(self, temp_project: str) -> None:
        """Test a complete task workflow from planning to execution."""
        planner = TaskPlanner()
        resolver = FileResolver()
        memory = SessionMemory()
        validator = OutputValidator()

        # 1. Plan the task
        task = "Extract data from @reports/sample.pdf and summarize"
        analysis = planner.analyze_task(task)
        assert analysis["task_type"] == "document_extraction"

        # 2. Create execution plan
        plan = planner.create_plan(task)
        assert len(plan.steps) > 0

        # 3. Resolve file references
        file_refs = resolver.extract_file_references(task)
        assert "reports/sample.pdf" in file_refs

        match = resolver.resolve("reports/sample.pdf")
        memory.cache_file_resolution("@reports/sample.pdf", str(match.path))

        # 4. Start task execution
        memory.start_task("workflow_001", task)

        # 5. Simulate step completion
        plan.mark_completed("resolve_files", {"files": [str(match.path)]})

        # 6. Generate output
        output = "The document contains quarterly financial data."
        result = validator.validate(output, task=task)
        assert result.is_valid is True

        # 7. Complete task
        memory.complete_task("workflow_001", {"output": output})
        assert memory.task_memory["workflow_001"].status == "completed"

    def test_capability_matching_for_different_tasks(self) -> None:
        """Test that capability registry matches appropriate skills for different task types."""
        registry = CapabilityRegistry()

        # Document task
        doc_matches = registry.match_capabilities("extract text from PDF document")
        assert len(doc_matches) > 0

        # Code task
        code_matches = registry.match_capabilities("debug the Python script and fix errors")
        assert len(code_matches) > 0

        # Research task
        research_matches = registry.match_capabilities("research AI trends and summarize findings")
        assert len(research_matches) > 0

    def test_session_memory_export_and_summary(self) -> None:
        """Test session memory tracking and export."""
        memory = SessionMemory()

        # Add various entries
        memory.start_task("task_1", "Task 1")
        memory.complete_task("task_1", {"result": "done"})

        memory.start_task("task_2", "Task 2")
        memory.fail_task("task_2", "Error occurred")

        memory.cache_file_resolution("@file.pdf", "/path/to/file.pdf")
        memory.record_capability_usage("document_qa", True, 1.5)

        # Get summary
        summary = memory.get_summary()
        assert summary["tasks_completed"] >= 1
        assert summary["tasks_failed"] >= 1
        assert summary["files_cached"] >= 1

        # Export
        exported = memory.export()
        assert "task_memory" in exported
        assert "file_cache" in exported


class TestAgentCoreBasic:
    """Basic tests for AgentCore without LLM."""

    def test_agent_core_initialization(self) -> None:
        """Test that AgentCore initializes correctly without LLM."""
        agent = AgentCore(llm_client=None)

        assert agent.file_resolver is not None
        assert agent.capability_registry is not None
        assert agent.task_planner is not None
        assert agent.error_recovery is not None
        assert agent.session_memory is not None
        assert agent.output_validator is not None

    def test_agent_core_analyze_task(self) -> None:
        """Test task analysis without LLM."""
        agent = AgentCore(llm_client=None)

        # Should be able to analyze task type
        analysis = agent.task_planner.analyze_task("Extract data from PDF")
        assert analysis["task_type"] == "document_extraction"

    def test_agent_core_capability_discovery(self) -> None:
        """Test that agent discovers capabilities."""
        agent = AgentCore(llm_client=None)

        # Force discovery
        agent.capability_registry.discover_all()

        # Should have discovered some capabilities
        all_caps = agent.capability_registry.get_all_capabilities()
        assert len(all_caps) > 0

        # Should have skills
        skills = agent.capability_registry.get_skills()
        assert len(skills) > 0
