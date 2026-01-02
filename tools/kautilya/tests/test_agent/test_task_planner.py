"""
Tests for TaskPlanner - Task decomposition and planning.

Module: tests/test_agent/test_task_planner.py
"""

import pytest

from kautilya.agent.task_planner import ExecutionPlan, PlanStep, StepStatus, TaskPlanner


class TestPlanStep:
    """Tests for PlanStep."""

    def test_step_creation(self) -> None:
        """Test creating a plan step."""
        step = PlanStep(
            id="step_1",
            description="First step",
            capability_type="skill",
            capability_name="document_qa",
        )

        assert step.id == "step_1"
        assert step.status == StepStatus.PENDING
        assert step.retries == 0

    def test_step_is_ready_no_deps(self) -> None:
        """Test step readiness with no dependencies."""
        step = PlanStep(
            id="step_1",
            description="No dependencies",
            capability_type="skill",
        )

        assert step.is_ready(set()) is True
        assert step.is_ready({"other_step"}) is True

    def test_step_is_ready_with_deps(self) -> None:
        """Test step readiness with dependencies."""
        step = PlanStep(
            id="step_2",
            description="Has dependencies",
            capability_type="skill",
            dependencies=["step_1"],
        )

        # Not ready - dependency not completed
        assert step.is_ready(set()) is False
        # Ready - dependency completed
        assert step.is_ready({"step_1"}) is True

    def test_step_can_retry(self) -> None:
        """Test retry logic."""
        step = PlanStep(
            id="step_1",
            description="Retryable",
            capability_type="skill",
            max_retries=3,
        )

        assert step.can_retry() is True

        step.retries = 3
        assert step.can_retry() is False


class TestExecutionPlan:
    """Tests for ExecutionPlan."""

    def test_plan_creation(self) -> None:
        """Test creating an execution plan."""
        plan = ExecutionPlan(task_description="Test task")

        assert plan.task_description == "Test task"
        assert len(plan.steps) == 0
        assert len(plan.completed_steps) == 0

    def test_add_step(self) -> None:
        """Test adding steps to a plan."""
        plan = ExecutionPlan(task_description="Test")
        step = PlanStep(id="step_1", description="First", capability_type="skill")

        plan.add_step(step)

        assert len(plan.steps) == 1
        assert plan.steps[0].id == "step_1"

    def test_get_ready_steps(self) -> None:
        """Test getting ready steps."""
        plan = ExecutionPlan(task_description="Test")
        step1 = PlanStep(id="step_1", description="First", capability_type="skill")
        step2 = PlanStep(
            id="step_2", description="Second", capability_type="skill",
            dependencies=["step_1"]
        )

        plan.add_step(step1)
        plan.add_step(step2)

        ready = plan.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].id == "step_1"

    def test_mark_completed(self) -> None:
        """Test marking a step as completed."""
        plan = ExecutionPlan(task_description="Test")
        step = PlanStep(id="step_1", description="First", capability_type="skill")
        plan.add_step(step)

        plan.mark_completed("step_1", {"result": "done"})

        assert "step_1" in plan.completed_steps
        assert plan.steps[0].status == StepStatus.COMPLETED
        assert "step_1" in plan.context

    def test_mark_failed(self) -> None:
        """Test marking a step as failed."""
        plan = ExecutionPlan(task_description="Test")
        step = PlanStep(id="step_1", description="First", capability_type="skill", max_retries=2)
        plan.add_step(step)

        plan.mark_failed("step_1", "Error occurred")

        assert plan.steps[0].status == StepStatus.FAILED
        assert plan.steps[0].error == "Error occurred"
        assert plan.steps[0].retries == 1

    def test_is_complete(self) -> None:
        """Test checking if plan is complete."""
        plan = ExecutionPlan(task_description="Test")
        step = PlanStep(id="step_1", description="First", capability_type="skill")
        plan.add_step(step)

        assert plan.is_complete() is False

        plan.mark_completed("step_1", {})
        assert plan.is_complete() is True

    def test_is_successful(self) -> None:
        """Test checking if plan was successful."""
        plan = ExecutionPlan(task_description="Test")
        step = PlanStep(id="step_1", description="First", capability_type="skill")
        plan.add_step(step)

        plan.mark_completed("step_1", {})
        assert plan.is_successful() is True

    def test_get_final_output(self) -> None:
        """Test getting final output."""
        plan = ExecutionPlan(task_description="Test")
        step1 = PlanStep(id="step_1", description="First", capability_type="skill")
        step2 = PlanStep(id="step_2", description="Second", capability_type="skill")
        plan.add_step(step1)
        plan.add_step(step2)

        plan.mark_completed("step_1", {"output": "first"})
        plan.mark_completed("step_2", {"output": "final"})

        final = plan.get_final_output()
        assert final["output"] == "final"


class TestTaskPlanner:
    """Tests for TaskPlanner."""

    @pytest.fixture
    def planner(self) -> TaskPlanner:
        """Create a TaskPlanner instance."""
        return TaskPlanner()

    def test_analyze_document_task(self, planner: TaskPlanner) -> None:
        """Test analyzing a document extraction task."""
        analysis = planner.analyze_task("Extract ESG metrics from @reports/sample.pdf")

        assert analysis["task_type"] == "document_extraction"
        assert analysis["has_file_reference"] is True
        assert "document_processing" in analysis["required_capabilities"]

    def test_analyze_research_task(self, planner: TaskPlanner) -> None:
        """Test analyzing a research task."""
        analysis = planner.analyze_task("Research the latest trends in AI")

        assert analysis["task_type"] == "research"
        assert "search" in analysis["matched_keywords"]

    def test_analyze_code_task(self, planner: TaskPlanner) -> None:
        """Test analyzing a code analysis task."""
        analysis = planner.analyze_task("Debug and fix the authentication code")

        assert analysis["task_type"] == "code_analysis"
        assert "code_execution" in analysis["required_capabilities"]

    def test_analyze_qa_task(self, planner: TaskPlanner) -> None:
        """Test analyzing a Q&A task."""
        analysis = planner.analyze_task("What is the capital of France?")

        assert analysis["task_type"] == "qa"

    def test_analyze_complexity(self, planner: TaskPlanner) -> None:
        """Test complexity analysis."""
        simple = planner.analyze_task("Hello")
        assert simple["complexity"] == "simple"

        complex_task = "First extract the data, and then analyze trends, then generate a report"
        complex_analysis = planner.analyze_task(complex_task)
        assert complex_analysis["complexity"] == "complex"

    def test_create_plan(self, planner: TaskPlanner) -> None:
        """Test creating an execution plan."""
        plan = planner.create_plan("Extract data from @report.pdf")

        assert len(plan.steps) > 0
        assert plan.task_description == "Extract data from @report.pdf"

        # Should have file resolution step
        step_ids = [s.id for s in plan.steps]
        assert "resolve_files" in step_ids

    def test_create_plan_no_file_ref(self, planner: TaskPlanner) -> None:
        """Test creating a plan without file references."""
        plan = planner.create_plan("What is the weather today?")

        step_ids = [s.id for s in plan.steps]
        assert "resolve_files" not in step_ids

    def test_decompose_complex_task(self, planner: TaskPlanner) -> None:
        """Test decomposing a complex task."""
        task = "Extract data from the PDF and then analyze trends"
        subtasks = planner.decompose_complex_task(task)

        assert len(subtasks) >= 2
        assert any("extract" in s.lower() for s in subtasks)
        assert any("analyze" in s.lower() for s in subtasks)

    def test_estimate_complexity(self, planner: TaskPlanner) -> None:
        """Test complexity estimation."""
        plan = ExecutionPlan(task_description="Test")
        for i in range(6):
            plan.add_step(PlanStep(
                id=f"step_{i}",
                description=f"Step {i}",
                capability_type="skill",
            ))

        complexity = planner.estimate_complexity(plan)

        assert complexity["total_steps"] == 6
        assert complexity["estimated_complexity"] == "high"
