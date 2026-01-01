"""
Tests for Reflective Agent (PLAN -> EXECUTE -> VALIDATE -> REFINE).

Module: subagent-manager/tests/test_reflective_agent.py
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from subagent_manager.service.reflective_agent import (
    ExecutionPlan,
    PlanStep,
    ReflectionPhase,
    ReflectionState,
    ReflectiveAgent,
    ReflectiveAgentConfig,
    RefinementAction,
    ValidationResult,
)
from subagent_manager.service.planning_prompts import PlanningPrompts
from subagent_manager.service.refinement_engine import (
    FailureAnalysis,
    FailureCategory,
    RefinementEngine,
    RefinementStrategy,
)


class TestReflectiveAgentConfig:
    """Tests for ReflectiveAgentConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ReflectiveAgentConfig()

        assert config.enable_planning is True
        assert config.enable_self_validation is True
        assert config.enable_refinement is True
        assert config.max_iterations == 3
        assert config.validation_strictness == "medium"
        assert config.min_quality_score == 0.7
        assert config.refinement_strategy == "adaptive"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ReflectiveAgentConfig(
            max_iterations=5,
            validation_strictness="high",
            min_quality_score=0.9,
            refinement_strategy="conservative",
        )

        assert config.max_iterations == 5
        assert config.validation_strictness == "high"
        assert config.min_quality_score == 0.9
        assert config.refinement_strategy == "conservative"


class TestExecutionPlan:
    """Tests for ExecutionPlan model."""

    def test_plan_creation(self) -> None:
        """Test creating an execution plan."""
        plan = ExecutionPlan(
            task_understanding="Find and fix bugs",
            approach="Search codebase for issues",
            steps=[
                PlanStep(
                    action="Search for error patterns",
                    tool="file_grep",
                    inputs={"pattern": "error"},
                    expected_output="List of errors",
                )
            ],
            success_criteria=["All bugs found", "Fixes validated"],
            confidence=0.85,
        )

        assert plan.task_understanding == "Find and fix bugs"
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "file_grep"
        assert plan.confidence == 0.85
        assert len(plan.success_criteria) == 2


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_valid_result(self) -> None:
        """Test a valid validation result."""
        result = ValidationResult(
            is_valid=True,
            score=0.95,
            errors=[],
            warnings=["Minor style issue"],
            suggestions=[],
            criteria_met={"criterion_1": True, "criterion_2": True},
        )

        assert result.is_valid is True
        assert result.score == 0.95
        assert len(result.errors) == 0
        assert len(result.warnings) == 1

    def test_invalid_result(self) -> None:
        """Test an invalid validation result."""
        result = ValidationResult(
            is_valid=False,
            score=0.3,
            errors=["Missing required output", "Type mismatch"],
            warnings=[],
            suggestions=["Check output format"],
            criteria_met={"criterion_1": False, "criterion_2": True},
        )

        assert result.is_valid is False
        assert result.score == 0.3
        assert len(result.errors) == 2


class TestReflectionState:
    """Tests for ReflectionState."""

    def test_initial_state(self) -> None:
        """Test initial reflection state."""
        state = ReflectionState(
            task_id="test-123",
            original_task="Refactor the authentication module",
        )

        assert state.task_id == "test-123"
        assert state.current_phase == ReflectionPhase.PLANNING
        assert state.iteration == 0
        assert state.max_iterations == 3
        assert state.success is False
        assert state.final_output is None

    def test_state_updates(self) -> None:
        """Test state updates during execution."""
        state = ReflectionState(
            task_id="test-456",
            original_task="Test task",
        )

        # Simulate phase progression
        state.current_phase = ReflectionPhase.EXECUTING
        state.iteration = 1

        assert state.current_phase == ReflectionPhase.EXECUTING
        assert state.iteration == 1


class TestPlanningPrompts:
    """Tests for PlanningPrompts."""

    def test_build_planning_prompt(self) -> None:
        """Test building a planning prompt."""
        prompts = PlanningPrompts()

        prompt = prompts.build_planning_prompt(
            task="Find security vulnerabilities",
            context={"codebase": "python"},
            constraints=["Do not modify files"],
            available_tools=["file_read", "file_grep"],
        )

        assert "Find security vulnerabilities" in prompt
        assert "codebase" in prompt
        assert "Do not modify files" in prompt
        assert "file_read" in prompt

    def test_build_validation_prompt(self) -> None:
        """Test building a validation prompt."""
        prompts = PlanningPrompts()

        plan = ExecutionPlan(
            task_understanding="Test task",
            approach="Test approach",
            steps=[],
            success_criteria=["Criterion 1"],
            confidence=0.8,
        )

        prompt = prompts.build_validation_prompt(
            task="Original task",
            plan=plan,
            execution_result={"output": "result"},
            success_criteria=["Criterion 1"],
            strictness="high",
        )

        assert "Original task" in prompt
        assert "Criterion 1" in prompt
        assert "HIGH" in prompt

    def test_build_refinement_prompt(self) -> None:
        """Test building a refinement prompt."""
        prompts = PlanningPrompts()

        validation = ValidationResult(
            is_valid=False,
            score=0.4,
            errors=["Test error"],
        )

        prompt = prompts.build_refinement_prompt(
            task="Task to refine",
            plan=None,
            execution_result={"status": "failed"},
            validation_result=validation,
            iteration=1,
            max_iterations=3,
            strategy="adaptive",
        )

        assert "Task to refine" in prompt
        assert "1 of 3" in prompt
        assert "ADAPTIVE" in prompt


class TestRefinementEngine:
    """Tests for RefinementEngine."""

    def test_categorize_tool_error(self) -> None:
        """Test categorizing tool errors."""
        engine = RefinementEngine()

        state = ReflectionState(task_id="test", original_task="test")

        analysis = engine.analyze_failure(
            state=state,
            execution_result={"errors": ["tool failed to execute"]},
        )

        assert analysis.category == FailureCategory.TOOL_ERROR

    def test_categorize_timeout(self) -> None:
        """Test categorizing timeout errors."""
        engine = RefinementEngine()

        state = ReflectionState(task_id="test", original_task="test")

        analysis = engine.analyze_failure(
            state=state,
            execution_result={"errors": ["operation timed out"]},
        )

        assert analysis.category == FailureCategory.TIMEOUT
        assert analysis.is_transient is True

    def test_strategy_selection_conservative(self) -> None:
        """Test conservative strategy selection."""
        engine = RefinementEngine()

        analysis = FailureAnalysis(
            category=FailureCategory.TOOL_ERROR,
            severity=0.5,
            is_transient=True,
            root_cause="Tool failed",
            affected_steps=["step_1"],
            suggested_strategies=[
                RefinementStrategy.RETRY_SAME,
                RefinementStrategy.MODIFY_APPROACH,
            ],
        )

        state = ReflectionState(
            task_id="test",
            original_task="test",
            max_iterations=3,
        )

        action = engine.choose_refinement_action(
            analysis=analysis,
            state=state,
            strategy_preference="conservative",
        )

        assert action.action_type == "retry"

    def test_strategy_selection_aggressive(self) -> None:
        """Test aggressive strategy selection."""
        engine = RefinementEngine()

        analysis = FailureAnalysis(
            category=FailureCategory.LOGIC_ERROR,
            severity=0.8,
            is_transient=False,
            root_cause="Wrong approach",
            affected_steps=["step_1", "step_2"],
            suggested_strategies=[
                RefinementStrategy.MODIFY_APPROACH,
                RefinementStrategy.DECOMPOSE_TASK,
            ],
        )

        state = ReflectionState(
            task_id="test",
            original_task="test",
            max_iterations=3,
        )

        action = engine.choose_refinement_action(
            analysis=analysis,
            state=state,
            strategy_preference="aggressive",
        )

        assert action.action_type == "modify"
        assert action.new_approach is not None


class TestRefinementAction:
    """Tests for RefinementAction model."""

    def test_retry_action(self) -> None:
        """Test retry action."""
        action = RefinementAction(
            action_type="retry",
            reasoning="Transient error, will retry",
        )

        assert action.action_type == "retry"
        assert action.new_approach is None

    def test_modify_action(self) -> None:
        """Test modify action with new approach."""
        action = RefinementAction(
            action_type="modify",
            reasoning="Need different approach",
            modifications={"skip_steps": ["step_1"]},
            new_approach="Use alternative method",
        )

        assert action.action_type == "modify"
        assert action.new_approach == "Use alternative method"
        assert "step_1" in action.modifications["skip_steps"]

    def test_abort_action(self) -> None:
        """Test abort action."""
        action = RefinementAction(
            action_type="abort",
            reasoning="Cannot proceed - fundamental issue",
        )

        assert action.action_type == "abort"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
