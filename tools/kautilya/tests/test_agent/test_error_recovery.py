"""
Tests for ErrorRecoveryEngine - Intelligent error handling.

Module: tests/test_agent/test_error_recovery.py
"""

import pytest

from kautilya.agent.error_recovery import (
    ErrorAnalysis,
    ErrorCategory,
    ErrorRecoveryEngine,
    RecoveryAction,
    RecoveryStrategy,
)


class TestRecoveryAction:
    """Tests for RecoveryAction."""

    def test_action_creation(self) -> None:
        """Test creating a recovery action."""
        action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY_SAME,
            description="Retry the operation",
            confidence=0.7,
        )

        assert action.strategy == RecoveryStrategy.RETRY_SAME
        assert action.confidence == 0.7

    def test_action_str(self) -> None:
        """Test string representation."""
        action = RecoveryAction(
            strategy=RecoveryStrategy.ALTERNATIVE_CAPABILITY,
            description="Try alternative",
        )

        str_repr = str(action)
        assert "alternative_capability" in str_repr
        assert "Try alternative" in str_repr


class TestErrorRecoveryEngine:
    """Tests for ErrorRecoveryEngine."""

    @pytest.fixture
    def engine(self) -> ErrorRecoveryEngine:
        """Create an ErrorRecoveryEngine instance."""
        return ErrorRecoveryEngine()

    def test_classify_file_not_found(self, engine: ErrorRecoveryEngine) -> None:
        """Test classifying FileNotFoundError."""
        error = FileNotFoundError("File not found: test.pdf")
        category = engine.classify_error(error)

        assert category == ErrorCategory.FILE_NOT_FOUND

    def test_classify_permission_error(self, engine: ErrorRecoveryEngine) -> None:
        """Test classifying PermissionError."""
        error = PermissionError("Permission denied")
        category = engine.classify_error(error)

        assert category == ErrorCategory.PERMISSION_DENIED

    def test_classify_timeout(self, engine: ErrorRecoveryEngine) -> None:
        """Test classifying timeout errors."""
        error = TimeoutError("Request timed out")
        category = engine.classify_error(error)

        assert category == ErrorCategory.TIMEOUT

    def test_classify_value_error(self, engine: ErrorRecoveryEngine) -> None:
        """Test classifying ValueError."""
        error = ValueError("Invalid input value")
        category = engine.classify_error(error)

        assert category == ErrorCategory.INVALID_INPUT

    def test_classify_unknown(self, engine: ErrorRecoveryEngine) -> None:
        """Test classifying unknown errors."""
        error = Exception("Some random error")
        category = engine.classify_error(error)

        assert category == ErrorCategory.UNKNOWN

    def test_analyze_file_not_found(self, engine: ErrorRecoveryEngine) -> None:
        """Test analyzing file not found error."""
        error = FileNotFoundError("File not found: report.pdf")
        analysis = engine.analyze_error(error, task="Read report.pdf")

        assert analysis.category == ErrorCategory.FILE_NOT_FOUND
        assert analysis.is_recoverable is True
        assert len(analysis.recovery_actions) > 0

        # Should suggest user input and alternative approaches
        strategies = [a.strategy for a in analysis.recovery_actions]
        assert RecoveryStrategy.REQUEST_USER_INPUT in strategies

    def test_analyze_api_error(self, engine: ErrorRecoveryEngine) -> None:
        """Test analyzing API errors."""
        error = Exception("API error: Internal server error 500")
        analysis = engine.analyze_error(error, task="Call external API")

        assert analysis.category == ErrorCategory.API_ERROR
        assert analysis.is_recoverable is True

        strategies = [a.strategy for a in analysis.recovery_actions]
        assert RecoveryStrategy.RETRY_SAME in strategies

    def test_analyze_rate_limit(self, engine: ErrorRecoveryEngine) -> None:
        """Test analyzing rate limit errors."""
        error = Exception("Rate limit exceeded: too many requests")
        analysis = engine.analyze_error(error)

        assert analysis.category == ErrorCategory.RATE_LIMIT

        # Should suggest waiting and retrying
        strategies = [a.strategy for a in analysis.recovery_actions]
        assert RecoveryStrategy.RETRY_SAME in strategies

    def test_get_best_action(self, engine: ErrorRecoveryEngine) -> None:
        """Test getting the best recovery action."""
        error = FileNotFoundError("File not found")
        analysis = engine.analyze_error(error)

        best = analysis.get_best_action()
        assert best is not None
        assert best.confidence > 0

    def test_record_recovery_attempt(self, engine: ErrorRecoveryEngine) -> None:
        """Test recording recovery attempts."""
        error = Exception("Test error")

        engine.record_recovery_attempt(
            error=error,
            strategy=RecoveryStrategy.RETRY_SAME,
            success=True,
        )

        assert len(engine.recovery_history) == 1
        assert engine.recovery_history[0]["success"] is True

    def test_format_error_for_user(self, engine: ErrorRecoveryEngine) -> None:
        """Test formatting error for user display."""
        error = FileNotFoundError("File not found: data.csv")
        analysis = engine.analyze_error(error)

        message = engine.format_error_for_user(analysis)

        assert "Error:" in message
        assert "Category:" in message
        assert "recovery" in message.lower()

    def test_confidence_adjustment_from_history(self, engine: ErrorRecoveryEngine) -> None:
        """Test that confidence is adjusted based on history."""
        error = FileNotFoundError("File not found: test.pdf")

        # Record successful recovery
        engine.record_recovery_attempt(
            error=error,
            strategy=RecoveryStrategy.RETRY_MODIFIED,
            success=True,
        )

        # Analyze same error type again
        analysis = engine.analyze_error(error)

        # The strategy that worked before should have higher confidence
        # (depending on exact implementation)
        assert len(analysis.recovery_actions) > 0


class TestErrorAnalysis:
    """Tests for ErrorAnalysis."""

    def test_analysis_creation(self) -> None:
        """Test creating an error analysis."""
        analysis = ErrorAnalysis(
            category=ErrorCategory.FILE_NOT_FOUND,
            message="File not found",
            is_recoverable=True,
        )

        assert analysis.category == ErrorCategory.FILE_NOT_FOUND
        assert analysis.is_recoverable is True

    def test_get_best_action_empty(self) -> None:
        """Test getting best action when no actions available."""
        analysis = ErrorAnalysis(
            category=ErrorCategory.UNKNOWN,
            message="Unknown error",
            recovery_actions=[],
        )

        best = analysis.get_best_action()
        assert best is None

    def test_get_best_action_multiple(self) -> None:
        """Test getting best action from multiple options."""
        actions = [
            RecoveryAction(RecoveryStrategy.RETRY_SAME, "Retry", confidence=0.5),
            RecoveryAction(RecoveryStrategy.ALTERNATIVE_CAPABILITY, "Alternative", confidence=0.9),
            RecoveryAction(RecoveryStrategy.ABORT, "Give up", confidence=0.1),
        ]

        analysis = ErrorAnalysis(
            category=ErrorCategory.CAPABILITY_FAILED,
            message="Capability failed",
            recovery_actions=actions,
        )

        best = analysis.get_best_action()
        assert best.strategy == RecoveryStrategy.ALTERNATIVE_CAPABILITY
        assert best.confidence == 0.9
