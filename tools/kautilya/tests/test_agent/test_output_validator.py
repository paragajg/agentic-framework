"""
Tests for OutputValidator - Self-critique and validation.

Module: tests/test_agent/test_output_validator.py
"""

import pytest

from kautilya.agent.output_validator import (
    IssueSeverity,
    IssueType,
    OutputValidator,
    ValidationIssue,
    ValidationLevel,
    ValidationResult,
)


class TestValidationIssue:
    """Tests for ValidationIssue."""

    def test_issue_creation(self) -> None:
        """Test creating a validation issue."""
        issue = ValidationIssue(
            type=IssueType.INCOMPLETE,
            severity=IssueSeverity.WARNING,
            message="Response is incomplete",
        )

        assert issue.type == IssueType.INCOMPLETE
        assert issue.severity == IssueSeverity.WARNING

    def test_issue_str(self) -> None:
        """Test string representation."""
        issue = ValidationIssue(
            type=IssueType.MISSING_SOURCES,
            severity=IssueSeverity.INFO,
            message="No citations provided",
        )

        str_repr = str(issue)
        assert "INFO" in str_repr
        assert "missing_sources" in str_repr
        assert "No citations" in str_repr


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_result_creation(self) -> None:
        """Test creating a validation result."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.95,
        )

        assert result.is_valid is True
        assert result.confidence == 0.95

    def test_result_with_issues(self) -> None:
        """Test result with issues."""
        issues = [
            ValidationIssue(IssueType.INCOMPLETE, IssueSeverity.WARNING, "Missing info"),
            ValidationIssue(IssueType.MISSING_SOURCES, IssueSeverity.INFO, "No sources"),
        ]

        result = ValidationResult(
            is_valid=True,
            confidence=0.8,
            issues=issues,
        )

        assert len(result.issues) == 2

    def test_get_summary(self) -> None:
        """Test getting result summary."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.9,
            issues=[
                ValidationIssue(IssueType.INCOMPLETE, IssueSeverity.WARNING, "Test issue"),
            ],
            suggestions=["Add more detail"],
        )

        summary = result.get_summary()
        assert "Valid: True" in summary
        assert "Confidence: 90%" in summary
        assert "Test issue" in summary
        assert "Add more detail" in summary


class TestOutputValidator:
    """Tests for OutputValidator."""

    @pytest.fixture
    def validator(self) -> OutputValidator:
        """Create an OutputValidator instance."""
        return OutputValidator()

    def test_validate_empty_output(self, validator: OutputValidator) -> None:
        """Test validating empty output."""
        result = validator.validate("")

        assert result.is_valid is False
        assert any(i.severity == IssueSeverity.CRITICAL for i in result.issues)

    def test_validate_short_output(self, validator: OutputValidator) -> None:
        """Test validating very short output."""
        result = validator.validate("OK")

        assert any(i.type == IssueType.INCOMPLETE for i in result.issues)

    def test_validate_valid_output(self, validator: OutputValidator) -> None:
        """Test validating a good output."""
        output = """
        The analysis shows that revenue increased by 15% in Q3.
        Key factors include market expansion and improved efficiency.
        The data supports continued growth projections.
        """

        result = validator.validate(output)

        assert result.is_valid is True
        assert result.confidence > 0.5

    def test_validate_with_error_indicators(self, validator: OutputValidator) -> None:
        """Test detecting error indicators."""
        output = "Error: Unable to process the request due to invalid input"

        result = validator.validate(output)

        assert any(i.type == IssueType.POTENTIALLY_INCORRECT for i in result.issues)

    def test_validate_with_schema(self, validator: OutputValidator) -> None:
        """Test validating against JSON schema."""
        schema = {
            "type": "object",
            "required": ["answer", "confidence"],
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }

        # Valid JSON
        valid_output = '{"answer": "Test answer", "confidence": 0.9}'
        result = validator.validate(valid_output, expected_schema=schema)
        assert result.is_valid is True

        # Missing required field
        invalid_output = '{"answer": "Test"}'
        result = validator.validate(invalid_output, expected_schema=schema)
        assert any(i.type == IssueType.INCOMPLETE for i in result.issues)

    def test_validate_schema_wrong_type(self, validator: OutputValidator) -> None:
        """Test validating wrong type against schema."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }

        output = '{"count": "not a number"}'
        result = validator.validate(output, expected_schema=schema)

        assert any(i.type == IssueType.FORMAT_ERROR for i in result.issues)

    def test_detect_hallucination_indicators(self, validator: OutputValidator) -> None:
        """Test detecting potential hallucination indicators."""
        output = """
        I believe the revenue was around $5 million.
        I think this might be accurate, but I'm not sure.
        Probably the company expanded, as far as I know.
        """

        result = validator.validate(output)

        assert any(i.type == IssueType.HALLUCINATION_RISK for i in result.issues)

    def test_detect_unsupported_claims(self, validator: OutputValidator) -> None:
        """Test detecting unsupported claims."""
        output = """
        According to studies, the method is effective.
        Research indicates significant improvements.
        Experts say this is the best approach.
        """

        result = validator.validate(output)

        assert any(i.type == IssueType.MISSING_SOURCES for i in result.issues)

    def test_task_relevance_check(self, validator: OutputValidator) -> None:
        """Test checking task relevance."""
        task = "What is the weather in Tokyo?"
        output = "Python is a programming language used for various applications."

        result = validator.validate(output, task=task)

        # Output doesn't address the task
        assert any(i.type == IssueType.INCONSISTENT for i in result.issues)

    def test_quick_check_valid(self, validator: OutputValidator) -> None:
        """Test quick validity check for valid output."""
        assert validator.quick_check("This is a valid response.") is True

    def test_quick_check_empty(self, validator: OutputValidator) -> None:
        """Test quick validity check for empty output."""
        assert validator.quick_check("") is False
        assert validator.quick_check("   ") is False

    def test_quick_check_error(self, validator: OutputValidator) -> None:
        """Test quick validity check for error output."""
        assert validator.quick_check("error: something went wrong") is False

    def test_calculate_confidence(self, validator: OutputValidator) -> None:
        """Test confidence calculation."""
        # No issues - high confidence
        result = validator.validate("A complete and valid response with proper detail.")
        assert result.confidence > 0.8

        # With critical issue - low confidence
        result = validator.validate("")
        assert result.confidence < 0.5

    def test_generate_suggestions(self, validator: OutputValidator) -> None:
        """Test generating improvement suggestions."""
        output = "I think research shows this might be correct."

        result = validator.validate(output)

        assert len(result.suggestions) > 0

    def test_validation_levels(self, validator: OutputValidator) -> None:
        """Test different validation levels."""
        output = "Test output for validation"

        # Basic validation
        basic = validator.validate(output, level=ValidationLevel.BASIC)

        # Standard validation (default)
        standard = validator.validate(output, level=ValidationLevel.STANDARD)

        # Standard should have at least as many checks as basic
        assert basic.confidence <= standard.confidence or len(basic.issues) <= len(standard.issues)
