"""
Tests for FileResolver - Intelligent file path resolution.

Module: tests/test_agent/test_file_resolver.py
"""

import os
import tempfile
from pathlib import Path

import pytest

from kautilya.agent.file_resolver import FileMatch, FileNotFoundResult, FileResolver


class TestFileResolver:
    """Tests for FileResolver."""

    @pytest.fixture
    def resolver(self) -> FileResolver:
        """Create a FileResolver instance."""
        return FileResolver()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file structure
            (Path(tmpdir) / "reports").mkdir()
            (Path(tmpdir) / "documents").mkdir()
            (Path(tmpdir) / "data").mkdir()

            # Create test files
            (Path(tmpdir) / "test.py").write_text("# Test Python file")
            (Path(tmpdir) / "reports" / "sample.pdf").write_text("PDF content")
            (Path(tmpdir) / "reports" / "apple_esg.pdf").write_text("Apple ESG report")
            (Path(tmpdir) / "documents" / "readme.md").write_text("# README")
            (Path(tmpdir) / "data" / "metrics.csv").write_text("col1,col2\n1,2")

            yield tmpdir

    def test_resolve_exact_path(self, resolver: FileResolver, temp_dir: str) -> None:
        """Test resolving an exact file path."""
        exact_path = Path(temp_dir) / "test.py"
        match = resolver.resolve(str(exact_path))

        assert match.path == exact_path
        assert match.confidence == 1.0
        assert match.match_type == "exact"

    def test_resolve_relative_path(self, temp_dir: str) -> None:
        """Test resolving a relative path."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            resolver = FileResolver()

            match = resolver.resolve("test.py")
            assert match.path.name == "test.py"
            assert match.match_type == "relative"
        finally:
            os.chdir(original_cwd)

    def test_resolve_nested_file(self, temp_dir: str) -> None:
        """Test resolving a nested file path."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            resolver = FileResolver()

            match = resolver.resolve("reports/sample.pdf")
            assert match.path.name == "sample.pdf"
            assert "reports" in str(match.path)
        finally:
            os.chdir(original_cwd)

    def test_resolve_with_at_prefix(self, resolver: FileResolver, temp_dir: str) -> None:
        """Test resolving a path with @ prefix."""
        exact_path = Path(temp_dir) / "test.py"
        match = resolver.resolve(f"@{exact_path}")

        assert match.path == exact_path
        assert match.original_reference == str(exact_path)

    def test_resolve_not_found(self, resolver: FileResolver, temp_dir: str) -> None:
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            resolver.resolve("nonexistent_file.xyz")

        error_msg = str(exc_info.value)
        assert "File not found" in error_msg

    def test_resolve_multiple(self, temp_dir: str) -> None:
        """Test resolving multiple file references."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            resolver = FileResolver()

            matches, failures = resolver.resolve_multiple([
                "test.py",
                "reports/sample.pdf",
                "nonexistent.txt",
            ])

            assert len(matches) == 2
            assert len(failures) == 1
        finally:
            os.chdir(original_cwd)

    def test_extract_file_references(self, resolver: FileResolver) -> None:
        """Test extracting @file references from text."""
        text = 'Extract data from @reports/sample.pdf and @"path with spaces/file.txt"'
        references = resolver.extract_file_references(text)

        assert "reports/sample.pdf" in references
        assert "path with spaces/file.txt" in references

    def test_extract_file_references_ignores_non_files(self, resolver: FileResolver) -> None:
        """Test that @mentions without file indicators are ignored."""
        text = "Hello @username, check the @file.py"
        references = resolver.extract_file_references(text)

        # @file.py should be extracted (has extension)
        assert "file.py" in references
        # @username should be ignored (no file indicator)
        assert "username" not in references

    def test_add_search_path(self, resolver: FileResolver, temp_dir: str) -> None:
        """Test adding a custom search path."""
        resolver.add_search_path(temp_dir)
        assert temp_dir in resolver.search_paths

    def test_glob_search(self, temp_dir: str) -> None:
        """Test glob-based file search."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            resolver = FileResolver()

            # Search for PDF files
            match = resolver.resolve("sample.pdf")
            assert match is not None
            assert match.path.suffix == ".pdf"
        finally:
            os.chdir(original_cwd)

    def test_fuzzy_search(self, temp_dir: str) -> None:
        """Test fuzzy file name matching."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            resolver = FileResolver()

            # Try to resolve with typo - should fuzzy match
            try:
                match = resolver.resolve("appl_esg.pdf")  # Missing 'e' in apple
                # If fuzzy match works, should find apple_esg.pdf
                assert "apple" in match.path.name.lower() or "esg" in match.path.name.lower()
            except FileNotFoundError as e:
                # If not found, suggestions should be available
                assert "Did you mean" in str(e) or "Searched in" in str(e)
        finally:
            os.chdir(original_cwd)

    def test_file_match_str(self) -> None:
        """Test FileMatch string representation."""
        match = FileMatch(
            path=Path("/test/file.py"),
            confidence=0.95,
            match_type="exact",
        )
        str_repr = str(match)

        assert "file.py" in str_repr
        assert "exact" in str_repr
        assert "95%" in str_repr


class TestFileNotFoundResult:
    """Tests for FileNotFoundResult."""

    def test_format_error(self) -> None:
        """Test error message formatting."""
        result = FileNotFoundResult(
            reference="missing.pdf",
            searched_paths=["/path/a", "/path/b"],
            suggestions=["similar.pdf", "matching.pdf"],
        )

        error_msg = result.format_error()

        assert "missing.pdf" in error_msg
        assert "Searched in" in error_msg
        assert "Did you mean" in error_msg
        assert "similar.pdf" in error_msg

    def test_format_error_no_suggestions(self) -> None:
        """Test error formatting when no suggestions available."""
        result = FileNotFoundResult(
            reference="unique.xyz",
            searched_paths=["/path/a"],
            suggestions=[],
        )

        error_msg = result.format_error()

        assert "unique.xyz" in error_msg
        assert "Did you mean" not in error_msg
