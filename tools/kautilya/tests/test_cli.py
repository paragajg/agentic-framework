"""
Tests for Kautilya CLI Utility.

Module: tests/test_cli.py
"""

import pytest
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import tempfile
import os

from kautilya.cli import cli
from kautilya.config import Config, load_config, save_config


class TestCLIBasics:
    """Test suite for basic CLI functionality."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cli_version(self, runner: CliRunner) -> None:
        """Test --version flag."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "Kautilya v1.0.0" in result.output

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test --help flag."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Kautilya" in result.output
        assert "Enterprise Agentic Framework CLI" in result.output


class TestInitCommand:
    """Test suite for /init command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_init_command_with_options(self, runner: CliRunner) -> None:
        """Test init command with all options."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli,
                [
                    "init",
                    "--name",
                    "test-project",
                    "--provider",
                    "anthropic",
                    "--mcp",
                ],
            )

            assert result.exit_code == 0
            assert "Created agent project structure" in result.output
            assert Path("test-project").exists()
            assert Path("test-project/agents").exists()
            assert Path("test-project/skills").exists()
            assert Path("test-project/manifests").exists()

    def test_init_creates_directory_structure(self, runner: CliRunner) -> None:
        """Test that init creates correct directory structure."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli, ["init", "--name", "my-agent", "--provider", "anthropic"]
            )

            assert result.exit_code == 0

            # Verify directories
            project_path = Path("my-agent")
            assert project_path.exists()
            assert (project_path / "agents").exists()
            assert (project_path / "skills").exists()
            assert (project_path / "manifests").exists()
            assert (project_path / "schemas").exists()
            assert (project_path / "tests").exists()
            assert (project_path / ".kautilya").exists()

            # Verify files
            assert (project_path / "README.md").exists()
            assert (project_path / "requirements.txt").exists()
            assert (project_path / ".gitignore").exists()


class TestAgentCommand:
    """Test suite for /agent command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_agent_new_command(self, runner: CliRunner) -> None:
        """Test creating a new agent."""
        with runner.isolated_filesystem():
            # Create agents directory first
            Path("agents").mkdir()

            result = runner.invoke(
                cli,
                [
                    "agent",
                    "new",
                    "research-agent",
                    "--role",
                    "research",
                    "--capabilities",
                    "web_search,summarize",
                    "--output-type",
                    "research_snippet",
                ],
            )

            assert result.exit_code == 0
            assert "Agent Created" in result.output

            # Verify agent directory structure
            agent_dir = Path("agents/research-agent")
            assert agent_dir.exists()
            assert (agent_dir / "config.yaml").exists()
            assert (agent_dir / "capabilities.json").exists()
            assert (agent_dir / "prompts" / "system.txt").exists()

    def test_agent_config_content(self, runner: CliRunner) -> None:
        """Test agent configuration content."""
        with runner.isolated_filesystem():
            Path("agents").mkdir()

            result = runner.invoke(
                cli,
                [
                    "agent",
                    "new",
                    "test-agent",
                    "--role",
                    "code",
                    "--capabilities",
                    "code_generation",
                    "--output-type",
                    "code_patch",
                ],
            )

            assert result.exit_code == 0

            # Read and verify config
            import yaml

            config_path = Path("agents/test-agent/config.yaml")
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert config["name"] == "test-agent"
            assert config["role"] == "code"
            assert "code_generation" in config["capabilities"]
            assert config["output_type"] == "code_patch"


class TestSkillCommand:
    """Test suite for /skill command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_skill_new_command(self, runner: CliRunner) -> None:
        """Test creating a new skill."""
        with runner.isolated_filesystem():
            Path("skills").mkdir()

            result = runner.invoke(
                cli,
                ["skill", "new", "test-skill", "--description", "Test skill description"],
            )

            assert result.exit_code == 0
            assert "Skill Created" in result.output

            # Verify skill directory structure
            skill_dir = Path("skills/test_skill")
            assert skill_dir.exists()
            assert (skill_dir / "skill.yaml").exists()
            assert (skill_dir / "schema.json").exists()
            assert (skill_dir / "handler.py").exists()
            assert (skill_dir / "test_handler.py").exists()

    def test_skill_handler_content(self, runner: CliRunner) -> None:
        """Test skill handler code generation."""
        with runner.isolated_filesystem():
            Path("skills").mkdir()

            result = runner.invoke(
                cli, ["skill", "new", "extract-entities", "--description", "Extract entities"]
            )

            assert result.exit_code == 0

            # Verify handler file exists and has content
            handler_path = Path("skills/extract_entities/handler.py")
            assert handler_path.exists()

            content = handler_path.read_text()
            assert "def extract_entities" in content
            assert "Extract entities" in content


class TestLLMCommand:
    """Test suite for /llm command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_llm_list_command(self, runner: CliRunner) -> None:
        """Test listing LLM providers."""
        result = runner.invoke(cli, ["llm", "list"])

        assert result.exit_code == 0
        assert "anthropic" in result.output.lower()
        assert "openai" in result.output.lower()
        assert "claude-sonnet" in result.output.lower() or "gpt-4" in result.output.lower()

    def test_llm_config_command(self, runner: CliRunner) -> None:
        """Test LLM configuration."""
        with runner.isolated_filesystem():
            Path(".kautilya").mkdir()

            result = runner.invoke(
                cli,
                [
                    "llm",
                    "config",
                    "--provider",
                    "anthropic",
                    "--model",
                    "claude-sonnet-4-20250514",
                    "--api-key-env",
                    "ANTHROPIC_API_KEY",
                ],
            )

            assert result.exit_code == 0
            assert "LLM Configured" in result.output

            # Verify config file created
            assert Path(".kautilya/llm.yaml").exists()


class TestManifestCommand:
    """Test suite for /manifest command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_manifest_validate_missing_file(self, runner: CliRunner) -> None:
        """Test validating non-existent manifest."""
        result = runner.invoke(cli, ["manifest", "validate", "nonexistent.yaml"])

        assert result.exit_code != 0 or "not found" in result.output.lower()


class TestRuntimeCommands:
    """Test suite for runtime commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    @patch("kautilya.commands.runtime.subprocess.run")
    def test_run_command(self, mock_run: MagicMock, runner: CliRunner) -> None:
        """Test run command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Services started")

        with runner.isolated_filesystem():
            # Create a dummy docker-compose.yml
            Path("docker-compose.yml").write_text("version: '3'")

            result = runner.invoke(cli, ["run"])

            # May fail if docker-compose not available, but should handle gracefully
            assert "Services" in result.output or "docker-compose" in result.output.lower()


class TestConfigManagement:
    """Test suite for configuration management."""

    def test_load_config_default(self) -> None:
        """Test loading default configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".kautilya"
            config = load_config(str(config_dir))

            assert config.default_provider == "anthropic"
            assert config.memory_backend == "redis"
            assert config.vector_db == "chroma"

    def test_save_and_load_config(self) -> None:
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".kautilya"

            # Create config
            from kautilya.config import ProjectConfig

            config = Config(
                project=ProjectConfig(name="test-project", version="1.0.0"),
                default_provider="openai",
            )

            # Save
            save_config(config, str(config_dir))

            # Load
            loaded = load_config(str(config_dir))

            assert loaded.project is not None
            assert loaded.project.name == "test-project"
            assert loaded.default_provider == "openai"


class TestInteractiveMode:
    """Test suite for interactive mode."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_interactive_mode_help(self, runner: CliRunner) -> None:
        """Test interactive mode help command."""
        # Interactive mode is hard to test fully, but we can test components
        from kautilya.interactive import InteractiveMode
        from kautilya.config import Config

        mode = InteractiveMode(".kautilya", Config())

        # Verify commands are registered
        assert "/help" in mode.commands
        assert "/init" in mode.commands
        assert "/agent" in mode.commands
        assert "/skill" in mode.commands
        assert "/llm" in mode.commands


class TestCLIErrorHandling:
    """Test suite for CLI error handling."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_agent_new_missing_directory(self, runner: CliRunner) -> None:
        """Test creating agent without agents directory."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli, ["agent", "new", "test-agent", "--role", "research"]
            )

            # Should either create directory or warn about missing directory
            assert result.exit_code == 0 or "agents" in result.output.lower()

    def test_invalid_command(self, runner: CliRunner) -> None:
        """Test invalid command."""
        result = runner.invoke(cli, ["invalid-command"])

        assert result.exit_code != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=kautilya", "--cov-report=html"])
