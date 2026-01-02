"""
Tests for CapabilityRegistry - Dynamic capability discovery.

Module: tests/test_agent/test_capability_registry.py
"""

import tempfile
from pathlib import Path

import pytest

from kautilya.agent.capability_registry import Capability, CapabilityRegistry


class TestCapability:
    """Tests for Capability dataclass."""

    def test_capability_creation(self) -> None:
        """Test creating a capability."""
        cap = Capability(
            name="test_skill",
            type="skill",
            description="A test skill",
            parameters={"input": "string"},
            when_to_use="Use for testing",
        )

        assert cap.name == "test_skill"
        assert cap.type == "skill"
        assert cap.description == "A test skill"

    def test_capability_to_dict(self) -> None:
        """Test converting capability to dictionary."""
        cap = Capability(
            name="test_skill",
            type="skill",
            description="A test skill",
        )

        d = cap.to_dict()
        assert d["name"] == "test_skill"
        assert d["type"] == "skill"

    def test_capability_to_openai_tool(self) -> None:
        """Test converting capability to OpenAI tool format."""
        cap = Capability(
            name="search_web",
            type="tool",
            description="Search the web for information",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum results"},
            },
        )

        tool = cap.to_openai_tool()

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "search_web"
        assert "Search the web" in tool["function"]["description"]
        assert "properties" in tool["function"]["parameters"]


class TestCapabilityRegistry:
    """Tests for CapabilityRegistry."""

    @pytest.fixture
    def registry(self) -> CapabilityRegistry:
        """Create a CapabilityRegistry instance."""
        return CapabilityRegistry()

    @pytest.fixture
    def temp_skills_dir(self):
        """Create a temporary directory with test skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # Create a test skill
            test_skill = skills_dir / "test_skill"
            test_skill.mkdir()

            # Create skill.yaml
            (test_skill / "skill.yaml").write_text("""
name: test_skill
version: 1.0.0
description: A test skill for unit testing
handler: handler.main
when_to_use: Use this for testing purposes
tags:
  - test
  - example
""")

            # Create schema.json
            (test_skill / "schema.json").write_text("""{
    "input": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Input text"}
        },
        "required": ["text"]
    },
    "output": {
        "type": "object",
        "properties": {
            "result": {"type": "string"}
        }
    }
}""")

            # Create handler.py
            (test_skill / "handler.py").write_text("""
def main(text: str) -> dict:
    return {"result": f"Processed: {text}"}
""")

            yield skills_dir

    def test_register_capability(self, registry: CapabilityRegistry) -> None:
        """Test registering a capability."""
        cap = Capability(
            name="custom_tool",
            type="tool",
            description="A custom tool",
        )

        registry.register(cap)

        assert registry.get("custom_tool") is not None
        assert registry.get("custom_tool").name == "custom_tool"

    def test_get_nonexistent_capability(self, registry: CapabilityRegistry) -> None:
        """Test getting a non-existent capability."""
        cap = registry.get("nonexistent")
        assert cap is None

    def test_get_all_capabilities(self, registry: CapabilityRegistry) -> None:
        """Test getting all capabilities."""
        cap1 = Capability(name="cap1", type="skill", description="First")
        cap2 = Capability(name="cap2", type="tool", description="Second")

        registry.register(cap1)
        registry.register(cap2)

        all_caps = registry.get_all()
        assert len(all_caps) >= 2

        names = [c.name for c in all_caps]
        assert "cap1" in names
        assert "cap2" in names

    def test_get_by_type(self, registry: CapabilityRegistry) -> None:
        """Test getting capabilities by type."""
        cap1 = Capability(name="skill1", type="skill", description="A skill")
        cap2 = Capability(name="tool1", type="tool", description="A tool")
        cap3 = Capability(name="skill2", type="skill", description="Another skill")

        registry.register(cap1)
        registry.register(cap2)
        registry.register(cap3)

        skills = registry.get_by_type("skill")
        assert len(skills) >= 2
        assert all(c.type == "skill" for c in skills)

    def test_match_capabilities(self, registry: CapabilityRegistry) -> None:
        """Test matching capabilities to a task description."""
        cap1 = Capability(
            name="document_qa",
            type="skill",
            description="Extract and answer questions from documents",
            tags=["document", "qa", "extraction"],
        )
        cap2 = Capability(
            name="web_search",
            type="tool",
            description="Search the web for information",
            tags=["search", "web"],
        )

        registry.register(cap1)
        registry.register(cap2)

        # Should match document_qa
        matches = registry.match_capabilities("Extract data from the PDF document")
        assert any(m.name == "document_qa" for m in matches)

        # Should match web_search
        matches = registry.match_capabilities("Search the web for latest news")
        assert any(m.name == "web_search" for m in matches)

    def test_discover_skills(self, temp_skills_dir: Path) -> None:
        """Test discovering skills from directory."""
        registry = CapabilityRegistry(skills_dir=temp_skills_dir)
        registry.discover_all()

        # Should find the test skill
        test_skill = registry.get("test_skill")
        assert test_skill is not None
        assert test_skill.type == "skill"
        assert "test skill" in test_skill.description.lower()

    def test_to_openai_tools(self, registry: CapabilityRegistry) -> None:
        """Test converting all capabilities to OpenAI tools format."""
        cap1 = Capability(
            name="skill1",
            type="skill",
            description="First skill",
            parameters={"input": {"type": "string"}},
        )
        cap2 = Capability(
            name="tool1",
            type="tool",
            description="First tool",
            parameters={"query": {"type": "string"}},
        )

        registry.register(cap1)
        registry.register(cap2)

        tools = registry.to_openai_tools()

        assert len(tools) >= 2
        assert all(t["type"] == "function" for t in tools)

    def test_clear_capabilities(self, registry: CapabilityRegistry) -> None:
        """Test clearing all capabilities."""
        cap = Capability(name="test", type="skill", description="Test")
        registry.register(cap)

        assert len(registry.get_all()) > 0

        registry.clear()
        assert len(registry.get_all()) == 0
