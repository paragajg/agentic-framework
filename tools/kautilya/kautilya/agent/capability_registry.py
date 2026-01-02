"""
Dynamic Capability Registry for Kautilya.

Discovers and manages all available capabilities:
- Skills (document_qa, deep_research, etc.)
- Tools (file_read, bash_exec, etc.)
- MCP Servers (external integrations)

All capabilities are discovered at runtime, no hardcoded mappings.
"""

import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Capability:
    """Represents a single capability (skill, tool, or MCP server)."""

    name: str
    type: str  # "skill", "tool", "mcp"
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)  # JSON Schema for inputs
    when_to_use: str = ""  # Semantic description of when to use
    handler: Optional[Callable] = None  # Callable handler for skills
    examples: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    requires_approval: bool = False
    safety_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "parameters": self.parameters,
            "when_to_use": self.when_to_use,
            "tags": self.tags,
        }

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI function tool format."""
        # Ensure parameters are in proper JSON Schema format
        params = self.parameters
        if params and "properties" not in params and "type" not in params:
            # Wrap raw parameter definitions in JSON Schema object format
            params = {
                "type": "object",
                "properties": params,
                "required": list(params.keys()),
            }
        elif not params:
            params = {"type": "object", "properties": {}}

        return {
            "type": "function",
            "function": {
                "name": f"{self.type}_{self.name}" if self.type == "skill" else self.name,
                "description": f"{self.description}\n\nWhen to use: {self.when_to_use}",
                "parameters": params,
            },
        }

    def matches_task(self, task_description: str) -> float:
        """
        Calculate how well this capability matches a task.

        Returns:
            Score from 0.0 to 1.0
        """
        task_lower = task_description.lower()
        score = 0.0

        # Check tags
        for tag in self.tags:
            if tag.lower() in task_lower:
                score += 0.2

        # Check description keywords
        desc_words = set(self.description.lower().split())
        task_words = set(task_lower.split())
        overlap = len(desc_words & task_words)
        if overlap > 0:
            score += min(0.3, overlap * 0.1)

        # Check when_to_use
        when_words = set(self.when_to_use.lower().split())
        when_overlap = len(when_words & task_words)
        if when_overlap > 0:
            score += min(0.3, when_overlap * 0.1)

        # Check examples
        for example in self.examples:
            if any(word in task_lower for word in example.lower().split()):
                score += 0.1

        return min(1.0, score)


class CapabilityRegistry:
    """
    Dynamic registry for all capabilities.

    Discovers:
    - Skills from skills/ directory
    - Tools from KAUTILYA_TOOLS
    - MCP servers from configuration
    """

    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        tools_list: Optional[List[Dict]] = None,
        mcp_config_path: Optional[Path] = None,
    ):
        """
        Initialize capability registry.

        Args:
            skills_dir: Directory containing skills
            tools_list: List of tool definitions (KAUTILYA_TOOLS format)
            mcp_config_path: Path to MCP configuration
        """
        self.skills_dir = skills_dir or self._detect_skills_dir()
        self.tools_list = tools_list
        self.mcp_config_path = mcp_config_path

        self._capabilities: Dict[str, Capability] = {}
        self._skills_cache: Dict[str, Any] = {}
        self._cleared: bool = False  # Track if explicitly cleared

    def _detect_skills_dir(self) -> Optional[Path]:
        """Detect skills directory from project structure."""
        # Try relative to this file
        current = Path(__file__).resolve()

        # Navigate up to find code-exec/skills
        for _ in range(10):
            skills_path = current / "code-exec" / "skills"
            if skills_path.exists():
                return skills_path

            skills_path = current / "skills"
            if skills_path.exists():
                return skills_path

            current = current.parent

        # Try from environment
        env_path = os.getenv("KAUTILYA_SKILLS_DIR")
        if env_path and Path(env_path).exists():
            return Path(env_path)

        return None

    def discover_all(self) -> List[Capability]:
        """
        Discover all capabilities from all sources.

        Returns:
            List of all discovered capabilities
        """
        self._capabilities.clear()

        # Discover skills
        skill_caps = self._discover_skills()
        for cap in skill_caps:
            self._capabilities[f"skill_{cap.name}"] = cap

        # Discover tools
        tool_caps = self._discover_tools()
        for cap in tool_caps:
            self._capabilities[cap.name] = cap

        # Discover MCP servers
        mcp_caps = self._discover_mcp_servers()
        for cap in mcp_caps:
            self._capabilities[f"mcp_{cap.name}"] = cap

        logger.info(
            f"Discovered {len(self._capabilities)} capabilities: "
            f"{len(skill_caps)} skills, {len(tool_caps)} tools, {len(mcp_caps)} MCP"
        )

        return list(self._capabilities.values())

    def _discover_skills(self) -> List[Capability]:
        """Discover all skills from skills directory."""
        capabilities = []

        if not self.skills_dir or not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return capabilities

        # Scan for skill.yaml files
        for skill_yaml in self.skills_dir.rglob("skill.yaml"):
            try:
                cap = self._load_skill(skill_yaml)
                if cap:
                    capabilities.append(cap)
            except Exception as e:
                logger.warning(f"Failed to load skill from {skill_yaml}: {e}")

        return capabilities

    def _load_skill(self, skill_yaml_path: Path) -> Optional[Capability]:
        """Load a skill from its YAML definition."""
        try:
            with open(skill_yaml_path) as f:
                skill_config = yaml.safe_load(f)

            skill_dir = skill_yaml_path.parent
            skill_name = skill_config.get("name", skill_dir.name)

            # Load schema if available
            schema_path = skill_dir / "schema.json"
            parameters = {"type": "object", "properties": {}, "required": []}
            if schema_path.exists():
                with open(schema_path) as f:
                    schema = json.load(f)
                    parameters = schema.get("input", parameters)

            # Load handler if available
            handler = None
            handler_path = skill_dir / "handler.py"
            if handler_path.exists():
                handler = self._load_handler(handler_path, skill_name)

            # Build when_to_use from description and tags
            description = skill_config.get("description", "")
            tags = skill_config.get("tags", [])
            when_to_use = self._build_when_to_use(skill_name, description, tags)

            # Build examples from SKILL.md if available
            examples = []
            skill_md_path = skill_dir / "SKILL.md"
            if skill_md_path.exists():
                examples = self._extract_examples_from_skillmd(skill_md_path)

            return Capability(
                name=skill_name,
                type="skill",
                description=description,
                parameters=parameters,
                when_to_use=when_to_use,
                handler=handler,
                examples=examples,
                tags=tags,
                requires_approval=skill_config.get("requires_approval", False),
                safety_flags=skill_config.get("safety_flags", []),
            )

        except Exception as e:
            logger.error(f"Error loading skill from {skill_yaml_path}: {e}")
            return None

    def _load_handler(self, handler_path: Path, skill_name: str) -> Optional[Callable]:
        """Load skill handler function."""
        try:
            spec = importlib.util.spec_from_file_location(
                f"skill_{skill_name}_handler",
                handler_path,
            )
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"skill_{skill_name}_handler"] = module
            spec.loader.exec_module(module)

            # Look for handler function with skill name or default names
            handler_names = [skill_name, "handler", "run", "execute", "main"]
            for name in handler_names:
                if hasattr(module, name):
                    handler = getattr(module, name)
                    if callable(handler):
                        self._skills_cache[skill_name] = handler
                        return handler

            return None

        except Exception as e:
            logger.warning(f"Failed to load handler from {handler_path}: {e}")
            return None

    def _build_when_to_use(
        self,
        name: str,
        description: str,
        tags: List[str],
    ) -> str:
        """Build semantic 'when to use' description."""
        # Map skill names to usage scenarios
        skill_usage_map = {
            "document_qa": (
                "Use when user wants to extract information from documents "
                "(PDF, DOCX, XLSX, PPTX), answer questions about file content, "
                "or analyze document data. Look for @file references or mentions "
                "of reading/extracting/analyzing documents."
            ),
            "deep_research": (
                "Use when user wants to research a topic using web search, "
                "find current information, news, or comprehensive analysis. "
                "Look for words like 'research', 'find out', 'latest', 'current'."
            ),
            "text_summarize": (
                "Use when user wants to summarize text, get key points, "
                "or condense long content into shorter form."
            ),
            "extract_entities": (
                "Use when user wants to extract named entities like people, "
                "organizations, locations, dates from text."
            ),
            "code_review": (
                "Use when user wants code reviewed for quality, security, "
                "best practices, or bugs."
            ),
        }

        if name in skill_usage_map:
            return skill_usage_map[name]

        # Build from description and tags
        parts = []
        if description:
            parts.append(f"Use when: {description}")
        if tags:
            parts.append(f"Related to: {', '.join(tags)}")

        return " ".join(parts) if parts else f"Use the {name} skill when needed."

    def _extract_examples_from_skillmd(self, skill_md_path: Path) -> List[str]:
        """Extract usage examples from SKILL.md."""
        examples = []
        try:
            with open(skill_md_path) as f:
                content = f.read()

            # Look for example sections
            import re

            # Find code blocks after "Example" headers
            pattern = r"##.*[Ee]xample.*?\n```.*?\n(.*?)```"
            matches = re.findall(pattern, content, re.DOTALL)
            examples.extend([m.strip()[:200] for m in matches[:3]])

        except Exception:
            pass

        return examples

    def _discover_tools(self) -> List[Capability]:
        """Discover tools from KAUTILYA_TOOLS list."""
        capabilities = []

        # If tools_list not provided, try to import from llm_client
        tools_list = self.tools_list
        if tools_list is None:
            try:
                from ..llm_client import KAUTILYA_TOOLS

                tools_list = KAUTILYA_TOOLS
            except ImportError:
                logger.warning("Could not import KAUTILYA_TOOLS")
                return capabilities

        for tool in tools_list:
            if tool.get("type") != "function":
                continue

            func = tool.get("function", {})
            name = func.get("name", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            # Build when_to_use from tool type
            when_to_use = self._build_tool_when_to_use(name, description)

            # Determine tags from name
            tags = self._infer_tool_tags(name)

            capabilities.append(
                Capability(
                    name=name,
                    type="tool",
                    description=description,
                    parameters=parameters,
                    when_to_use=when_to_use,
                    handler=None,  # Tools are executed by tool_executor
                    tags=tags,
                )
            )

        return capabilities

    def _build_tool_when_to_use(self, name: str, description: str) -> str:
        """Build when_to_use for tools."""
        tool_usage_map = {
            "file_read": "Use to read file contents. Required before editing files.",
            "file_edit": "Use to modify existing files. Always read first.",
            "file_write": "Use to create new files or overwrite existing ones.",
            "file_glob": "Use to find files by pattern (e.g., **/*.py).",
            "file_grep": "Use to search for text patterns in files.",
            "bash_exec": "Use to run shell commands (git, npm, etc.).",
            "python_exec": "Use to execute Python code.",
            "web_search": "Use to search the web for current information.",
            "web_fetch": "Use to fetch content from a URL.",
        }

        return tool_usage_map.get(name, description[:100] if description else "")

    def _infer_tool_tags(self, name: str) -> List[str]:
        """Infer tags from tool name."""
        tag_map = {
            "file": ["file", "filesystem", "io"],
            "bash": ["shell", "command", "terminal"],
            "python": ["code", "execution", "python"],
            "web": ["internet", "search", "http"],
            "mcp": ["external", "integration"],
        }

        tags = []
        for prefix, tool_tags in tag_map.items():
            if name.startswith(prefix):
                tags.extend(tool_tags)

        return tags

    def _discover_mcp_servers(self) -> List[Capability]:
        """Discover MCP servers from configuration."""
        capabilities = []

        # Try to load from MCP config
        if self.mcp_config_path and self.mcp_config_path.exists():
            try:
                with open(self.mcp_config_path) as f:
                    config = yaml.safe_load(f)

                for server in config.get("servers", []):
                    for tool in server.get("tools", []):
                        capabilities.append(
                            Capability(
                                name=f"{server['name']}_{tool['name']}",
                                type="mcp",
                                description=tool.get("description", ""),
                                parameters=tool.get("parameters", {}),
                                when_to_use=tool.get("when_to_use", ""),
                                tags=["mcp", server["name"]],
                            )
                        )
            except Exception as e:
                logger.warning(f"Failed to load MCP config: {e}")

        return capabilities

    def get_capability(self, name: str) -> Optional[Capability]:
        """Get a capability by name."""
        # Ensure capabilities are discovered
        if not self._capabilities:
            self.discover_all()

        return self._capabilities.get(name)

    def get(self, name: str) -> Optional[Capability]:
        """Get a capability by name (alias for get_capability)."""
        # Try direct name first
        cap = self.get_capability(name)
        if cap:
            return cap

        # Try with skill_ prefix
        cap = self.get_capability(f"skill_{name}")
        if cap:
            return cap

        # Try without skill_ prefix
        if name.startswith("skill_"):
            cap = self.get_capability(name[6:])
            if cap:
                return cap

        return None

    def register(self, capability: Capability) -> None:
        """Register a new capability."""
        key = capability.name
        if capability.type == "skill" and not key.startswith("skill_"):
            key = f"skill_{key}"
        self._capabilities[key] = capability

    def clear(self) -> None:
        """Clear all registered capabilities."""
        self._capabilities.clear()
        self._skills_cache.clear()
        self._cleared = True  # Prevent auto-discovery after clear

    def get_all(self) -> List[Capability]:
        """Get all capabilities (alias for get_all_capabilities)."""
        return self.get_all_capabilities()

    def get_by_type(self, cap_type: str) -> List[Capability]:
        """Get all capabilities of a specific type."""
        if not self._capabilities and not self._cleared:
            self.discover_all()
        return [c for c in self._capabilities.values() if c.type == cap_type]

    def get_all_capabilities(self) -> List[Capability]:
        """Get all discovered capabilities."""
        if not self._capabilities and not self._cleared:
            self.discover_all()
        return list(self._capabilities.values())

    def get_skills(self) -> List[Capability]:
        """Get all skill capabilities."""
        return [c for c in self.get_all_capabilities() if c.type == "skill"]

    def get_tools(self) -> List[Capability]:
        """Get all tool capabilities."""
        return [c for c in self.get_all_capabilities() if c.type == "tool"]

    def match_capabilities(
        self,
        task_description: str,
        min_score: float = 0.2,
        max_results: int = 5,
    ) -> List[Capability]:
        """
        Find capabilities that match a task description.

        Args:
            task_description: Description of the task
            min_score: Minimum match score (0.0-1.0)
            max_results: Maximum number of results

        Returns:
            List of matching capabilities, sorted by relevance
        """
        if not self._capabilities:
            self.discover_all()

        scored = []
        for cap in self._capabilities.values():
            score = cap.matches_task(task_description)
            if score >= min_score:
                scored.append((score, cap))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return [cap for _, cap in scored[:max_results]]

    def execute_skill(
        self,
        skill_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a skill by name.

        Args:
            skill_name: Name of the skill
            inputs: Input parameters

        Returns:
            Skill execution result
        """
        # Get capability
        cap_name = f"skill_{skill_name}" if not skill_name.startswith("skill_") else skill_name
        cap = self.get_capability(cap_name)

        if not cap:
            return {"success": False, "error": f"Skill not found: {skill_name}"}

        if not cap.handler:
            return {"success": False, "error": f"Skill has no handler: {skill_name}"}

        try:
            result = cap.handler(**inputs)
            return result if isinstance(result, dict) else {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert all capabilities to OpenAI tool format."""
        if not self._capabilities:
            self.discover_all()

        return [cap.to_openai_tool() for cap in self._capabilities.values()]

    def format_capabilities_for_prompt(self) -> str:
        """Format capabilities for inclusion in system prompt."""
        if not self._capabilities:
            self.discover_all()

        lines = ["# Available Capabilities\n"]

        # Group by type
        by_type: Dict[str, List[Capability]] = {}
        for cap in self._capabilities.values():
            by_type.setdefault(cap.type, []).append(cap)

        for cap_type, caps in by_type.items():
            lines.append(f"\n## {cap_type.title()}s\n")
            for cap in caps:
                lines.append(f"### {cap.name}")
                lines.append(f"- Description: {cap.description}")
                lines.append(f"- When to use: {cap.when_to_use}")
                if cap.tags:
                    lines.append(f"- Tags: {', '.join(cap.tags)}")
                lines.append("")

        return "\n".join(lines)
