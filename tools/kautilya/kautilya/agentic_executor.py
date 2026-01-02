"""
Agentic Executor for Kautilya Interactive Mode.

This module provides a bridge between the AgentCore (intelligent skill-based execution)
and the InteractiveMode (streaming UI). It enables the CLI to use the full power of
the agentic framework including:
- Two-stage intelligent skill selection
- document_qa for PDF/document extraction
- deep_research for web research
- ReAct loop with proper reasoning

Module: kautilya/agentic_executor.py
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class AgenticResult:
    """Result from agentic execution with streaming-compatible output."""

    success: bool
    response: str
    iterations: int = 0
    tools_used: List[str] = field(default_factory=list)
    skills_used: List[str] = field(default_factory=list)
    files_resolved: List[str] = field(default_factory=list)
    duration: float = 0.0
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None  # Token usage


class AgenticExecutor:
    """
    Bridge between AgentCore and InteractiveMode.

    Provides a streaming-compatible interface that:
    1. Uses AgentCore for intelligent skill selection
    2. Emits progress markers for iteration display
    3. Returns results compatible with existing UI

    Usage:
        executor = AgenticExecutor(config_dir=".kautilya")
        for chunk in executor.execute("Extract ESG KPIs from report.pdf"):
            print(chunk, end="")
    """

    def __init__(
        self,
        config_dir: str = ".kautilya",
        skills_dir: Optional[Path] = None,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize the agentic executor.

        Args:
            config_dir: Configuration directory
            skills_dir: Directory containing skills (auto-detected if None)
            max_iterations: Maximum ReAct iterations
            verbose: Enable verbose logging
        """
        self.config_dir = config_dir
        self.verbose = verbose
        self.max_iterations = max_iterations or int(
            os.getenv("KAUTILYA_MAX_ITERATIONS", "5")
        )

        # Auto-detect skills directory
        if skills_dir is None:
            skills_dir = self._find_skills_dir()
        self.skills_dir = skills_dir

        # Lazy-loaded components
        self._agent_core = None
        self._llm_client = None
        self._tool_executor = None
        self._initialized = False

    def _find_skills_dir(self) -> Optional[Path]:
        """Find the skills directory by searching common locations."""
        # Common locations to search
        search_paths = [
            Path.cwd() / "code-exec" / "skills",
            Path.cwd() / "skills",
            Path(__file__).parent.parent.parent.parent / "code-exec" / "skills",
            Path(__file__).parent.parent / "skills",
        ]

        for path in search_paths:
            if path.exists() and path.is_dir():
                logger.info(f"Found skills directory: {path}")
                return path

        logger.warning("Could not find skills directory")
        return None

    def _ensure_initialized(self) -> bool:
        """Ensure all components are initialized."""
        if self._initialized:
            return True

        try:
            # Import components
            from .llm_client import KautilyaLLMClient
            from .tool_executor import ToolExecutor
            from .agent.core import AgentCore

            # Initialize LLM client
            self._llm_client = KautilyaLLMClient(
                config_dir=self.config_dir,
                max_iterations=self.max_iterations,
            )

            # Initialize tool executor
            self._tool_executor = ToolExecutor(config_dir=self.config_dir)

            # Initialize AgentCore with components
            self._agent_core = AgentCore(
                llm_client=self._create_llm_adapter(),
                tool_executor=self._tool_executor,
                skills_dir=self.skills_dir,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
            )

            self._initialized = True
            logger.info(
                f"AgenticExecutor initialized with {len(self._agent_core.get_capabilities())} capabilities"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AgenticExecutor: {e}")
            return False

    def _create_llm_adapter(self) -> "LLMClientAdapter":
        """Create an adapter that wraps KautilyaLLMClient for AgentCore."""
        return LLMClientAdapter(self._llm_client, self._tool_executor)

    def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        attached_files: Optional[Dict[str, str]] = None,
    ) -> Tuple[AgenticResult, List[str]]:
        """
        Execute a query using the agentic framework.

        Args:
            query: User's query/request
            context: Additional context
            attached_files: Dict of {path: content} for attached files

        Returns:
            Tuple of (AgenticResult, list of progress messages)
        """
        progress_messages = []

        if not self._ensure_initialized():
            return AgenticResult(
                success=False,
                response="Failed to initialize agentic executor",
                error="Initialization failed",
            ), ["[Error] Failed to initialize agentic executor"]

        start_time = time.time()
        context = context or {}

        # Add attached files to context
        if attached_files:
            context["attached_files"] = attached_files

        # Show which skills were selected for this query
        relevant_caps = self._agent_core.capability_registry.get_relevant_capabilities(
            query, max_results=5
        )
        if relevant_caps:
            skills_text = ", ".join(c.name for c in relevant_caps[:3])
            progress_messages.append(f"[Skills selected: {skills_text}]")

            # Show primary skill
            primary_skill = relevant_caps[0]
            progress_messages.append(f"> Executing: {primary_skill.name}")

            # Show queued skills
            for cap in relevant_caps[1:3]:
                progress_messages.append(f"> Queued: {cap.name}")

        # Run the async process in sync context
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the actual processing
            result = loop.run_until_complete(
                self._agent_core.process(query, context)
            )

            # Build final response
            duration = time.time() - start_time

            return AgenticResult(
                success=result.success,
                response=result.response,
                iterations=len(result.react_trace.steps) if result.react_trace else 0,
                tools_used=result.capabilities_used,
                skills_used=result.capabilities_used,
                files_resolved=[str(f.path) for f in result.files_resolved],
                duration=duration,
                error=result.error,
            ), progress_messages

        except Exception as e:
            logger.error(f"Agentic execution failed: {e}")
            import traceback
            traceback.print_exc()
            return AgenticResult(
                success=False,
                response=f"Execution failed: {str(e)}",
                duration=time.time() - start_time,
                error=str(e),
            ), progress_messages + [f"[Error] {str(e)}"]

    def get_available_skills(self) -> List[str]:
        """Get list of available skill names."""
        if not self._ensure_initialized():
            return []
        return [c.name for c in self._agent_core.get_capabilities()]

    def get_skill_info(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific skill."""
        if not self._ensure_initialized():
            return None

        cap = self._agent_core.get_capability(skill_name)
        if not cap:
            return None

        return {
            "name": cap.name,
            "description": cap.description,
            "when_to_use": cap.when_to_use,
            "when_not_to_use": cap.when_not_to_use,
            "category": cap.category,
            "tags": cap.tags,
        }


class LLMClientAdapter:
    """
    Adapter that wraps KautilyaLLMClient for use with AgentCore.

    AgentCore expects an LLM client with simple async chat() method,
    while KautilyaLLMClient has a more complex streaming interface.
    """

    def __init__(self, llm_client: Any, tool_executor: Any):
        """
        Initialize the adapter.

        Args:
            llm_client: KautilyaLLMClient instance
            tool_executor: ToolExecutor instance
        """
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.history = llm_client.history if hasattr(llm_client, "history") else None

    async def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Send a chat message and get response.

        Args:
            message: User message
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters

        Returns:
            Response text
        """
        try:
            # Use the LLM client's chat method
            response_text = ""

            # KautilyaLLMClient.chat() is a generator for streaming
            for chunk in self.llm_client.chat(
                message,
                tool_executor=self.tool_executor,
                stream=False,  # Non-streaming for AgentCore
            ):
                if isinstance(chunk, str):
                    response_text += chunk

            return response_text

        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            raise

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """
        Complete a prompt (simpler interface).

        Args:
            prompt: Prompt to complete
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            Completion text
        """
        return await self.chat(prompt, **kwargs)

    def clear_history(self) -> None:
        """Clear conversation history."""
        if hasattr(self.llm_client, "clear_history"):
            self.llm_client.clear_history()


def create_agentic_executor(
    config_dir: str = ".kautilya",
    verbose: bool = False,
) -> AgenticExecutor:
    """
    Factory function to create an AgenticExecutor.

    Args:
        config_dir: Configuration directory
        verbose: Enable verbose logging

    Returns:
        Configured AgenticExecutor instance
    """
    return AgenticExecutor(
        config_dir=config_dir,
        verbose=verbose,
    )
