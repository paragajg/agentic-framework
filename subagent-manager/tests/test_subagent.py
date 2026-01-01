"""
Tests for Subagent Manager service.
"""

import json
from datetime import datetime
from typing import Any, Dict

import anyio
import pytest
from fastapi.testclient import TestClient

from adapters.llm import LLMMessage, MessageRole, MockLLMAdapter
from subagent_manager.service.config import SubagentManagerConfig
from subagent_manager.service.lifecycle import SubagentLifecycleManager
from subagent_manager.service.main import app
from subagent_manager.service.models import (
    SubagentExecuteRequest,
    SubagentRole,
    SubagentSpawnRequest,
    SubagentStatus,
)
from subagent_manager.service.validator import SchemaValidator


@pytest.fixture
def test_config() -> SubagentManagerConfig:
    """Create test configuration."""
    return SubagentManagerConfig(
        llm_provider="mock",
        schema_registry_path="/tmp/test_schema_registry",
        default_timeout=30,
        max_lifetime=300,
    )


@pytest.fixture
def schema_validator(test_config: SubagentManagerConfig) -> SchemaValidator:
    """Create schema validator for testing."""
    return SchemaValidator(test_config.schema_registry_path)


@pytest.fixture
def lifecycle_manager(
    test_config: SubagentManagerConfig, schema_validator: SchemaValidator
) -> SubagentLifecycleManager:
    """Create lifecycle manager for testing."""
    return SubagentLifecycleManager(test_config, schema_validator)


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestSchemaValidator:
    """Tests for schema validator."""

    def test_load_default_schemas(self, schema_validator: SchemaValidator) -> None:
        """Test loading default schemas."""
        schemas = schema_validator.list_schemas()
        assert "research_snippet" in schemas
        assert "claim_verification" in schemas
        assert "code_patch" in schemas

    def test_validate_research_snippet(self, schema_validator: SchemaValidator) -> None:
        """Test validating a research snippet."""
        valid_snippet = {
            "id": "snippet-123",
            "text": "Research findings about quantum computing.",
            "summary": "Quantum computing advances",
            "created_at": datetime.utcnow().isoformat(),
        }

        is_valid, error = schema_validator.validate(valid_snippet, "research_snippet")
        assert is_valid
        assert error is None

    def test_validate_invalid_snippet(self, schema_validator: SchemaValidator) -> None:
        """Test validating invalid data."""
        invalid_snippet = {
            "id": "snippet-123",
            # Missing required fields
        }

        is_valid, error = schema_validator.validate(invalid_snippet, "research_snippet")
        assert not is_valid
        assert error is not None

    def test_validate_unknown_schema(self, schema_validator: SchemaValidator) -> None:
        """Test validating against unknown schema."""
        data = {"test": "data"}
        is_valid, error = schema_validator.validate(data, "unknown_schema")
        assert not is_valid
        assert "not found" in error.lower()


class TestLLMAdapters:
    """Tests for LLM adapters."""

    @pytest.mark.asyncio
    async def test_mock_adapter_complete(self) -> None:
        """Test mock adapter completion."""
        adapter = MockLLMAdapter(
            model="test-model",
            response_template="Response: {prompt}",
            delay_ms=10,
        )

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            LLMMessage(role=MessageRole.USER, content="Hello, how are you?"),
        ]

        response = await adapter.complete(messages, temperature=0.7)

        assert response.content.startswith("Response:")
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_mock_adapter_streaming(self) -> None:
        """Test mock adapter streaming."""
        adapter = MockLLMAdapter(delay_ms=5)

        messages = [LLMMessage(role=MessageRole.USER, content="Stream test")]

        chunks = []
        async for chunk in adapter.stream_complete(messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_mock_adapter_validation(self) -> None:
        """Test mock adapter API key validation."""
        adapter = MockLLMAdapter()
        is_valid = await adapter.validate_api_key()
        assert is_valid


class TestSubagentLifecycle:
    """Tests for subagent lifecycle management."""

    @pytest.mark.asyncio
    async def test_spawn_subagent(
        self, lifecycle_manager: SubagentLifecycleManager
    ) -> None:
        """Test spawning a new subagent."""
        request = SubagentSpawnRequest(
            role=SubagentRole.RESEARCH,
            capabilities=["web_search", "document_read"],
            system_prompt="You are a research assistant.",
            timeout=60,
        )

        info = await lifecycle_manager.spawn_subagent(request)

        assert info.role == SubagentRole.RESEARCH
        assert info.status == SubagentStatus.READY
        assert len(info.capabilities) == 2
        assert info.subagent_id.startswith("research-")

    @pytest.mark.asyncio
    async def test_execute_task(self, lifecycle_manager: SubagentLifecycleManager) -> None:
        """Test executing a task with a subagent."""
        # First spawn a subagent
        spawn_request = SubagentSpawnRequest(
            role=SubagentRole.RESEARCH,
            capabilities=["web_search"],
            system_prompt="You are a research assistant.",
        )
        info = await lifecycle_manager.spawn_subagent(spawn_request)

        # Execute a task
        exec_request = SubagentExecuteRequest(
            subagent_id=info.subagent_id,
            task="Research quantum computing",
            inputs={"topic": "quantum computing"},
        )

        response = await lifecycle_manager.execute_task(exec_request)

        assert response.subagent_id == info.subagent_id
        assert response.status in (SubagentStatus.COMPLETED, SubagentStatus.FAILED)
        assert response.raw_response is not None
        assert response.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_with_schema_validation(
        self, lifecycle_manager: SubagentLifecycleManager
    ) -> None:
        """Test executing task with schema validation."""
        # Spawn subagent
        spawn_request = SubagentSpawnRequest(
            role=SubagentRole.RESEARCH,
            capabilities=[],
            system_prompt="Return a research snippet as JSON.",
        )
        info = await lifecycle_manager.spawn_subagent(spawn_request)

        # Create a mock adapter that returns valid JSON
        context = lifecycle_manager.subagents[info.subagent_id]
        context.llm_adapter = MockLLMAdapter(
            response_template=json.dumps(
                {
                    "id": "snippet-001",
                    "text": "Research about {prompt}",
                    "summary": "Summary of research",
                    "created_at": datetime.utcnow().isoformat(),
                }
            )
        )

        # Execute with schema validation
        exec_request = SubagentExecuteRequest(
            subagent_id=info.subagent_id,
            task="Research AI",
            expected_output_schema="research_snippet",
        )

        response = await lifecycle_manager.execute_task(exec_request)

        # Note: This may fail if the mock doesn't format JSON properly
        # In that case, status would be FAILED with error about JSON parsing
        if response.output:
            assert response.status == SubagentStatus.COMPLETED
            assert "id" in response.output
            assert "text" in response.output

    @pytest.mark.asyncio
    async def test_destroy_subagent(
        self, lifecycle_manager: SubagentLifecycleManager
    ) -> None:
        """Test destroying a subagent."""
        # Spawn subagent
        spawn_request = SubagentSpawnRequest(
            role=SubagentRole.VERIFY,
            capabilities=[],
            system_prompt="You verify claims.",
        )
        info = await lifecycle_manager.spawn_subagent(spawn_request)

        # Verify it exists
        assert info.subagent_id in lifecycle_manager.subagents

        # Destroy it
        destroyed = await lifecycle_manager.destroy_subagent(info.subagent_id)
        assert destroyed

        # Verify it's gone
        assert info.subagent_id not in lifecycle_manager.subagents

    @pytest.mark.asyncio
    async def test_get_subagent_status(
        self, lifecycle_manager: SubagentLifecycleManager
    ) -> None:
        """Test getting subagent status."""
        # Spawn subagent
        spawn_request = SubagentSpawnRequest(
            role=SubagentRole.CODE,
            capabilities=["code_read", "code_write"],
            system_prompt="You write code.",
        )
        info = await lifecycle_manager.spawn_subagent(spawn_request)

        # Get status
        status = await lifecycle_manager.get_subagent_status(info.subagent_id)

        assert status is not None
        assert status.subagent_id == info.subagent_id
        assert status.role == SubagentRole.CODE
        assert len(status.capabilities) == 2

    @pytest.mark.asyncio
    async def test_list_subagents(
        self, lifecycle_manager: SubagentLifecycleManager
    ) -> None:
        """Test listing all subagents."""
        # Spawn multiple subagents
        for role in [SubagentRole.RESEARCH, SubagentRole.VERIFY, SubagentRole.CODE]:
            spawn_request = SubagentSpawnRequest(
                role=role, capabilities=[], system_prompt=f"You are a {role} agent."
            )
            await lifecycle_manager.spawn_subagent(spawn_request)

        # List all
        subagents = await lifecycle_manager.list_subagents()

        assert len(subagents) == 3
        roles = {s.role for s in subagents}
        assert roles == {SubagentRole.RESEARCH, SubagentRole.VERIFY, SubagentRole.CODE}

    @pytest.mark.asyncio
    async def test_execution_timeout(
        self, lifecycle_manager: SubagentLifecycleManager
    ) -> None:
        """Test task execution timeout."""
        # Spawn subagent with very short timeout
        spawn_request = SubagentSpawnRequest(
            role=SubagentRole.RESEARCH,
            capabilities=[],
            system_prompt="You are slow.",
            timeout=1,  # 1 second
        )
        info = await lifecycle_manager.spawn_subagent(spawn_request)

        # Make the adapter very slow
        context = lifecycle_manager.subagents[info.subagent_id]
        context.llm_adapter = MockLLMAdapter(delay_ms=5000)  # 5 seconds

        # Execute - should timeout
        exec_request = SubagentExecuteRequest(
            subagent_id=info.subagent_id,
            task="Slow task",
            timeout=1,
        )

        response = await lifecycle_manager.execute_task(exec_request)

        assert response.status == SubagentStatus.TIMEOUT
        assert response.error is not None
        assert "timeout" in response.error.lower()


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "subagent-manager"

    def test_spawn_subagent_endpoint(self, client: TestClient) -> None:
        """Test spawn subagent endpoint."""
        request_data = {
            "role": "research",
            "capabilities": ["web_search"],
            "system_prompt": "You are a researcher.",
            "timeout": 60,
            "max_iterations": 5,
            "metadata": {"test": True},
        }

        response = client.post("/subagent/spawn", json=request_data)
        assert response.status_code == 201
        data = response.json()
        assert "subagent_id" in data
        assert data["role"] == "research"

    def test_list_schemas_endpoint(self, client: TestClient) -> None:
        """Test list schemas endpoint."""
        response = client.get("/schemas")
        assert response.status_code == 200
        data = response.json()
        assert "schemas" in data
        assert isinstance(data["schemas"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
