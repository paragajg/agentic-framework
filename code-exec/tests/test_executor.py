"""
Tests for Code Executor Service.
Module: code-exec/tests/test_executor.py
"""

import json
from typing import Any, Dict

import pytest
from httpx import AsyncClient

from code_exec.service.config import CodeExecSettings
from code_exec.service.executor import SandboxedExecutor
from code_exec.service.main import app
from code_exec.service.models import ExecutionRequest, SafetyFlag
from code_exec.service.registry import SkillRegistry


@pytest.fixture
def settings() -> CodeExecSettings:
    """Create test settings."""
    return CodeExecSettings(
        skills_directory="/Users/paragpradhan/Projects/Agent framework/agent-framework/code-exec/skills",
        max_execution_time=5,
        debug=True,
    )


@pytest.fixture
async def registry(settings: CodeExecSettings) -> SkillRegistry:
    """Create and load skill registry."""
    reg = SkillRegistry(settings.skills_directory)
    await reg.load_all_skills()
    return reg


@pytest.fixture
async def executor(registry: SkillRegistry) -> SandboxedExecutor:
    """Create executor instance."""
    return SandboxedExecutor(registry)


@pytest.mark.asyncio
async def test_registry_loads_skills(registry: SkillRegistry) -> None:
    """Test that registry loads all skills successfully."""
    assert len(registry.skills) >= 4, "Should load at least 4 sample skills"

    # Check specific skills
    assert "text_summarize" in registry.skills
    assert "extract_entities" in registry.skills
    assert "embed_text" in registry.skills
    assert "compact_memory" in registry.skills


@pytest.mark.asyncio
async def test_text_summarize_skill(executor: SandboxedExecutor) -> None:
    """Test text summarization skill execution."""
    request = ExecutionRequest(
        skill="text_summarize",
        args={
            "text": "This is a long text that needs summarization. "
            "It contains multiple sentences. "
            "Each sentence provides different information. "
            "The summarization should extract the most important sentences. "
            "This helps reduce token count while preserving meaning.",
            "max_sentences": 2,
            "style": "concise",
        },
        actor_id="test_actor",
        actor_type="test",
    )

    result = await executor.execute(request)

    assert result.success is True
    assert result.result is not None
    assert "summary" in result.result
    assert "compression_ratio" in result.result
    assert result.inputs_hash != ""
    assert result.outputs_hash != ""
    assert len(result.logs) > 0


@pytest.mark.asyncio
async def test_extract_entities_skill(executor: SandboxedExecutor) -> None:
    """Test entity extraction skill execution."""
    request = ExecutionRequest(
        skill="extract_entities",
        args={
            "text": "Contact John Doe at john@example.com or call +1-555-123-4567. "
            "Visit https://example.com for more information.",
            "entity_types": ["EMAIL", "PHONE", "URL"],
            "min_confidence": 0.5,
        },
    )

    result = await executor.execute(request)

    assert result.success is True
    assert result.result is not None
    assert "entities" in result.result
    assert "total_entities" in result.result
    assert result.result["total_entities"] >= 3  # Should find email, phone, URL


@pytest.mark.asyncio
async def test_embed_text_skill(executor: SandboxedExecutor) -> None:
    """Test text embedding skill execution."""
    request = ExecutionRequest(
        skill="embed_text",
        args={
            "text": "This is a test sentence for embedding generation.",
            "model": "all-MiniLM-L6-v2",
            "normalize": True,
        },
    )

    result = await executor.execute(request)

    assert result.success is True
    assert result.result is not None
    assert "embedding" in result.result
    assert "dimension" in result.result
    assert isinstance(result.result["embedding"], list)
    assert len(result.result["embedding"]) > 0


@pytest.mark.asyncio
async def test_compact_memory_skill(executor: SandboxedExecutor) -> None:
    """Test memory compaction skill execution."""
    artifacts = [
        {
            "id": "artifact_1",
            "type": "research_snippet",
            "content": {"text": "Sample research finding", "confidence": 0.9},
            "created_at": "2025-12-28T12:00:00Z",
        },
        {
            "id": "artifact_2",
            "type": "claim_verification",
            "content": {"claim_text": "Test claim", "verdict": "verified", "confidence": 0.95},
            "created_at": "2025-12-28T12:05:00Z",
        },
    ]

    request = ExecutionRequest(
        skill="compact_memory",
        args={
            "artifacts": artifacts,
            "strategy": "summarize",
            "max_tokens": 1000,
        },
    )

    result = await executor.execute(request)

    assert result.success is True
    assert result.result is not None
    assert "compacted_artifacts" in result.result
    assert "compression_ratio" in result.result
    assert result.result["original_count"] == 2


@pytest.mark.asyncio
async def test_input_validation_failure(executor: SandboxedExecutor) -> None:
    """Test that invalid inputs are rejected."""
    request = ExecutionRequest(
        skill="text_summarize",
        args={
            # Missing required 'text' field
            "max_sentences": 5,
        },
    )

    result = await executor.execute(request)

    assert result.success is False
    assert result.provenance.error_message is not None
    assert "validation" in result.provenance.error_message.lower()


@pytest.mark.asyncio
async def test_skill_not_found(executor: SandboxedExecutor) -> None:
    """Test execution with non-existent skill."""
    request = ExecutionRequest(
        skill="nonexistent_skill",
        args={"test": "data"},
    )

    result = await executor.execute(request)

    assert result.success is False
    assert "not found" in result.provenance.error_message.lower()


@pytest.mark.asyncio
async def test_deterministic_hashing(executor: SandboxedExecutor) -> None:
    """Test that hashing is deterministic for same inputs."""
    args = {
        "text": "Test text for hashing",
        "max_sentences": 3,
        "style": "concise",
    }

    request1 = ExecutionRequest(skill="text_summarize", args=args)
    request2 = ExecutionRequest(skill="text_summarize", args=args)

    result1 = await executor.execute(request1)
    result2 = await executor.execute(request2)

    # Same inputs should produce same input hash
    assert result1.inputs_hash == result2.inputs_hash

    # Same handler with same inputs should produce same output hash
    assert result1.outputs_hash == result2.outputs_hash


@pytest.mark.asyncio
async def test_provenance_tracking(executor: SandboxedExecutor) -> None:
    """Test that provenance is correctly tracked."""
    request = ExecutionRequest(
        skill="text_summarize",
        args={"text": "Test text", "max_sentences": 1},
        actor_id="test_actor_123",
        actor_type="subagent",
    )

    result = await executor.execute(request)

    assert result.success is True
    assert result.provenance.skill_name == "text_summarize"
    assert result.provenance.skill_version == "1.0.0"
    assert result.provenance.actor_id == "test_actor_123"
    assert result.provenance.actor_type == "subagent"
    assert result.provenance.inputs_hash != ""
    assert result.provenance.outputs_hash != ""
    assert result.provenance.execution_time_ms > 0


@pytest.mark.asyncio
async def test_execution_logs(executor: SandboxedExecutor) -> None:
    """Test that execution generates appropriate logs."""
    request = ExecutionRequest(
        skill="text_summarize",
        args={"text": "Test text", "max_sentences": 1},
    )

    result = await executor.execute(request)

    assert len(result.logs) > 0

    # Check for expected log messages
    log_messages = [log.message for log in result.logs]
    assert any("Validating inputs" in msg for msg in log_messages)
    assert any("Executing skill" in msg for msg in log_messages)
    assert any("Validating outputs" in msg for msg in log_messages)


@pytest.mark.asyncio
async def test_api_health_endpoint() -> None:
    """Test health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_api_list_skills() -> None:
    """Test skills listing endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/skills/list")

    assert response.status_code == 200
    data = response.json()
    assert "skills" in data
    assert "total" in data
    assert data["total"] >= 4


@pytest.mark.asyncio
async def test_api_get_skill_schema() -> None:
    """Test get skill schema endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/skills/text_summarize/schema")

    assert response.status_code == 200
    data = response.json()
    assert data["skill_name"] == "text_summarize"
    assert "input_schema" in data
    assert "output_schema" in data


@pytest.mark.asyncio
async def test_api_execute_skill() -> None:
    """Test skill execution via API."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/skills/execute",
            json={
                "skill": "text_summarize",
                "args": {
                    "text": "This is a test text for API execution.",
                    "max_sentences": 1,
                },
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "result" in data
    assert "provenance" in data
    assert "execution_id" in data
