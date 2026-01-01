"""
Comprehensive tests for the Orchestrator service.

Tests cover:
- Configuration loading
- Model validation
- Workflow engine functionality
- API endpoints
- Error handling
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from orchestrator.service.config import OrchestratorConfig
from orchestrator.service.main import app
from orchestrator.service.models import (
    ArtifactHandleRequest,
    ArtifactType,
    ClaimVerification,
    CodePatch,
    FileChange,
    ProvenanceRecord,
    ResearchSnippet,
    ResearchSource,
    SafetyClass,
    SubagentRequest,
    SubagentRole,
    SynthesisResult,
    Verdict,
    WorkflowManifest,
    WorkflowStartRequest,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepInput,
    WorkflowStepOutput,
)
from orchestrator.service.workflow_engine import (
    ManifestValidationError,
    WorkflowEngine,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_config() -> OrchestratorConfig:
    """Create test configuration."""
    return OrchestratorConfig(
        host="127.0.0.1",
        port=8000,
        log_level="DEBUG",
        postgres_url="postgresql://test:test@localhost:5432/test",
        redis_url="redis://localhost:6379/1",
        use_chroma_dev=True,
    )


@pytest.fixture
def test_client() -> TestClient:
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_provenance() -> ProvenanceRecord:
    """Create sample provenance record."""
    return ProvenanceRecord(
        actor_id="test-subagent-001",
        actor_type="subagent",
        inputs_hash="abc123",
        outputs_hash="def456",
        tool_ids=["web_search", "summarize"],
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def sample_research_snippet(sample_provenance: ProvenanceRecord) -> ResearchSnippet:
    """Create sample research snippet artifact."""
    return ResearchSnippet(
        created_by="research-agent-001",
        provenance=sample_provenance,
        source=ResearchSource(
            url="https://example.com/article",
            title="Sample Article",
        ),
        text="This is a sample research snippet with detailed information.",
        summary="Sample research about topic X.",
        tags=["research", "topic-x"],
        confidence=0.95,
        safety_class=SafetyClass.PUBLIC,
    )


@pytest.fixture
def sample_claim_verification(sample_provenance: ProvenanceRecord) -> ClaimVerification:
    """Create sample claim verification artifact."""
    return ClaimVerification(
        created_by="verify-agent-001",
        provenance=sample_provenance,
        claim_text="The system supports 10,000 concurrent users.",
        verdict=Verdict.VERIFIED,
        confidence=0.88,
        evidence_refs=["artifact-001", "artifact-002"],
        method="benchmark_analysis",
        verifier="verify-agent-001",
        safety_class=SafetyClass.INTERNAL,
    )


@pytest.fixture
def sample_code_patch(sample_provenance: ProvenanceRecord) -> CodePatch:
    """Create sample code patch artifact."""
    return CodePatch(
        created_by="code-agent-001",
        provenance=sample_provenance,
        repo="example/repo",
        base_commit="abc123def456",
        files_changed=[
            FileChange(
                file_path="src/main.py",
                change_type="modified",
                diff="@@ -1,3 +1,4 @@\n+import logging\n def main():\n     pass",
                lines_added=1,
                lines_removed=0,
            )
        ],
        patch_summary="Add logging import to main.py",
        tests=["pytest tests/test_main.py"],
        confidence=0.92,
        risks=["None identified"],
        authoring_subagent="code-agent-001",
        merge_ready=False,
        safety_class=SafetyClass.INTERNAL,
    )


@pytest.fixture
def sample_workflow_manifest() -> WorkflowManifest:
    """Create sample workflow manifest."""
    return WorkflowManifest(
        name="test-workflow",
        version="1.0.0",
        description="Test workflow for unit tests",
        steps=[
            WorkflowStep(
                id="step-1",
                role=SubagentRole.RESEARCH,
                capabilities=["web_search", "summarize"],
                inputs=[
                    WorkflowStepInput(
                        name="topic",
                        source="user_input",
                        required=True,
                    )
                ],
                outputs=[
                    WorkflowStepOutput(
                        name="research_results",
                        artifact_type=ArtifactType.RESEARCH_SNIPPET,
                    )
                ],
                timeout=300,
            ),
            WorkflowStep(
                id="step-2",
                role=SubagentRole.SYNTHESIS,
                capabilities=["synthesize"],
                inputs=[
                    WorkflowStepInput(
                        name="research_results",
                        source="previous_step",
                        required=True,
                    )
                ],
                outputs=[
                    WorkflowStepOutput(
                        name="final_report",
                        artifact_type=ArtifactType.SYNTHESIS_RESULT,
                    )
                ],
                timeout=180,
            ),
        ],
    )


# ============================================================================
# Configuration Tests
# ============================================================================


def test_config_validation() -> None:
    """Test configuration validation."""
    config = OrchestratorConfig(
        log_level="INFO",
        default_llm_provider="anthropic",
    )
    assert config.log_level == "INFO"
    assert config.default_llm_provider == "anthropic"


def test_config_invalid_log_level() -> None:
    """Test configuration rejects invalid log level."""
    with pytest.raises(ValueError, match="log_level must be one of"):
        OrchestratorConfig(log_level="INVALID")


def test_config_invalid_llm_provider() -> None:
    """Test configuration rejects invalid LLM provider."""
    with pytest.raises(ValueError, match="default_llm_provider must be one of"):
        OrchestratorConfig(default_llm_provider="invalid")


# ============================================================================
# Model Tests
# ============================================================================


def test_research_snippet_validation(sample_research_snippet: ResearchSnippet) -> None:
    """Test research snippet model validation."""
    assert sample_research_snippet.artifact_type == ArtifactType.RESEARCH_SNIPPET
    assert sample_research_snippet.confidence >= 0.0
    assert sample_research_snippet.confidence <= 1.0
    assert len(sample_research_snippet.text) > 0
    assert len(sample_research_snippet.summary) > 0


def test_claim_verification_validation(
    sample_claim_verification: ClaimVerification,
) -> None:
    """Test claim verification model validation."""
    assert sample_claim_verification.artifact_type == ArtifactType.CLAIM_VERIFICATION
    assert sample_claim_verification.verdict in Verdict
    assert sample_claim_verification.confidence >= 0.0
    assert sample_claim_verification.confidence <= 1.0
    assert len(sample_claim_verification.evidence_refs) > 0


def test_code_patch_validation(sample_code_patch: CodePatch) -> None:
    """Test code patch model validation."""
    assert sample_code_patch.artifact_type == ArtifactType.CODE_PATCH
    assert len(sample_code_patch.files_changed) > 0
    assert sample_code_patch.confidence >= 0.0
    assert sample_code_patch.confidence <= 1.0


def test_workflow_manifest_validation(sample_workflow_manifest: WorkflowManifest) -> None:
    """Test workflow manifest model validation."""
    assert len(sample_workflow_manifest.steps) >= 1
    assert sample_workflow_manifest.version == "1.0.0"
    for step in sample_workflow_manifest.steps:
        assert len(step.capabilities) > 0
        assert len(step.outputs) > 0


def test_workflow_manifest_invalid_version() -> None:
    """Test workflow manifest rejects invalid version."""
    with pytest.raises(ValueError, match="Version must follow semver"):
        WorkflowManifest(
            name="test",
            version="invalid",
            steps=[
                WorkflowStep(
                    id="step-1",
                    role=SubagentRole.RESEARCH,
                    capabilities=["test"],
                    inputs=[],
                    outputs=[
                        WorkflowStepOutput(
                            name="output",
                            artifact_type=ArtifactType.RESEARCH_SNIPPET,
                        )
                    ],
                )
            ],
        )


# ============================================================================
# Workflow Engine Tests
# ============================================================================


@pytest.mark.asyncio
async def test_workflow_engine_initialization() -> None:
    """Test workflow engine initialization."""
    engine = WorkflowEngine()
    assert engine is not None
    await engine.close()


@pytest.mark.asyncio
async def test_load_manifest_from_yaml(sample_workflow_manifest: WorkflowManifest) -> None:
    """Test loading manifest from YAML string."""
    yaml_content = """
name: test-workflow
version: 1.0.0
steps:
  - id: step-1
    role: research
    capabilities:
      - web_search
    inputs:
      - name: topic
        source: user_input
    outputs:
      - name: results
        artifact_type: research_snippet
"""
    engine = WorkflowEngine()
    manifest = await engine.load_manifest_from_yaml(yaml_content)
    assert manifest.name == "test-workflow"
    assert len(manifest.steps) == 1
    await engine.close()


@pytest.mark.asyncio
async def test_load_invalid_yaml() -> None:
    """Test loading invalid YAML raises error."""
    invalid_yaml = """
name: test
version: 1.0.0
steps: [invalid: yaml: structure
"""
    engine = WorkflowEngine()
    with pytest.raises(ManifestValidationError):
        await engine.load_manifest_from_yaml(invalid_yaml)
    await engine.close()


@pytest.mark.asyncio
async def test_compute_hash() -> None:
    """Test hash computation for provenance."""
    engine = WorkflowEngine()
    data = {"key": "value", "number": 42}
    hash1 = engine._compute_hash(data)
    hash2 = engine._compute_hash(data)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest length
    await engine.close()


@pytest.mark.asyncio
async def test_create_provenance() -> None:
    """Test provenance record creation."""
    engine = WorkflowEngine()
    inputs = {"query": "test"}
    outputs = {"result": "success"}
    provenance = engine._create_provenance(
        actor_id="test-agent",
        actor_type="subagent",
        inputs=inputs,
        outputs=outputs,
        tool_ids=["tool1", "tool2"],
    )
    assert provenance.actor_id == "test-agent"
    assert provenance.actor_type == "subagent"
    assert len(provenance.tool_ids) == 2
    assert len(provenance.inputs_hash) == 64
    await engine.close()


# ============================================================================
# API Endpoint Tests
# ============================================================================


def test_root_endpoint(test_client: TestClient) -> None:
    """Test root endpoint returns service info."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Lead Agent/Orchestrator"
    assert "version" in data


@pytest.mark.asyncio
async def test_health_check_endpoint() -> None:
    """Test health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "dependencies" in data
        assert "uptime_seconds" in data


def test_workflow_start_missing_manifest(test_client: TestClient) -> None:
    """Test workflow start requires manifest."""
    request = WorkflowStartRequest(user_input={"topic": "test"})
    response = test_client.post("/workflows/start", json=request.model_dump())
    assert response.status_code == 422  # Validation error


def test_artifact_handle_validation(
    test_client: TestClient, sample_research_snippet: ResearchSnippet
) -> None:
    """Test artifact validation endpoint."""
    request = ArtifactHandleRequest(
        artifact_data=sample_research_snippet.model_dump(),
        artifact_type=ArtifactType.RESEARCH_SNIPPET,
        validate_only=True,
    )
    response = test_client.post("/artifact/handle", json=request.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert len(data["validation_errors"]) == 0


def test_artifact_handle_invalid_type(test_client: TestClient) -> None:
    """Test artifact validation with wrong type."""
    request = ArtifactHandleRequest(
        artifact_data={"invalid": "data"},
        artifact_type=ArtifactType.RESEARCH_SNIPPET,
        validate_only=True,
    )
    response = test_client.post("/artifact/handle", json=request.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert len(data["validation_errors"]) > 0


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_workflow_execution_flow(sample_workflow_manifest: WorkflowManifest) -> None:
    """Test complete workflow execution flow."""
    engine = WorkflowEngine()

    with patch.object(engine, "_execute_subagent_step", new_callable=AsyncMock) as mock_exec:
        # Mock subagent responses
        from orchestrator.service.models import SubagentResponse

        mock_exec.return_value = SubagentResponse(
            subagent_id=str(uuid4()),
            workflow_id=str(uuid4()),
            step_id="step-1",
            status="success",
            artifacts=[str(uuid4())],
            execution_time_seconds=1.5,
        )

        user_input = {"topic": "test topic"}
        context = await engine.execute_workflow(sample_workflow_manifest, user_input)

        assert context.status == WorkflowStatus.COMPLETED
        assert context.completed_at is not None
        assert len(context.step_artifacts) == len(sample_workflow_manifest.steps)

    await engine.close()


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_workflow_step_retry_logic(sample_workflow_manifest: WorkflowManifest) -> None:
    """Test workflow step retry on failure."""
    engine = WorkflowEngine()

    with patch.object(engine, "_execute_subagent_step", new_callable=AsyncMock) as mock_exec:
        from orchestrator.service.models import SubagentResponse

        # First two attempts fail, third succeeds
        mock_exec.side_effect = [
            Exception("Temporary failure"),
            Exception("Another failure"),
            SubagentResponse(
                subagent_id=str(uuid4()),
                workflow_id=str(uuid4()),
                step_id="step-1",
                status="success",
                artifacts=[str(uuid4())],
                execution_time_seconds=1.5,
            ),
        ]

        user_input = {"topic": "test"}
        context = await engine.execute_workflow(sample_workflow_manifest, user_input)

        # Should eventually succeed after retries
        assert mock_exec.call_count == 3

    await engine.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
