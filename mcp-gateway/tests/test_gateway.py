"""
Comprehensive tests for MCP Gateway service.

Tests catalog management, authentication, rate limiting, tool invocation,
PII detection, and provenance logging.
"""

from typing import Any, AsyncGenerator
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import time

# Import service components
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.main import app
from service.models import (
    MCPServerRegistration,
    ToolSchema,
    ToolParameter,
    ToolClassification,
    AuthFlow,
    RateLimitConfig,
    ToolInvocationRequest,
    RuntimeMode,
    EphemeralTokenRequest,
    TokenScope,
)
from service.catalog import get_catalog
from service.auth import get_token_manager
from service.rate_limit import get_rate_limiter
from service.proxy import get_proxy, PIIDetector


# Test client
client = TestClient(app)


@pytest.fixture
def sample_tool_registration() -> MCPServerRegistration:
    """Fixture providing a sample tool registration."""
    return MCPServerRegistration(
        tool_id="test_tool",
        name="Test Tool",
        version="1.0.0",
        owner="test-team",
        contact="test@example.com",
        tools=[
            ToolSchema(
                name="test_operation",
                description="Test operation",
                parameters=[
                    ToolParameter(
                        name="input",
                        type="string",
                        description="Test input",
                        required=True,
                    )
                ],
                returns="Test output",
            )
        ],
        auth_flow=AuthFlow.NONE,
        classification=[ToolClassification.SAFE],
        rate_limits=RateLimitConfig(max_calls=10, window_seconds=60),
    )


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check(self) -> None:
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "catalog_size" in data


class TestCatalogManagement:
    """Test catalog CRUD operations."""

    def test_list_tools(self) -> None:
        """Test listing tools from catalog."""
        response = client.get("/catalog/tools")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "servers" in data
        assert data["total"] >= 0

    def test_register_tool(self, sample_tool_registration: MCPServerRegistration) -> None:
        """Test registering a new tool."""
        response = client.post(
            "/catalog/register", json=sample_tool_registration.model_dump()
        )
        assert response.status_code == 201
        data = response.json()
        assert data["tool_id"] == "test_tool"
        assert "registered_at" in data

    def test_register_duplicate_tool(
        self, sample_tool_registration: MCPServerRegistration
    ) -> None:
        """Test registering duplicate tool fails."""
        # First registration should succeed
        response1 = client.post(
            "/catalog/register", json=sample_tool_registration.model_dump()
        )
        assert response1.status_code == 201

        # Second registration should fail
        response2 = client.post(
            "/catalog/register", json=sample_tool_registration.model_dump()
        )
        assert response2.status_code == 409

    def test_get_tool_schema(self) -> None:
        """Test retrieving tool schema."""
        # Use a pre-registered sample tool
        response = client.get("/catalog/tools/web_search/schema/search")
        assert response.status_code == 200
        data = response.json()
        assert data["tool_id"] == "web_search"
        assert data["tool_name"] == "search"
        assert "schema" in data
        assert "classification" in data

    def test_enable_disable_tool(self, sample_tool_registration: MCPServerRegistration) -> None:
        """Test enabling and disabling tools."""
        # Register tool first
        client.post("/catalog/register", json=sample_tool_registration.model_dump())

        # Disable
        response = client.patch("/catalog/tools/test_tool/disable")
        assert response.status_code == 200

        # Enable
        response = client.patch("/catalog/tools/test_tool/enable")
        assert response.status_code == 200

    def test_search_by_classification(self) -> None:
        """Test searching tools by classification."""
        response = client.get(
            "/catalog/search", params={"classification": ToolClassification.SAFE.value}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "servers" in data


class TestAuthentication:
    """Test authentication and token management."""

    def test_mint_token(self) -> None:
        """Test minting ephemeral token."""
        request = EphemeralTokenRequest(
            scope=TokenScope(
                tool_ids=["web_search", "text_processing"],
                actor_id="test_agent",
                actor_type="subagent",
                max_invocations=10,
            ),
            ttl_minutes=15,
        )

        response = client.post("/auth/token", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert "expires_at" in data
        assert "scope" in data

    def test_validate_token(self) -> None:
        """Test token validation."""
        # First mint a token
        request = EphemeralTokenRequest(
            scope=TokenScope(
                tool_ids=["web_search"],
                actor_id="test_agent",
                actor_type="subagent",
            )
        )
        mint_response = client.post("/auth/token", json=request.model_dump())
        token = mint_response.json()["token"]

        # Validate it
        response = client.post("/auth/validate", params={"token": token})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "scope" in data

    def test_validate_invalid_token(self) -> None:
        """Test validation fails for invalid token."""
        response = client.post("/auth/validate", params={"token": "invalid.token.here"})
        assert response.status_code == 401


class TestToolInvocation:
    """Test tool invocation with various scenarios."""

    def test_invoke_tool_orchestrated_mode(self) -> None:
        """Test invoking tool in orchestrated mode."""
        request = ToolInvocationRequest(
            tool_id="text_processing",
            tool_name="summarize",
            arguments={"text": "This is a long text to summarize", "max_length": 50},
            actor_id="lead_agent",
            actor_type="lead_agent",
            runtime_mode=RuntimeMode.ORCHESTRATED,
        )

        response = client.post("/tools/invoke", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
        assert "invocation_id" in data
        assert "execution_time_ms" in data

    def test_invoke_tool_scoped_direct_mode(self) -> None:
        """Test invoking tool in scoped direct mode with token."""
        # First mint a token
        token_request = EphemeralTokenRequest(
            scope=TokenScope(
                tool_ids=["text_processing"],
                actor_id="test_subagent",
                actor_type="subagent",
            )
        )
        token_response = client.post("/auth/token", json=token_request.model_dump())
        token = token_response.json()["token"]

        # Invoke tool with token
        invocation_request = ToolInvocationRequest(
            tool_id="text_processing",
            tool_name="extract_entities",
            arguments={"text": "Apple Inc. is located in Cupertino"},
            actor_id="test_subagent",
            actor_type="subagent",
            runtime_mode=RuntimeMode.SCOPED_DIRECT,
            token=token,
        )

        response = client.post("/tools/invoke", json=invocation_request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_invoke_nonexistent_tool(self) -> None:
        """Test invoking nonexistent tool fails."""
        request = ToolInvocationRequest(
            tool_id="nonexistent_tool",
            tool_name="test",
            arguments={},
            actor_id="test_agent",
            actor_type="lead_agent",
        )

        response = client.post("/tools/invoke", json=request.model_dump())
        assert response.status_code == 400

    def test_invoke_without_token_in_scoped_mode(self) -> None:
        """Test invoking in scoped mode without token fails."""
        request = ToolInvocationRequest(
            tool_id="text_processing",
            tool_name="summarize",
            arguments={"text": "Test"},
            actor_id="test_agent",
            actor_type="subagent",
            runtime_mode=RuntimeMode.SCOPED_DIRECT,
        )

        response = client.post("/tools/invoke", json=request.model_dump())
        assert response.status_code == 403

    def test_invoke_with_wrong_scope_token(self) -> None:
        """Test invoking tool not in token scope fails."""
        # Mint token for one tool
        token_request = EphemeralTokenRequest(
            scope=TokenScope(
                tool_ids=["web_search"],
                actor_id="test_agent",
                actor_type="subagent",
            )
        )
        token_response = client.post("/auth/token", json=token_request.model_dump())
        token = token_response.json()["token"]

        # Try to use it for different tool
        invocation_request = ToolInvocationRequest(
            tool_id="text_processing",
            tool_name="summarize",
            arguments={"text": "Test"},
            actor_id="test_agent",
            actor_type="subagent",
            runtime_mode=RuntimeMode.SCOPED_DIRECT,
            token=token,
        )

        response = client.post("/tools/invoke", json=invocation_request.model_dump())
        assert response.status_code == 403


class TestPIIDetection:
    """Test PII detection functionality."""

    def test_pii_detector_keyword_detection(self) -> None:
        """Test PII detector identifies keywords."""
        detector = PIIDetector()

        # Test data with PII keywords
        data_with_pii = {"api_key": "secret123", "query": "search term"}
        assert detector.detect(data_with_pii) is True

        # Test data without PII
        data_without_pii = {"query": "normal search", "max_results": 10}
        assert detector.detect(data_without_pii) is False

    def test_pii_detector_pattern_detection(self) -> None:
        """Test PII detector identifies patterns."""
        detector = PIIDetector()

        # SSN pattern
        data_with_ssn = {"text": "My SSN is 123-45-6789"}
        assert detector.detect(data_with_ssn) is True

        # Email pattern
        data_with_email = {"contact": "user@example.com"}
        assert detector.detect(data_with_email) is True

    def test_invocation_pii_detection(self) -> None:
        """Test PII detection in tool invocation."""
        request = ToolInvocationRequest(
            tool_id="text_processing",
            tool_name="summarize",
            arguments={"text": "Contact me at secret_api_key=abc123"},
            actor_id="test_agent",
            actor_type="lead_agent",
        )

        response = client.post("/tools/invoke", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        # Should detect PII but still succeed
        assert data["pii_detected"] is True


class TestProvenance:
    """Test provenance logging."""

    def test_provenance_logging(self) -> None:
        """Test that invocations are logged."""
        # Make an invocation
        request = ToolInvocationRequest(
            tool_id="text_processing",
            tool_name="summarize",
            arguments={"text": "Test text"},
            actor_id="test_agent",
            actor_type="lead_agent",
        )
        client.post("/tools/invoke", json=request.model_dump())

        # Check provenance logs
        response = client.get("/provenance/logs", params={"limit": 10})
        assert response.status_code == 200
        logs = response.json()
        assert len(logs) > 0

        # Verify log structure
        log = logs[-1]
        assert "invocation_id" in log
        assert "tool_id" in log
        assert "actor_id" in log
        assert "arguments_hash" in log
        assert "execution_time_ms" in log

    def test_provenance_filtering(self) -> None:
        """Test filtering provenance logs by tool_id."""
        # Make invocations for specific tool
        request = ToolInvocationRequest(
            tool_id="text_processing",
            tool_name="summarize",
            arguments={"text": "Test"},
            actor_id="test_agent",
            actor_type="lead_agent",
        )
        client.post("/tools/invoke", json=request.model_dump())

        # Query filtered logs
        response = client.get(
            "/provenance/logs", params={"limit": 10, "tool_id": "text_processing"}
        )
        assert response.status_code == 200
        logs = response.json()

        # All logs should be for the specified tool
        for log in logs:
            assert log["tool_id"] == "text_processing"


class TestIntegration:
    """Integration tests for end-to-end workflows."""

    def test_complete_workflow_orchestrated(self) -> None:
        """Test complete workflow in orchestrated mode."""
        # 1. Check catalog
        catalog_response = client.get("/catalog/tools")
        assert catalog_response.status_code == 200

        # 2. Get tool schema
        schema_response = client.get("/catalog/tools/web_search/schema/search")
        assert schema_response.status_code == 200

        # 3. Invoke tool
        invocation_request = ToolInvocationRequest(
            tool_id="web_search",
            tool_name="search",
            arguments={"query": "test query", "max_results": 5},
            actor_id="lead_agent",
            actor_type="lead_agent",
            runtime_mode=RuntimeMode.ORCHESTRATED,
        )
        invocation_response = client.post(
            "/tools/invoke", json=invocation_request.model_dump()
        )
        assert invocation_response.status_code == 200

        # 4. Verify provenance
        provenance_response = client.get("/provenance/logs")
        assert provenance_response.status_code == 200
        assert len(provenance_response.json()) > 0

    def test_complete_workflow_scoped_direct(self) -> None:
        """Test complete workflow in scoped direct mode."""
        # 1. Mint ephemeral token
        token_request = EphemeralTokenRequest(
            scope=TokenScope(
                tool_ids=["text_processing", "file_operations"],
                actor_id="research_subagent",
                actor_type="subagent",
                max_invocations=5,
            ),
            ttl_minutes=10,
        )
        token_response = client.post("/auth/token", json=token_request.model_dump())
        assert token_response.status_code == 200
        token = token_response.json()["token"]

        # 2. Validate token
        validate_response = client.post("/auth/validate", params={"token": token})
        assert validate_response.status_code == 200

        # 3. Invoke tool with token
        invocation_request = ToolInvocationRequest(
            tool_id="text_processing",
            tool_name="extract_entities",
            arguments={"text": "Microsoft is based in Redmond"},
            actor_id="research_subagent",
            actor_type="subagent",
            runtime_mode=RuntimeMode.SCOPED_DIRECT,
            token=token,
        )
        invocation_response = client.post(
            "/tools/invoke", json=invocation_request.model_dump()
        )
        assert invocation_response.status_code == 200
        assert invocation_response.json()["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
