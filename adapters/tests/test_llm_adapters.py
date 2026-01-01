"""
Tests for all LLM adapters.
Module: adapters/tests/test_llm_adapters.py
"""

import os
from typing import Any, Dict

import pytest

from adapters.llm import (
    AnthropicAdapter,
    AzureOpenAIAdapter,
    GeminiAdapter,
    LocalLLMAdapter,
    LLMMessage,
    LLMResponse,
    MessageRole,
    MockLLMAdapter,
    OpenAIAdapter,
    VLLMAdapter,
)


class TestMockLLMAdapter:
    """Test MockLLMAdapter for testing without API calls."""

    @pytest.mark.asyncio
    async def test_mock_adapter_basic_completion(self) -> None:
        """Test basic completion with mock adapter."""
        adapter = MockLLMAdapter(model="mock-model")

        messages = [LLMMessage(role=MessageRole.USER, content="Hello")]

        response = await adapter.complete(messages)

        assert isinstance(response, LLMResponse)
        assert "Mock response" in response.content
        assert response.model == "mock-model"

    @pytest.mark.asyncio
    async def test_mock_adapter_custom_response(self) -> None:
        """Test mock adapter with custom response template."""
        adapter = MockLLMAdapter(
            model="test-model",
            response_template="Response to: {prompt}",
        )

        messages = [LLMMessage(role=MessageRole.USER, content="test input")]

        response = await adapter.complete(messages)

        assert "Response to" in response.content


class TestAdapterInitialization:
    """Test adapter initialization with various configurations."""

    def test_anthropic_adapter_requires_api_key(self) -> None:
        """Test that AnthropicAdapter requires API key."""
        # Clear env var if set
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="API key required"):
                AnthropicAdapter(api_key=None)
        finally:
            # Restore env var
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_openai_adapter_requires_api_key(self) -> None:
        """Test that OpenAIAdapter requires API key."""
        old_key = os.environ.pop("OPENAI_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="API key required"):
                OpenAIAdapter(api_key=None)
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_azure_adapter_requires_credentials(self) -> None:
        """Test that AzureOpenAIAdapter requires API key and endpoint."""
        old_key = os.environ.pop("AZURE_OPENAI_KEY", None)
        old_endpoint = os.environ.pop("AZURE_OPENAI_ENDPOINT", None)

        try:
            # Missing API key
            with pytest.raises(ValueError, match="API key required"):
                AzureOpenAIAdapter(api_key=None)

            # Missing endpoint
            with pytest.raises(ValueError, match="endpoint required"):
                AzureOpenAIAdapter(api_key="test-key", endpoint=None)
        finally:
            if old_key:
                os.environ["AZURE_OPENAI_KEY"] = old_key
            if old_endpoint:
                os.environ["AZURE_OPENAI_ENDPOINT"] = old_endpoint

    def test_gemini_adapter_requires_api_key(self) -> None:
        """Test that GeminiAdapter requires API key."""
        old_key = os.environ.pop("GEMINI_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="API key required"):
                GeminiAdapter(api_key=None)
        finally:
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key

    def test_local_adapter_uses_default_endpoint(self) -> None:
        """Test that LocalLLMAdapter uses default endpoint."""
        adapter = LocalLLMAdapter(model="llama3.1:70b")

        assert adapter.endpoint == "http://localhost:11434"
        assert adapter.model == "llama3.1:70b"

    def test_local_adapter_accepts_custom_endpoint(self) -> None:
        """Test that LocalLLMAdapter accepts custom endpoint."""
        adapter = LocalLLMAdapter(
            model="llama3.1:70b",
            endpoint="http://custom-ollama:11434",
        )

        assert adapter.endpoint == "http://custom-ollama:11434"

    def test_vllm_adapter_uses_default_endpoint(self) -> None:
        """Test that VLLMAdapter uses default endpoint."""
        adapter = VLLMAdapter(model="meta-llama/Llama-2-70b-hf")

        assert adapter.endpoint == "http://localhost:8000"
        assert adapter.model == "meta-llama/Llama-2-70b-hf"

    def test_vllm_adapter_accepts_custom_endpoint(self) -> None:
        """Test that VLLMAdapter accepts custom endpoint."""
        adapter = VLLMAdapter(
            model="meta-llama/Llama-2-70b-hf",
            endpoint="http://custom-vllm:8000",
        )

        assert adapter.endpoint == "http://custom-vllm:8000"


class TestAdapterMessageConversion:
    """Test message conversion for different adapters."""

    def test_anthropic_adapter_message_conversion(self) -> None:
        """Test AnthropicAdapter message conversion."""
        adapter = AnthropicAdapter(api_key="test-key")

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content="You are helpful"),
            LLMMessage(role=MessageRole.USER, content="Hello"),
        ]

        system_prompt, api_messages = adapter._convert_messages(messages)

        assert system_prompt == "You are helpful"
        assert len(api_messages) == 1
        assert api_messages[0]["role"] == "user"
        assert api_messages[0]["content"] == "Hello"

    def test_openai_adapter_message_conversion(self) -> None:
        """Test OpenAIAdapter message conversion."""
        adapter = OpenAIAdapter(api_key="test-key")

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content="You are helpful"),
            LLMMessage(role=MessageRole.USER, content="Hello"),
        ]

        api_messages = adapter._convert_messages(messages)

        assert len(api_messages) == 2
        assert api_messages[0]["role"] == "system"
        assert api_messages[1]["role"] == "user"

    def test_azure_adapter_message_conversion(self) -> None:
        """Test AzureOpenAIAdapter message conversion."""
        adapter = AzureOpenAIAdapter(
            api_key="test-key",
            endpoint="https://test.openai.azure.com/",
        )

        messages = [
            LLMMessage(role=MessageRole.USER, content="Hello"),
        ]

        api_messages = adapter._convert_messages(messages)

        assert len(api_messages) == 1
        assert api_messages[0]["role"] == "user"

    def test_gemini_adapter_message_conversion(self) -> None:
        """Test GeminiAdapter message conversion."""
        adapter = GeminiAdapter(api_key="test-key")

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content="You are helpful"),
            LLMMessage(role=MessageRole.USER, content="Hello"),
        ]

        system_prompt, api_messages = adapter._convert_messages(messages)

        assert system_prompt == "You are helpful"
        assert len(api_messages) == 1
        assert api_messages[0]["role"] == "user"

    def test_local_adapter_message_conversion(self) -> None:
        """Test LocalLLMAdapter message conversion."""
        adapter = LocalLLMAdapter(model="llama3.1:70b")

        messages = [
            LLMMessage(role=MessageRole.USER, content="Hello"),
        ]

        api_messages = adapter._convert_messages(messages)

        assert len(api_messages) == 1
        assert api_messages[0]["role"] == "user"
        assert api_messages[0]["content"] == "Hello"

    def test_vllm_adapter_message_conversion(self) -> None:
        """Test VLLMAdapter message conversion."""
        adapter = VLLMAdapter(model="meta-llama/Llama-2-70b-hf")

        messages = [
            LLMMessage(role=MessageRole.USER, content="Hello"),
        ]

        api_messages = adapter._convert_messages(messages)

        assert len(api_messages) == 1
        assert api_messages[0]["role"] == "user"


class TestAdapterConfiguration:
    """Test adapter configuration options."""

    def test_anthropic_adapter_configuration(self) -> None:
        """Test AnthropicAdapter configuration."""
        adapter = AnthropicAdapter(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            api_version="2023-06-01",
        )

        assert adapter.model == "claude-sonnet-4-20250514"
        assert adapter.api_version == "2023-06-01"

    def test_azure_adapter_deployment_name(self) -> None:
        """Test AzureOpenAIAdapter deployment name configuration."""
        adapter = AzureOpenAIAdapter(
            model="gpt-4o",
            api_key="test-key",
            endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o-deployment",
        )

        assert adapter.deployment_name == "gpt-4o-deployment"

    def test_local_adapter_with_api_key(self) -> None:
        """Test LocalLLMAdapter with optional API key."""
        adapter = LocalLLMAdapter(
            model="llama3.1:70b",
            api_key="optional-key",
        )

        assert adapter.api_key == "optional-key"

    def test_vllm_adapter_with_api_key(self) -> None:
        """Test VLLMAdapter with optional API key."""
        adapter = VLLMAdapter(
            model="meta-llama/Llama-2-70b-hf",
            api_key="optional-key",
        )

        assert adapter.api_key == "optional-key"


class TestAdapterPayloadGeneration:
    """Test payload generation for API requests."""

    def test_anthropic_adapter_payload(self) -> None:
        """Test AnthropicAdapter payload generation."""
        adapter = AnthropicAdapter(api_key="test-key")

        # Adapter should accept various parameters
        # (Actual payload generation tested in integration tests)
        assert adapter.model
        assert adapter.api_key

    def test_azure_adapter_endpoint_format(self) -> None:
        """Test AzureOpenAIAdapter endpoint formatting."""
        endpoint = "https://my-resource.openai.azure.com/"
        adapter = AzureOpenAIAdapter(
            api_key="test-key",
            endpoint=endpoint,
            deployment_name="test-deployment",
        )

        assert adapter.endpoint == "https://my-resource.openai.azure.com"
        assert adapter.deployment_name == "test-deployment"

    def test_local_adapter_endpoint_cleanup(self) -> None:
        """Test LocalLLMAdapter endpoint cleanup."""
        adapter = LocalLLMAdapter(
            model="llama3.1:70b",
            endpoint="http://localhost:11434/",
        )

        # Trailing slash should be removed
        assert adapter.endpoint == "http://localhost:11434"

    def test_vllm_adapter_endpoint_cleanup(self) -> None:
        """Test VLLMAdapter endpoint cleanup."""
        adapter = VLLMAdapter(
            model="meta-llama/Llama-2-70b-hf",
            endpoint="http://localhost:8000/",
        )

        # Trailing slash should be removed
        assert adapter.endpoint == "http://localhost:8000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
