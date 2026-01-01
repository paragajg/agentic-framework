"""
LLM adapter implementations for various providers.

Supported adapters:
- AnthropicAdapter: Anthropic Claude API
- OpenAIAdapter: OpenAI GPT models
- AzureOpenAIAdapter: Azure OpenAI services
- GeminiAdapter: Google Gemini API
- LocalLLMAdapter: Ollama and compatible local servers
- VLLMAdapter: vLLM local inference server
- MockLLMAdapter: Mock adapter for testing
"""

from adapters.llm.base import LLMAdapter, LLMError, LLMResponse, LLMMessage, MessageRole
from adapters.llm.anthropic import AnthropicAdapter
from adapters.llm.openai import OpenAIAdapter
from adapters.llm.azure import AzureOpenAIAdapter
from adapters.llm.gemini import GeminiAdapter
from adapters.llm.local import LocalLLMAdapter
from adapters.llm.vllm import VLLMAdapter
from adapters.llm.mock import MockLLMAdapter

__all__ = [
    "LLMAdapter",
    "LLMError",
    "LLMResponse",
    "LLMMessage",
    "MessageRole",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "AzureOpenAIAdapter",
    "GeminiAdapter",
    "LocalLLMAdapter",
    "VLLMAdapter",
    "MockLLMAdapter",
]
