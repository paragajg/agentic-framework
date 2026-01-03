"""
Unified Embedding Adapter for Document Q&A and Agents.

Module: code-exec/skills/document_qa/components/embedding_adapter.py

Provides consistent embedding functionality across the framework by:
1. Reading embedding model configuration from environment variables
2. Supporting multiple embedding providers (OpenAI, Sentence-Transformers, etc.)
3. Ensuring document and query embeddings use the same model

Configuration via environment:
- OPENAI_EMBEDDING_MODEL: OpenAI embedding model (e.g., text-embedding-3-large)
- OPENAI_API_KEY: Required for OpenAI embeddings
- EMBEDDING_PROVIDER: Force a specific provider (openai, sentence-transformers)
- EMBEDDING_FALLBACK: Fallback model if primary fails (default: all-MiniLM-L6-v2)

Usage:
    adapter = EmbeddingAdapter()

    # Embed documents
    docs_with_embeddings = adapter.embed_documents(documents)

    # Embed query
    query_embedding = adapter.embed_query("What is the revenue?")
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EmbeddingAdapter:
    """
    Unified embedding adapter that ensures consistent embedding across document indexing and queries.

    Always uses the same model for both document and query embedding to prevent dimension mismatches.
    """

    # Supported providers and their default models
    PROVIDERS = {
        "openai": {
            "models": [
                "text-embedding-3-large",  # 3072 dimensions
                "text-embedding-3-small",  # 1536 dimensions
                "text-embedding-ada-002",  # 1536 dimensions (legacy)
            ],
            "default": "text-embedding-3-large",
            "dimensions": {
                "text-embedding-3-large": 3072,
                "text-embedding-3-small": 1536,
                "text-embedding-ada-002": 1536,
            }
        },
        "sentence-transformers": {
            "models": [
                "sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
                "sentence-transformers/all-mpnet-base-v2",  # 768 dimensions
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384 dimensions
            ],
            "default": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": {
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
            }
        }
    }

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the embedding adapter.

        Priority for model selection:
        1. Explicit model parameter
        2. OPENAI_EMBEDDING_MODEL environment variable
        3. EMBEDDING_MODEL environment variable
        4. Fallback to sentence-transformers

        Args:
            model: Explicit model name (overrides env vars)
            provider: Force a specific provider
            api_key: API key for cloud providers (overrides env vars)
        """
        self._model = None
        self._provider = None
        self._api_key = None
        self._embedder = None
        self._query_embedder = None
        self._initialized = False

        # Determine configuration
        self._configure(model, provider, api_key)

    def _configure(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Configure the adapter based on parameters and environment."""

        # 1. Get model from parameters or environment
        if model:
            self._model = model
        else:
            # Check environment variables in priority order
            self._model = (
                os.getenv("OPENAI_EMBEDDING_MODEL") or
                os.getenv("EMBEDDING_MODEL") or
                os.getenv("EMBEDDING_FALLBACK", "sentence-transformers/all-MiniLM-L6-v2")
            )

        # 2. Determine provider from model name or parameter
        if provider:
            self._provider = provider
        else:
            self._provider = self._detect_provider(self._model)

        # 3. Get API key if needed
        if self._provider == "openai":
            self._api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self._api_key:
                logger.warning(
                    f"OpenAI embedding model '{self._model}' configured but OPENAI_API_KEY not set. "
                    "Falling back to sentence-transformers."
                )
                self._provider = "sentence-transformers"
                self._model = os.getenv("EMBEDDING_FALLBACK", "sentence-transformers/all-MiniLM-L6-v2")

        logger.info(f"EmbeddingAdapter configured: model={self._model}, provider={self._provider}")

    def _detect_provider(self, model: str) -> str:
        """Detect provider from model name."""
        model_lower = model.lower()

        if "text-embedding" in model_lower or model_lower.startswith("openai"):
            return "openai"
        elif "sentence-transformers" in model_lower or model_lower in [
            "all-minilm-l6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-minilm-l12-v2"
        ]:
            return "sentence-transformers"
        else:
            # Default to sentence-transformers for unknown models
            return "sentence-transformers"

    def _initialize(self) -> bool:
        """Lazy initialization of embedding components."""
        if self._initialized:
            return True

        try:
            if self._provider == "openai":
                self._init_openai()
            else:
                self._init_sentence_transformers()

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize {self._provider} embedder: {e}")

            # Try fallback if not already using sentence-transformers
            if self._provider != "sentence-transformers":
                logger.info("Attempting fallback to sentence-transformers...")
                self._provider = "sentence-transformers"
                self._model = os.getenv("EMBEDDING_FALLBACK", "sentence-transformers/all-MiniLM-L6-v2")
                try:
                    self._init_sentence_transformers()
                    self._initialized = True
                    return True
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")

            return False

    def _init_openai(self) -> None:
        """Initialize OpenAI embedders."""
        try:
            from haystack.components.embedders import (
                OpenAIDocumentEmbedder,
                OpenAITextEmbedder,
            )
            from haystack.utils import Secret

            # Convert API key to Haystack Secret type
            api_key_secret = Secret.from_token(self._api_key)

            # Get batch size from environment or use default
            batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

            # Document embedder with batching for performance
            # OpenAI API supports up to 2048 inputs per request
            # Using batch_size=100 to balance throughput and memory
            # Each input is limited to ~4000 tokens (handled by semantic_chunker)
            self._embedder = OpenAIDocumentEmbedder(
                model=self._model,
                api_key=api_key_secret,
                batch_size=batch_size,  # Batch processing for 16x speedup
                progress_bar=True,  # Show progress during embedding
            )

            # Query embedder
            self._query_embedder = OpenAITextEmbedder(
                model=self._model,
                api_key=api_key_secret,
            )

            logger.info(
                f"Initialized OpenAI embedders: {self._model} "
                f"(batch_size={batch_size})"
            )

        except ImportError as e:
            raise ImportError(
                f"OpenAI embedder requires haystack-ai with OpenAI support: {e}"
            )

    def _init_sentence_transformers(self) -> None:
        """Initialize Sentence-Transformers embedders."""
        try:
            from haystack.components.embedders import (
                SentenceTransformersDocumentEmbedder,
                SentenceTransformersTextEmbedder,
            )

            # Normalize model name
            model_name = self._model
            if not model_name.startswith("sentence-transformers/"):
                model_name = f"sentence-transformers/{model_name}"

            # Get batch size from environment or use default (higher for local models)
            batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))

            # Document embedder with batching
            # Local models can handle larger batches since there's no API latency
            self._embedder = SentenceTransformersDocumentEmbedder(
                model=model_name,
                batch_size=batch_size,
                progress_bar=True,
            )

            # Query embedder
            self._query_embedder = SentenceTransformersTextEmbedder(model=model_name)

            # Warm up the embedders (loads models)
            logger.info(f"Loading Sentence-Transformers model: {model_name}")
            self._embedder.warm_up()
            self._query_embedder.warm_up()

            logger.info(
                f"Initialized Sentence-Transformers embedders: {model_name} "
                f"(batch_size={batch_size})"
            )

        except ImportError as e:
            raise ImportError(
                f"Sentence-Transformers embedder requires sentence-transformers package: {e}"
            )

    @property
    def model(self) -> str:
        """Get the configured embedding model name."""
        return self._model

    @property
    def provider(self) -> str:
        """Get the configured provider name."""
        return self._provider

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions for the configured model."""
        provider_info = self.PROVIDERS.get(self._provider, {})
        dimensions = provider_info.get("dimensions", {})

        # Try exact match
        if self._model in dimensions:
            return dimensions[self._model]

        # Try without prefix
        model_short = self._model.split("/")[-1]
        if model_short in dimensions:
            return dimensions[model_short]

        # Default dimensions by provider
        if self._provider == "openai":
            return 3072  # text-embedding-3-large default
        else:
            return 384  # all-MiniLM-L6-v2 default

    def get_document_embedder(self) -> Any:
        """
        Get the document embedder component for Haystack pipelines.

        Returns:
            Haystack document embedder component
        """
        if not self._initialize():
            raise RuntimeError("Failed to initialize embedding adapter")
        return self._embedder

    def get_query_embedder(self) -> Any:
        """
        Get the query embedder component for Haystack pipelines.

        Returns:
            Haystack text embedder component
        """
        if not self._initialize():
            raise RuntimeError("Failed to initialize embedding adapter")
        return self._query_embedder

    def embed_documents(self, documents: List[Any]) -> List[Any]:
        """
        Embed a list of documents.

        Args:
            documents: List of Haystack Document objects

        Returns:
            List of documents with embeddings added
        """
        if not self._initialize():
            raise RuntimeError("Failed to initialize embedding adapter")

        result = self._embedder.run(documents=documents)
        return result.get("documents", [])

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not self._initialize():
            raise RuntimeError("Failed to initialize embedding adapter")

        result = self._query_embedder.run(text=query)
        return result.get("embedding", [])

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Dictionary with model, provider, and dimensions
        """
        return {
            "model": self._model,
            "provider": self._provider,
            "dimensions": self.dimensions,
            "initialized": self._initialized,
        }


# Global singleton for consistent embeddings across the application
_global_adapter: Optional[EmbeddingAdapter] = None


def get_embedding_adapter(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    force_new: bool = False,
) -> EmbeddingAdapter:
    """
    Get the global embedding adapter instance.

    Uses singleton pattern to ensure consistent embeddings across
    document indexing and query processing.

    Args:
        model: Override model (only used if creating new instance)
        provider: Override provider (only used if creating new instance)
        force_new: Force creation of a new adapter instance

    Returns:
        EmbeddingAdapter instance
    """
    global _global_adapter

    if _global_adapter is None or force_new:
        _global_adapter = EmbeddingAdapter(model=model, provider=provider)

    return _global_adapter


def create_embedding_adapter(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EmbeddingAdapter:
    """
    Create a new embedding adapter instance (not singleton).

    Use this when you need a separate adapter with different configuration.

    Args:
        model: Embedding model name
        provider: Provider name (openai, sentence-transformers)
        api_key: API key for cloud providers

    Returns:
        New EmbeddingAdapter instance
    """
    return EmbeddingAdapter(model=model, provider=provider, api_key=api_key)
