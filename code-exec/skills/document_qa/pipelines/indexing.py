"""
Indexing Pipeline for Document Q&A.

Module: code-exec/skills/document_qa/pipelines/indexing.py

Haystack 2.x pipeline for document indexing:
1. Embedding generation using unified EmbeddingAdapter
2. Document store writing (in-memory or persistent)

Configuration via environment:
- OPENAI_EMBEDDING_MODEL: OpenAI embedding model (e.g., text-embedding-3-large)
- OPENAI_API_KEY: Required for OpenAI embeddings
- EMBEDDING_FALLBACK: Fallback model if primary fails

Uses EmbeddingAdapter to ensure consistent embeddings between indexing and queries.
"""

import logging
from typing import Any, Dict, List, Optional

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

from ..components.embedding_adapter import get_embedding_adapter, EmbeddingAdapter

logger = logging.getLogger(__name__)


def create_indexing_pipeline(
    document_store: Optional[InMemoryDocumentStore] = None,
    embedding_adapter: Optional[EmbeddingAdapter] = None,
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
) -> tuple[Pipeline, InMemoryDocumentStore, EmbeddingAdapter]:
    """
    Create the document indexing pipeline.

    Pipeline flow:
    documents -> Embedder (via EmbeddingAdapter) -> DocumentWriter -> document_store

    Args:
        document_store: Optional document store (creates InMemory if not provided)
        embedding_adapter: Optional EmbeddingAdapter (uses global singleton if not provided)
        duplicate_policy: How to handle duplicate documents

    Returns:
        Tuple of (Pipeline, DocumentStore, EmbeddingAdapter)
    """
    # Create or use provided document store
    if document_store is None:
        document_store = InMemoryDocumentStore()

    # Get embedding adapter (uses global singleton for consistency)
    if embedding_adapter is None:
        embedding_adapter = get_embedding_adapter()

    logger.info(
        f"Creating indexing pipeline with: model={embedding_adapter.model}, "
        f"provider={embedding_adapter.provider}, dimensions={embedding_adapter.dimensions}"
    )

    pipeline = Pipeline()

    # Add embedding component from adapter
    embedder = embedding_adapter.get_document_embedder()
    pipeline.add_component("embedder", embedder)

    # Add document writer
    pipeline.add_component(
        "writer",
        DocumentWriter(
            document_store=document_store,
            policy=duplicate_policy,
        ),
    )

    # Connect components
    pipeline.connect("embedder", "writer")

    return pipeline, document_store, embedding_adapter


def run_indexing(
    documents: List[Document],
    document_store: Optional[InMemoryDocumentStore] = None,
    embedding_adapter: Optional[EmbeddingAdapter] = None,
) -> Dict[str, Any]:
    """
    Run the indexing pipeline on documents.

    Args:
        documents: List of Haystack Documents to index
        document_store: Optional document store
        embedding_adapter: Optional EmbeddingAdapter (uses global singleton if not provided)

    Returns:
        Dictionary with 'document_store', 'embedding_adapter', 'documents_written', and 'statistics'
    """
    import time

    # Get embedding adapter early so we can return it even on empty documents
    if embedding_adapter is None:
        embedding_adapter = get_embedding_adapter()

    if not documents:
        return {
            "document_store": document_store or InMemoryDocumentStore(),
            "embedding_adapter": embedding_adapter,
            "documents_written": 0,
            "statistics": {"total_documents": 0},
        }

    pipeline, store, adapter = create_indexing_pipeline(
        document_store=document_store,
        embedding_adapter=embedding_adapter,
    )

    # Log start of embedding process
    logger.info(
        f"Starting embedding generation for {len(documents)} documents "
        f"using {adapter.model} (provider={adapter.provider})..."
    )

    # Run pipeline with timing
    start_time = time.time()
    result = pipeline.run({"embedder": {"documents": documents}})
    elapsed = time.time() - start_time

    documents_written = result.get("writer", {}).get("documents_written", 0)

    statistics = {
        "total_documents": len(documents),
        "documents_written": documents_written,
        "embedding_model": adapter.model,
        "embedding_provider": adapter.provider,
        "embedding_dimensions": adapter.dimensions,
        "store_type": type(store).__name__,
        "embedding_time_seconds": round(elapsed, 2),
        "documents_per_second": round(len(documents) / elapsed, 2) if elapsed > 0 else 0,
    }

    logger.info(
        f"Indexing complete: {documents_written} documents indexed in {elapsed:.1f}s "
        f"({statistics['documents_per_second']:.1f} docs/sec, "
        f"model={adapter.model}, dimensions={adapter.dimensions})"
    )

    return {
        "document_store": store,
        "embedding_adapter": adapter,
        "documents_written": documents_written,
        "statistics": statistics,
    }


def create_query_embedding(
    query: str,
    embedding_adapter: Optional[EmbeddingAdapter] = None,
) -> List[float]:
    """
    Create embedding for a query string.

    Uses the same embedding adapter as indexing to ensure dimension consistency.

    Args:
        query: Query text
        embedding_adapter: Optional EmbeddingAdapter (uses global singleton if not provided)

    Returns:
        Embedding vector as list of floats
    """
    if embedding_adapter is None:
        embedding_adapter = get_embedding_adapter()

    logger.info(
        f"Creating query embedding with: {embedding_adapter.model} "
        f"(provider={embedding_adapter.provider})"
    )

    return embedding_adapter.embed_query(query)
