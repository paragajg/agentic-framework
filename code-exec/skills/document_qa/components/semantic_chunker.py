"""
Semantic Chunker Component for Haystack 2.x.

Module: code-exec/skills/document_qa/components/semantic_chunker.py

Structure-aware chunking that preserves document semantics.
Never splits tables, code blocks, or lists mid-content.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from haystack import Document, component

logger = logging.getLogger(__name__)


@component
class SemanticChunker:
    """
    Structure-aware chunking that preserves document semantics.

    Features:
    - Detects structure: headings, tables, lists, code blocks
    - Preserves tables and code blocks where possible (splits if too large)
    - Chunks by semantic boundaries (sections)
    - Adds overlap for context continuity
    - Preserves source metadata in each chunk
    - Respects embedding model token limits (max 8000 tokens per chunk)

    Configuration via environment:
    - DOCUMENT_QA_CHUNK_SIZE: Target chunk size in tokens (default: 512)
    - DOCUMENT_QA_CHUNK_OVERLAP: Overlap between chunks (default: 50)
    """

    # Maximum tokens per chunk for embedding models
    # OpenAI text-embedding models have 8191 token limit; use 8000 for safety margin
    # Conservative limit - OpenAI API limit is 8191, but our token estimation
    # (len/4) can be off by 2x for certain content (unicode, special chars).
    # Using 4000 to ensure we stay well under the limit.
    MAX_EMBEDDING_TOKENS = 4000

    # Regex patterns for structure detection
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    TABLE_START = re.compile(r"^\|.+\|$", re.MULTILINE)
    TABLE_SEPARATOR = re.compile(r"^\|[-:| ]+\|$", re.MULTILINE)
    CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    LIST_ITEM = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)
    NUMBERED_LIST = re.compile(r"^[\s]*\d+\.\s+", re.MULTILINE)

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        preserve_tables: bool = True,
        preserve_code_blocks: bool = True,
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target chunk size in tokens (default from env: 512)
            chunk_overlap: Overlap between chunks (default from env: 50)
            preserve_tables: Never split tables (may exceed chunk size)
            preserve_code_blocks: Never split code blocks
        """
        self.chunk_size = chunk_size or int(
            os.getenv("DOCUMENT_QA_CHUNK_SIZE", "512")
        )
        self.chunk_overlap = chunk_overlap or int(
            os.getenv("DOCUMENT_QA_CHUNK_OVERLAP", "50")
        )
        self.preserve_tables = preserve_tables
        self.preserve_code_blocks = preserve_code_blocks

    @component.output_types(chunks=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Chunk documents with semantic awareness.

        Args:
            documents: List of Haystack Documents to chunk

        Returns:
            Dictionary with 'chunks' key containing chunked documents
        """
        all_chunks: List[Document] = []

        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return {"chunks": all_chunks}

    def _chunk_document(self, document: Document) -> List[Document]:
        """Chunk a single document."""
        content = document.content
        if not content:
            return []

        # First, extract protected blocks (tables, code)
        protected_blocks, remaining_content = self._extract_protected_blocks(content)

        # Split remaining content into sections
        sections = self._split_by_sections(remaining_content)

        # Create chunks from sections
        chunks = []
        current_section: Optional[str] = None

        for section_title, section_content in sections:
            if section_title:
                current_section = section_title

            section_chunks = self._chunk_section(
                content=section_content,
                section_title=current_section,
                base_meta=document.meta,
            )
            chunks.extend(section_chunks)

        # Add protected blocks as chunks (split if too large for embedding model)
        for block_type, block_content, position in protected_blocks:
            block_chunks = self._chunk_protected_block(
                content=block_content,
                block_type=block_type,
                base_meta=document.meta,
            )
            chunks.extend(block_chunks)

        # Sort chunks by original position if possible
        return chunks

    def _extract_protected_blocks(
        self, content: str
    ) -> Tuple[List[Tuple[str, str, int]], str]:
        """
        Extract tables and code blocks that shouldn't be split.

        Returns:
            Tuple of (protected_blocks, remaining_content)
        """
        protected: List[Tuple[str, str, int]] = []
        remaining = content

        # Extract code blocks
        if self.preserve_code_blocks:
            for match in self.CODE_BLOCK.finditer(content):
                block = match.group(0)
                protected.append(("code", block, match.start()))
                remaining = remaining.replace(block, "\n[CODE_BLOCK]\n", 1)

        # Extract tables
        if self.preserve_tables:
            tables = self._extract_tables(remaining)
            for table, position in tables:
                protected.append(("table", table, position))
                remaining = remaining.replace(table, "\n[TABLE]\n", 1)

        return protected, remaining

    def _extract_tables(self, content: str) -> List[Tuple[str, int]]:
        """Extract markdown tables from content."""
        tables = []
        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this looks like a table row
            if self.TABLE_START.match(line):
                table_lines = [line]
                j = i + 1

                # Check for separator row
                if j < len(lines) and self.TABLE_SEPARATOR.match(lines[j]):
                    table_lines.append(lines[j])
                    j += 1

                    # Collect remaining table rows
                    while j < len(lines) and self.TABLE_START.match(lines[j]):
                        table_lines.append(lines[j])
                        j += 1

                    if len(table_lines) >= 3:  # Header + separator + at least 1 row
                        table = "\n".join(table_lines)
                        position = content.find(table)
                        tables.append((table, position))

                i = j
            else:
                i += 1

        return tables

    def _split_by_sections(self, content: str) -> List[Tuple[Optional[str], str]]:
        """
        Split content by headings into sections.

        Returns:
            List of (section_title, section_content) tuples
        """
        sections = []
        lines = content.split("\n")
        current_title: Optional[str] = None
        current_lines: List[str] = []

        for line in lines:
            heading_match = self.HEADING_PATTERN.match(line)

            if heading_match:
                # Save previous section
                if current_lines:
                    section_content = "\n".join(current_lines).strip()
                    if section_content:
                        sections.append((current_title, section_content))

                # Start new section
                current_title = heading_match.group(2).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            section_content = "\n".join(current_lines).strip()
            if section_content:
                sections.append((current_title, section_content))

        return sections

    def _chunk_section(
        self,
        content: str,
        section_title: Optional[str],
        base_meta: Dict[str, Any],
    ) -> List[Document]:
        """Chunk a single section."""
        if not content.strip():
            return []

        # Estimate tokens (rough: 1 token ≈ 4 chars for English)
        tokens = len(content) // 4

        # Safety check: even "small" sections must not exceed embedding limit
        if tokens <= self.chunk_size and tokens <= self.MAX_EMBEDDING_TOKENS:
            # Single chunk for small sections
            meta = {
                **base_meta,
                "section": section_title,
                "chunk_type": "text",
            }
            return [Document(content=content.strip(), meta=meta)]

        # If section is larger than embedding limit, force split regardless of chunk_size
        effective_chunk_size = min(self.chunk_size, self.MAX_EMBEDDING_TOKENS)

        # Need to split - use sentence boundaries
        chunks = []
        sentences = self._split_sentences(content)

        current_chunk: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(sentence) // 4

            # Safety: if a single sentence exceeds the limit, force-split it
            if sentence_tokens > self.MAX_EMBEDDING_TOKENS:
                # First, save any pending chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    meta = {
                        **base_meta,
                        "section": section_title,
                        "chunk_type": "text",
                        "chunk_index": len(chunks),
                    }
                    chunks.append(Document(content=chunk_text.strip(), meta=meta))
                    current_chunk = []
                    current_tokens = 0

                # Split the long sentence by character limit
                split_chunks = self._force_split_text(sentence, section_title, base_meta, len(chunks))
                chunks.extend(split_chunks)
                continue

            if current_tokens + sentence_tokens > effective_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                meta = {
                    **base_meta,
                    "section": section_title,
                    "chunk_type": "text",
                    "chunk_index": len(chunks),
                }
                chunks.append(Document(content=chunk_text.strip(), meta=meta))

                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = len(overlap_text) // 4 if overlap_text else 0

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Save last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            meta = {
                **base_meta,
                "section": section_title,
                "chunk_type": "text",
                "chunk_index": len(chunks),
            }
            chunks.append(Document(content=chunk_text.strip(), meta=meta))

        return chunks

    def _force_split_text(
        self,
        text: str,
        section_title: Optional[str],
        base_meta: Dict[str, Any],
        start_index: int,
    ) -> List[Document]:
        """Force-split text that exceeds embedding limits by character count."""
        max_chars = self.MAX_EMBEDDING_TOKENS * 4 - 100  # Leave margin
        chunks = []

        logger.warning(
            f"Force-splitting oversized text ({len(text) // 4} tokens) "
            f"into chunks of max {self.MAX_EMBEDDING_TOKENS} tokens"
        )

        for i in range(0, len(text), max_chars):
            chunk_text = text[i:i + max_chars].strip()
            if chunk_text:
                meta = {
                    **base_meta,
                    "section": section_title,
                    "chunk_type": "text",
                    "chunk_index": start_index + len(chunks),
                    "force_split": True,
                }
                chunks.append(Document(content=chunk_text, meta=meta))

        return chunks

    def _split_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        # Simple sentence splitting on punctuation
        sentences = re.split(r"(?<=[.!?])\s+", content)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap(self, chunks: List[str]) -> str:
        """Get overlap text from the end of chunks."""
        if not chunks:
            return ""

        overlap_tokens = self.chunk_overlap
        overlap_chars = overlap_tokens * 4

        # Get last few sentences up to overlap size
        combined = " ".join(chunks)
        if len(combined) <= overlap_chars:
            return combined

        return combined[-overlap_chars:]

    def _chunk_protected_block(
        self,
        content: str,
        block_type: str,
        base_meta: Dict[str, Any],
    ) -> List[Document]:
        """
        Chunk a protected block if it exceeds embedding model token limits.

        Preserves block integrity where possible, but splits large blocks
        to avoid embedding API failures.

        Args:
            content: The block content (table or code block)
            block_type: Type of block ("table" or "code")
            base_meta: Base metadata from parent document

        Returns:
            List of Document chunks
        """
        content = content.strip()
        estimated_tokens = len(content) // 4

        # If small enough, keep as single chunk
        if estimated_tokens <= self.MAX_EMBEDDING_TOKENS:
            return [Document(
                content=content,
                meta={**base_meta, "chunk_type": block_type, "is_protected": True}
            )]

        # Too large - must split (even though it's a protected block)
        logger.warning(
            f"Protected {block_type} block ({estimated_tokens} estimated tokens) exceeds "
            f"embedding limit ({self.MAX_EMBEDDING_TOKENS}). Splitting into chunks..."
        )

        if block_type == "table":
            return self._split_large_table(content, base_meta)
        else:
            return self._split_large_code_block(content, base_meta)

    def _split_large_table(
        self,
        content: str,
        base_meta: Dict[str, Any],
    ) -> List[Document]:
        """
        Split a large table into multiple chunks by rows.

        Preserves header row in each chunk for context.
        """
        lines = content.split("\n")
        chunks = []

        # Find header and separator
        header_line = lines[0] if lines else ""
        separator_line = lines[1] if len(lines) > 1 and self.TABLE_SEPARATOR.match(lines[1]) else ""
        header_tokens = (len(header_line) + len(separator_line)) // 4

        # Calculate max rows per chunk (leave room for header)
        max_tokens_for_data = self.MAX_EMBEDDING_TOKENS - header_tokens - 100  # margin
        max_chars_for_data = max_tokens_for_data * 4

        # Start with header
        current_chunk_lines = [header_line]
        if separator_line:
            current_chunk_lines.append(separator_line)
        current_chars = len(header_line) + len(separator_line)

        data_start = 2 if separator_line else 1

        for i, line in enumerate(lines[data_start:], start=data_start):
            line_chars = len(line)

            if current_chars + line_chars > max_chars_for_data and len(current_chunk_lines) > 2:
                # Save current chunk
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(Document(
                    content=chunk_content,
                    meta={
                        **base_meta,
                        "chunk_type": "table",
                        "is_protected": True,
                        "chunk_index": len(chunks),
                        "split_from_large_table": True,
                    }
                ))

                # Start new chunk with header
                current_chunk_lines = [header_line]
                if separator_line:
                    current_chunk_lines.append(separator_line)
                current_chars = len(header_line) + len(separator_line)

            current_chunk_lines.append(line)
            current_chars += line_chars

        # Save last chunk
        if len(current_chunk_lines) > (2 if separator_line else 1):
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(Document(
                content=chunk_content,
                meta={
                    **base_meta,
                    "chunk_type": "table",
                    "is_protected": True,
                    "chunk_index": len(chunks),
                    "split_from_large_table": True,
                }
            ))

        logger.info(f"Split large table into {len(chunks)} chunks")
        return chunks

    def _split_large_code_block(
        self,
        content: str,
        base_meta: Dict[str, Any],
    ) -> List[Document]:
        """
        Split a large code block into multiple chunks by lines.

        Preserves code fence markers and adds context about continuation.
        """
        chunks = []

        # Extract language from code fence if present
        lines = content.split("\n")
        language = ""
        code_start = 0
        code_end = len(lines)

        if lines and lines[0].startswith("```"):
            language = lines[0][3:].strip()
            code_start = 1

        if lines and lines[-1].strip() == "```":
            code_end = len(lines) - 1

        code_lines = lines[code_start:code_end]

        # Calculate max lines per chunk
        fence_overhead = len(f"```{language}\n") + len("\n```") + 50  # margin for context
        max_chars_per_chunk = (self.MAX_EMBEDDING_TOKENS * 4) - fence_overhead

        current_chunk_lines = []
        current_chars = 0

        for i, line in enumerate(code_lines):
            line_chars = len(line) + 1  # +1 for newline

            if current_chars + line_chars > max_chars_per_chunk and current_chunk_lines:
                # Save current chunk
                chunk_content = f"```{language}\n" + "\n".join(current_chunk_lines) + "\n```"
                chunks.append(Document(
                    content=chunk_content,
                    meta={
                        **base_meta,
                        "chunk_type": "code",
                        "is_protected": True,
                        "chunk_index": len(chunks),
                        "split_from_large_code": True,
                        "code_language": language,
                    }
                ))

                # Start new chunk with overlap (last few lines for context)
                overlap_lines = current_chunk_lines[-3:] if len(current_chunk_lines) > 3 else []
                current_chunk_lines = overlap_lines
                current_chars = sum(len(l) + 1 for l in overlap_lines)

            current_chunk_lines.append(line)
            current_chars += line_chars

        # Save last chunk
        if current_chunk_lines:
            chunk_content = f"```{language}\n" + "\n".join(current_chunk_lines) + "\n```"
            chunks.append(Document(
                content=chunk_content,
                meta={
                    **base_meta,
                    "chunk_type": "code",
                    "is_protected": True,
                    "chunk_index": len(chunks),
                    "split_from_large_code": True,
                    "code_language": language,
                }
            ))

        logger.info(f"Split large code block into {len(chunks)} chunks")
        return chunks


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    """
    Convenience function to chunk plain text.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks

    Returns:
        List of chunk strings
    """
    chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    doc = Document(content=text)
    result = chunker.run([doc])

    return [chunk.content for chunk in result["chunks"]]
