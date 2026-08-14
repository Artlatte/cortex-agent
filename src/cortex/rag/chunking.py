"""Recursive character text splitting for RAG ingestion.

The splitter walks a priority-ordered list of separators, splitting on the
first separator actually present in the text and recursing into any piece that
still exceeds ``chunk_size``. Neighbouring chunks share up to ``chunk_overlap``
characters so context is not lost at chunk boundaries.
"""

from __future__ import annotations

from cortex.rag.document import Document

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", " ", ""]


class RecursiveCharacterTextSplitter:
    """Split text recursively into fixed-size chunks with overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators) if separators is not None else list(DEFAULT_SEPARATORS)

    def split_text(self, text: str) -> list[str]:
        """Split ``text`` into chunks no longer than ``chunk_size`` (when possible)."""
        if not text:
            return []
        return self._split(text, self.separators)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents, annotating each chunk with ``chunk_index``/``chunk_count``."""
        chunks: list[Document] = []
        for document in documents:
            pieces = self.split_text(document.page_content)
            chunk_count = len(pieces)
            for index, piece in enumerate(pieces):
                metadata = dict(document.metadata)
                metadata["chunk_index"] = index
                metadata["chunk_count"] = chunk_count
                chunks.append(Document(page_content=piece, metadata=metadata))
        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        chosen: str | None = None
        remaining: list[str] = []
        for index, separator in enumerate(separators):
            if separator == "" or separator in text:
                chosen = separator
                remaining = separators[index + 1 :]
                break

        if chosen is None:
            # No separator matched and no empty-separator fallback: atomic piece.
            return [text]

        pieces = list(text) if chosen == "" else [p for p in text.split(chosen) if p]
        result: list[str] = []
        good: list[str] = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                good.append(piece)
            else:
                result.extend(self._merge(good, chosen))
                good = []
                if not remaining:
                    result.append(piece)
                else:
                    result.extend(self._split(piece, remaining))
        result.extend(self._merge(good, chosen))
        return result

    def _merge(self, pieces: list[str], separator: str) -> list[str]:
        sep_len = len(separator)
        chunks: list[str] = []
        current: list[str] = []
        total = 0
        for piece in pieces:
            piece_len = len(piece)
            if current and total + sep_len + piece_len > self.chunk_size:
                chunks.append(separator.join(current))
                while (
                    total > self.chunk_overlap
                    or (total + piece_len + sep_len > self.chunk_size and total > 0)
                ):
                    total -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                    current = current[1:]
            current.append(piece)
            total += piece_len + (sep_len if len(current) > 1 else 0)
        if current:
            chunks.append(separator.join(current))
        return chunks
