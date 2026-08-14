"""Core document type shared by loaders, chunking, indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A unit of text plus optional metadata.

    ``page_content`` holds the raw text while ``metadata`` carries provenance
    such as the source path, filename, filetype and (after chunking) the chunk
    offsets within the original document.
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
