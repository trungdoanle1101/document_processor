from __future__ import annotations

from dataclasses import dataclass

from .ranges import TextRange


@dataclass(frozen=True)
class BlockSpan:
    block_id: str
    text_range: TextRange | None


@dataclass(frozen=True)
class LogicalNode:
    """
    Unit of the document a reader would name:
    E.g., a chapter, article, clause, item
    """

    # Basic information
    id: str

    # Interpretation within the provided hierarchy
    level: str
    number: str | None

    # Blocks
    marker: BlockSpan | None
    title: tuple[BlockSpan, ...]
    body: tuple[BlockSpan, ...]

    # Relationships with other nodes
    children_ids: tuple[str, ...]


@dataclass(frozen=True)
class LogicalDocument:
    """
    Logical interpretation of a document
    """

    id: str
    physical_document_id: str
    source_file_hash: str

    profile_name: str
    profile_version: str

    front_matter: tuple[BlockSpan, ...]
    nodes: tuple[LogicalNode, ...]
    end_matter: tuple[BlockSpan, ...]
