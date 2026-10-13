from __future__ import annotations

from dataclasses import dataclass

from .ranges import TextRange


@dataclass(frozen=True, kw_only=True)
class BlockSpan:
    block_id: str
    # Span always carries explicit range
    text_range: TextRange

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValueError("Empty block_id not allowed")


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
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
    root_ids: tuple[str, ...]
    nodes: tuple[LogicalNode, ...]
    end_matter: tuple[BlockSpan, ...]

    def __post_init__(self) -> None:
        # Check duplicate id's in nodes
        # Will probably modify this to create mapping between id -> LogicalNode
        seen_ids: set[str] = set()
        for i, n in enumerate(self.nodes):
            if n.id in seen_ids:
                raise ValueError(
                    f"LogicalDocument {self.id!r}: Duplicate node id {n.id!r} at position {i}"
                )
            seen_ids.add(n.id)

        # TODO: Check coverage from root_ids.
