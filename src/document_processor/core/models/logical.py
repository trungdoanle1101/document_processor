from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
from types import MappingProxyType

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
    ## number: number given to logical level
    number: str | None

    # Blocks
    marker: BlockSpan | None
    title: tuple[BlockSpan, ...]
    body: tuple[BlockSpan, ...]

    # Relationships with other nodes
    children_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Empty LogicalNode id not allowed")
        if not self.level.strip():
            raise ValueError(f"LogicalNode: {self.id!r}: empty attribute 'level'")
        if self.number is not None and not self.number.strip():
            raise ValueError(f"LogicalNode: {self.id!r}: empty attribute 'number'")

        # Check children_ids duplicate:
        seen_ids: set[str] = set()
        for c_id in self.children_ids:
            if c_id in seen_ids:
                raise ValueError(f"LogicalNode: {self.id!r} contains duplicate children ids: {c_id}")
            seen_ids.add(c_id)
            


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

    _node_by_id_mapping: Mapping[str, LogicalNode] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Check duplicate id's in nodes
        node_by_id: dict[str, LogicalNode] = dict()
        for i, n in enumerate(self.nodes):
            if n.id in node_by_id:
                raise ValueError(
                    f"LogicalDocument {self.id!r}: duplicate node id {n.id!r} at position {i}"
                )
            node_by_id[n.id] = n

        
        # Check coverage from root_ids.
        # Use dict[str, None] as ordered set
        order: dict[str, None] = dict()
        for root_id in self.root_ids:
            stack: list[str] = [root_id]
            
            while stack:
                node_id = stack.pop()

                if node_id not in node_by_id:
                    raise ValueError(f"LogicalDocument {self.id!r}: no node with id {node_id}")
                node = node_by_id[node_id]

                if node_id in order:
                    raise ValueError(f"LogicalDocument {self.id!r}: node id {node_id!r} already visited")
                
                order[node_id] = None
                  
                for c_id in reversed(node.children_ids):
                    stack.append(c_id)
        
        if len(order) != len(node_by_id):
            unreachable_ids = sorted(set(node_by_id) - set(order))
            raise ValueError(f"LogicalDocument {self.id!r}: attribute 'nodes' contains unreachable nodes when traversing trees with roots from attribute 'root_ids'. Unreachable ids: {unreachable_ids}")

        if list(order) != [n.id for n in self.nodes]:
            raise ValueError(f"LogicalDocument {self.id!r}: attribute 'nodes' contains nodes not in pre-order")

        # Set the mapping as attribute post init
        object.__setattr__(self, "_node_by_id_mapping", MappingProxyType(node_by_id))
    
    def get_node(self, node_id: str) -> LogicalNode:
        if node_id not in self._node_by_id_mapping:
            raise KeyError(f"LogicalDocument {self.id!r}: no node with id {node_id}")
        return self._node_by_id_mapping[node_id]
