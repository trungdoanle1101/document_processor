from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from document_processor.core.models.physical import ImageInfo, TableInfo

from ._checks import (
    require_range,
    require_unique,
    require_visible,
    require_visible_or_none,
)

type Segment = TextSegment | ImageSegment | TableSegment | HeadingSegment


@dataclass(frozen=True, kw_only=True)
class BaseSegment:
    block_id: str

    def __post_init__(self) -> None:
        require_visible(type(self).__name__, "block_id", self.block_id)


@dataclass(frozen=True, kw_only=True)
class TextSegment(BaseSegment):
    text: str

    def __post_init__(self) -> None:
        super().__post_init__()
        owner = f"{type(self).__name__} {self.block_id!r}"
        require_visible(owner, "text", self.text)


@dataclass(frozen=True, kw_only=True)
class ImageSegment(BaseSegment):
    image: ImageInfo


@dataclass(frozen=True, kw_only=True)
class TableSegment(BaseSegment):
    table: TableInfo


@dataclass(frozen=True, kw_only=True)
class HeadingSegment(BaseSegment):
    """A node's heading, at the start of its range: what the builder takes out of the heading block."""

    node_id: str
    marker: str | None
    title: str | None

    def __post_init__(self) -> None:
        super().__post_init__()

        error_owner = type(self).__name__
        require_visible(error_owner, "node_id", self.node_id)

        error_owner = f"{self.__class__.__name__} {self.node_id!r}"

        for name in ("marker", "title"):
            require_visible_or_none(error_owner, name, getattr(self, name))


type DocumentPart = Literal["front_matter", "body", "end_matter"]


@dataclass(frozen=True, kw_only=True)
class LogicalNode:
    """
    Unit of the document a reader would name:
    E.g., a chapter, article, clause, item
    """

    # Stable, unique id
    id: str

    # Enclosing parent node; a root node has None parent_id (meaning the entire document)
    parent_id: str | None

    # Profile level name
    level: str

    # The numbering as written. None for levels without numbers (e.g., bullets)
    number: str | None

    # Marker used
    marker: str | None

    # Title of this node
    title: str | None

    # block ids where marker/title came from
    head_block_ids: tuple[str, ...]

    # Covered segments range: [start, end)
    # Index of starting segment, inclusive
    start: int

    # Index of ending segment, exclusive
    end: int

    def __post_init__(self) -> None:
        require_visible(type(self).__name__, "id", self.id)

        owner = f"{self.__class__.__name__} {self.id!r}"
        require_visible(owner, "level", self.level)

        if self.parent_id == self.id:
            raise ValueError(f"{owner}: 'parent_id' cannot be itself")

        for name in ("parent_id", "number", "marker", "title"):
            require_visible_or_none(owner, name, getattr(self, name))

        if not self.head_block_ids:
            raise ValueError(f"{owner}: 'head_block_ids' cannot be empty")

        for b in self.head_block_ids:
            require_visible(owner, "head_block_ids", b)
        require_unique(owner, "head_block_ids", self.head_block_ids)

        require_range(owner, self.start, self.end)
        if self.start == self.end:
            raise ValueError(f"{owner}: a node holds at least its own heading segment, so start < end; got start=end={self.start}")


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

    # Segments of the entire document
    segments: tuple[Segment, ...]

    # Nodes listed in pre-order
    nodes: tuple[LogicalNode, ...]

    # body: segments[body_start:body_end]
    body_start: int
    body_end: int

    # Front matter will be segments[:body_start]
    # End matter will be segments[body_end:]

    _node_by_id: Mapping[str, LogicalNode] = field(
        init=False, repr=False, compare=False
    )
    _children_by_id: Mapping[str | None, tuple[str, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:

        owner = self.__class__.__name__
        require_visible(owner, "id", self.id)

        owner = f"{self.__class__.__name__} {self.id!r}"
        for name in (
            "physical_document_id",
            "source_file_hash",
            "profile_name",
            "profile_version",
        ):
            require_visible(owner, name, getattr(self, name))

        require_range(
            owner,
            self.body_start,
            self.body_end,
            lower=0,
            upper=len(self.segments),
            start_label="body_start",
            end_label="body_end",
        )

        # Check duplicate node ids
        require_unique(owner, "nodes", (n.id for n in self.nodes))

        # Check duplicate id's in nodes and generate mapping
        self._generate_node_by_id_mapping()

        # Check outline of the nodes
        self._check_outline()

        # Populate _children_by_id
        self._generate_children_by_id_mapping()

    def get_node(self, node_id: str) -> LogicalNode:
        if node_id not in self._node_by_id:
            raise KeyError(
                f"{self.__class__.__name__} {self.id!r}: no node with id {node_id}"
            )
        return self._node_by_id[node_id]

    def get_children(self, node_id: str | None) -> tuple[str, ...]:
        # None node_id = document "root"
        if node_id not in self._children_by_id:
            raise KeyError(
                f"{self.__class__.__name__} {self.id!r}: no node with id {node_id}"
            )
        return self._children_by_id[node_id]

    @property
    def root_ids(self) -> tuple[str, ...]:
        return self._children_by_id[None]

    def _check_outline(self) -> None:
        # stack: keeps track of a node's open ancestors, innermost last
        # top of a stack is a node's parent
        stack: list[LogicalNode] = []

        # previous sibling's end segment index, per parent
        last_end: dict[str | None, int] = {None: self.body_start}

        for index, segment in enumerate(self.segments):
            if isinstance(segment, HeadingSegment):
                node = self._node_by_id.get(segment.node_id)
                if node is None or node.start != index:
                    raise ValueError(
                        f"{type(self).__name__} {self.id!r}: HeadingSegment at {index} for node "
                        f"{segment.node_id!r} is not at that node's start"
                    )

        for i, node in enumerate(self.nodes):
            owner = f"{type(self).__name__} {self.id!r}, node {node.id!r} at pos {i}"

            if (node.parent_id is not None) and (
                node.parent_id not in self._node_by_id
            ):
                raise ValueError(f"{owner}: unknown parent {node.parent_id!r}")

            # Pop until the top of the stack is this node's parent
            while stack and stack[-1].id != node.parent_id:
                stack.pop()

            if node.parent_id is None:
                # a root node can't have ancestors
                if stack:
                    raise AssertionError
                # parent of a root must be the entire body
                lo, hi = self.body_start, self.body_end

            else:
                if not stack:
                    raise ValueError(
                        f"{owner}: parent {node.parent_id!r} is not an open ancestor (nodes not in pre-order) "
                    )
                parent = stack[-1]
                lo, hi = parent.start, parent.end

            require_range(owner, node.start, node.end, upper=len(self.segments))

            if not (lo <= node.start and node.end <= hi):
                raise ValueError(
                    f"{owner}: invalid range [{node.start}, {node.end}), outside parent's range [{lo}, {hi})"
                )

            if node.start < last_end[node.parent_id]:
                raise ValueError(f"{owner}: overlaps or precedes its previous sibling")

            last_end[node.parent_id] = node.end
            last_end[node.id] = node.start

            # Node's first segment has to be a heading segment
            start_segment = self.segments[node.start]
            if not isinstance(start_segment, HeadingSegment):
                raise ValueError(
                    f"{owner}: segment {node.start} must be the node's HeadingSegment, "
                    f"got {type(start_segment).__name__} (block {start_segment.block_id!r})"
                )
            
            if start_segment.node_id != node.id:
                raise ValueError(
                    f"{owner}: segment {node.start} has a different node_id from the node's id, got {start_segment.node_id!r}"
                )
    

            if start_segment.marker != node.marker:
                raise ValueError(
                    f"{owner}: start segment has different marker from node, start_segment.marker={start_segment.marker!r}, node.marker={node.marker!r}"
                )

            if start_segment.title != node.title:
                raise ValueError(
                    f"{owner}: start segment has different title from node, start_segment.title={start_segment.title!r}, node.title={node.title!r}"
                )
            stack.append(node)

    def _generate_node_by_id_mapping(self) -> None:
        node_by_id: dict[str, LogicalNode] = {}
        for n in self.nodes:
            node_by_id[n.id] = n

        object.__setattr__(self, "_node_by_id", MappingProxyType(node_by_id))

    def _generate_children_by_id_mapping(self) -> None:
        children: dict[str | None, list[str]] = {None: []}
        for node in self.nodes:
            children.setdefault(node.parent_id, []).append(node.id)
            children[node.id] = []

        object.__setattr__(
            self,
            "_children_by_id",
            MappingProxyType({k: tuple(v) for k, v in children.items()}),
        )
