from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from itertools import groupby
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

from document_processor.chunking import Chunk
from document_processor.core.models import (
    DocumentPart,
    HeadingSegment,
    LogicalDocument,
    LogicalNode,
    Segment,
    TableCell,
    TableInfo,
    heading_label,
)

type ChunkOrGap = Chunk | Gap

if TYPE_CHECKING:
    from document_processor.pipeline.pipeline import PipelineResult


@dataclass(frozen=True, kw_only=True)
class ReviewWarning:
    code: str
    message: str
    chunk_id: str | None = None
    segment: int | None = None


@dataclass(frozen=True, kw_only=True)
class ResultSummary:
    source: Path
    block_count: int
    segment_count: int
    nodes_by_levels: tuple[tuple[str, int], ...]
    front_matter: tuple[int, int]
    end_matter: tuple[int, int]
    chunk_count: int
    chunk_tokens: tuple[int, ...]
    warnings: tuple[ReviewWarning, ...]
    max_tokens: int

    @property
    def smallest_chunk_tokens(self) -> int | None:
        return min(self.chunk_tokens) if self.chunk_tokens else None

    @property
    def median_chunk_tokens(self) -> float | None:
        return median(self.chunk_tokens) if self.chunk_tokens else None

    @property
    def largest_chunk_tokens(self) -> int | None:
        return max(self.chunk_tokens) if self.chunk_tokens else None

    def format_statistics(self) -> str:
        node_count = sum(count for _, count in self.nodes_by_levels)
        levels = (
            " · ".join(f"{level} {count}" for level, count in self.nodes_by_levels)
            or "none"
        )
        if self.chunk_tokens:
            sizes = (
                f"min {self.smallest_chunk_tokens} · median {round(self.median_chunk_tokens or 0)}"
                f" · max {self.largest_chunk_tokens} (budget {self.max_tokens})"
            )
        else:
            sizes = "none"
        lines = [
            self.source.name,
            f"  {self.block_count} blocks → {self.segment_count} segments, {node_count} nodes, {self.chunk_count} chunks",
            f"  front matter [{self.front_matter[0]}, {self.front_matter[1]}) · end matter [{self.end_matter[0]}, {self.end_matter[1]})",
            f"  nodes: {levels}",
            f"  chunk tokens: {sizes}",
        ]
        return "\n".join(lines)


    def format(self) -> str:
        warning_lines = [f"  ⚠ {w.code}  {w.message}" for w in self.warnings] or ["  no warnings"]
        return "\n".join([self.format_statistics(), *warning_lines])


@dataclass(frozen=True, kw_only=True)
class Gap:
    """Segments [start, end) in no chunk: headings a later breadcrumb carries, or dropped"""

    # "gap-<first segment index>"; warning anchors and HTML ids use it
    id: str
    part: DocumentPart
    start: int
    end: int
    in_breadcrumb: bool


@dataclass(frozen=True, kw_only=True)
class ReviewNode:
    node_id: str
    label: str
    level: str
    depth: int
    is_section: bool
    # chunks whose range overlaps the node's
    overlap_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ReviewData:
    summary: ResultSummary
    document: LogicalDocument
    depths: Mapping[str, int]  # node id -> depth
    outline: tuple[ReviewNode, ...]
    # document order, covering every segment
    chunks_or_gaps: tuple[ChunkOrGap, ...]


def build_review_data(result: PipelineResult) -> ReviewData:
    depths = _node_depths(result.logical.nodes)
    return ReviewData(
        summary=result.summarize(),
        document=result.logical,
        depths=depths,
        outline=_build_outline(result.logical, result.chunks, depths),
        chunks_or_gaps=_build_chunks_or_gaps(result.logical, result.chunks),
    )


def build_table_rows(
    table: TableInfo, start: int, end: int
) -> tuple[tuple[TableCell, ...], ...]:
    """The header rows, then body rows [start, end), as render_table_rows show them"""

    shown_rows = [*range(table.n_header_rows), *range(start, end)]

    cells_by_row: dict[int, list[TableCell]] = {row: [] for row in shown_rows}

    for cell in table.cells:
        if cell.row < table.n_header_rows:
            cells_by_row[cell.row].append(cell)
            continue
        first_row = max(cell.row, start)
        last_row = min(cell.row + cell.row_span, end)
        if first_row < last_row:
            cells_by_row[first_row].append(replace(cell, row=first_row, row_span=last_row - first_row))

    return tuple(
        tuple(sorted(cells_by_row[row], key=lambda cell: cell.col))
        for row in shown_rows
    )


def format_outline(
    document: LogicalDocument, sections_only: bool = False, indent: int = 2
) -> str:
    outline_builder: list[str] = []

    if document.body_start == 0:
        outline_builder.append("front matter  none")
    else:
        outline_builder.append(f"front matter  [0, {document.body_start})")

    depths = _node_depths(document.nodes)

    for node in document.nodes:
        if sections_only and not node.is_section:
            continue
        label = node_label(node)
        line = f"{indent * depths[node.id] * ' '}{label}  [{node.start}, {node.end})"
        outline_builder.append(line)

    if document.body_end == len(document.segments):
        outline_builder.append("end matter  none")
    else:
        outline_builder.append(
            f"end matter  [{document.body_end}, {len(document.segments)})"
        )

    return "\n".join(outline_builder)


def _build_outline(
    document: LogicalDocument, chunks: Sequence[Chunk], depths: Mapping[str, int]
) -> tuple[ReviewNode, ...]:
    outline: list[ReviewNode] = []

    for node in document.nodes:
        outline.append(
            ReviewNode(
                node_id=node.id,
                label=node_label(node),
                level=node.level,
                depth=depths[node.id],
                is_section=node.is_section,
                overlap_chunk_ids=tuple(
                    c.id for c in chunks if c.start < node.end and node.start < c.end
                ),
            )
        )
    return tuple(outline)


def _build_chunks_or_gaps(
    document: LogicalDocument, chunks: Sequence[Chunk]
) -> tuple[ChunkOrGap, ...]:
    chunks_and_gaps: list[ChunkOrGap] = []

    breadcrumb_node_ids = {
        step.node_id
        for chunk in chunks
        for step in chunk.path
        if step.start < chunk.start
    }

    cursor = 0
    for chunk in chunks:
        chunks_and_gaps.extend(
            _build_uncovered(document, cursor, chunk.start, breadcrumb_node_ids)
        )
        chunks_and_gaps.append(chunk)
        cursor = max(cursor, chunk.end)
    chunks_and_gaps.extend(
        _build_uncovered(document, cursor, len(document.segments), breadcrumb_node_ids)
    )

    return tuple(chunks_and_gaps)


def _build_uncovered(
    document: LogicalDocument,
    start: int,
    end: int,
    breadcrumb_node_ids: AbstractSet[str],
) -> tuple[Gap, ...]:
    """Segments [start, end) that no chunk covers, one gap per part they fall in"""
    parts: tuple[tuple[DocumentPart, int, int], ...] = (
        ("front_matter", 0, document.body_start),
        ("body", document.body_start, document.body_end),
        ("end_matter", document.body_end, len(document.segments)),
    )

    gaps: list[Gap] = []
    for part, part_start, part_end in parts:
        run_start, run_end = max(start, part_start), min(end, part_end)
        runs = groupby(
            range(run_start, run_end),
            key=lambda index: _is_in_breadcrumb(
                document.segments[index], breadcrumb_node_ids
            ),
        )

        for in_breadcrumb, run in runs:
            indexes = list(run)
            gaps.append(
                Gap(
                    id=f"gap-{indexes[0]}",
                    part=part,
                    start=indexes[0],
                    end=indexes[-1] + 1,
                    in_breadcrumb=in_breadcrumb,
                )
            )
    return tuple(gaps)


def _is_in_breadcrumb(segment: Segment, breadcrumb_node_ids: AbstractSet[str]) -> bool:
    return (
        isinstance(segment, HeadingSegment) and segment.node_id in breadcrumb_node_ids
    )


def node_label(node: LogicalNode) -> str:
    return heading_label(node.marker, node.title) or f"[{node.level}]"


def _node_depths(nodes: Sequence[LogicalNode]) -> dict[str, int]:
    """Depth by the parent chain, roots at 0; nodes in pre-order, so a parent comes before its children"""
    depths: dict[str, int] = {}
    for node in nodes:
        depths[node.id] = (
            depths[node.parent_id] + 1 if node.parent_id is not None else 0
        )
    return depths
