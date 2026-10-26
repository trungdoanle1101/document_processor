from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, assert_never

from document_processor.chunking import Chunk
from document_processor.core.models import (
    DocumentPart,
    LogicalDocument,
    LogicalNode,
    TextSegment,
    TableInfo,
    TableSegment,
    ImageSegment,
    HeadingSegment,
    heading_label,
)

type ReviewSegment = ReviewHeading | ReviewText | ReviewTable | ReviewImage
type ReviewItem = ReviewChunk | ReviewGap

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

    def format(self) -> str:
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
        lines.extend(f"  ⚠ {w.code}  {w.message}" for w in self.warnings)
        if not self.warnings:
            lines.append("  no warnings")
        return "\n".join(lines)


@dataclass(frozen=True, kw_only=True)
class BaseReviewSegment:
    index: int
    block_id: str


@dataclass(frozen=True, kw_only=True)
class ReviewHeading(BaseReviewSegment):
    label: str
    level: str
    depth: int


@dataclass(frozen=True, kw_only=True)
class ReviewText(BaseReviewSegment):
    text: str


@dataclass(frozen=True, kw_only=True)
class ReviewCell:
    text: str
    # clipped to the rows this piece shows
    row_span: int
    col_span: int


@dataclass(frozen=True, kw_only=True)
class ReviewTable(BaseReviewSegment):
    n_header_rows: int
    # header rows, body rows
    rows: tuple[tuple[ReviewCell, ...], ...]


@dataclass(frozen=True, kw_only=True)
class ReviewImage(BaseReviewSegment):
    pass


@dataclass(frozen=True, kw_only=True)
class ReviewChunk:
    chunk: Chunk
    segments: tuple[ReviewSegment, ...]


@dataclass(frozen=True, kw_only=True)
class ReviewGap:
    """Segments that end up in no chunk"""

    # "gap-<first segment index>"; warning anchors and HTML ids use it
    id: str
    part: DocumentPart
    segments: tuple[ReviewSegment, ...]


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
    outline: tuple[ReviewNode, ...]
    # document order, covering every segment
    items: tuple[ReviewItem, ...]


def build_review_data(result: PipelineResult) -> ReviewData:
    return ReviewData(
        summary=result.summarize(),
        outline=_build_outline(result.logical, result.chunks),
        items=_build_items(result.logical, result.chunks),
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
        label = _node_label(node)
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
    document: LogicalDocument, chunks: Sequence[Chunk]
) -> tuple[ReviewNode, ...]:
    outline: list[ReviewNode] = []
    depths = _node_depths(document.nodes)

    for node in document.nodes:
        outline.append(
            ReviewNode(
                node_id=node.id,
                label=_node_label(node),
                level=node.level,
                depth=depths[node.id],
                is_section=node.is_section,
                overlap_chunk_ids=tuple(
                    c.id for c in chunks if c.start < node.end and node.start < c.end
                ),
            )
        )
    return tuple(outline)


def _build_items(
    document: LogicalDocument, chunks: Sequence[Chunk]
) -> tuple[ReviewItem, ...]:
    items: list[ReviewItem] = []
    depths = _node_depths(document.nodes)
    cursor = 0
    for chunk in chunks:
        if chunk.start > cursor:
            items.extend(_build_gaps(document, cursor, chunk.start, depths))
        items.append(
            ReviewChunk(
                chunk=chunk,
                segments=tuple(
                    _build_segment(
                        document,
                        index,
                        table_rows=chunk.table_rows,
                        depths=depths,
                    )
                    for index in range(chunk.start, chunk.end)
                ),
            )
        )
        cursor = max(cursor, chunk.end)
    items.extend(_build_gaps(document, cursor, len(document.segments), depths))

    return tuple(items)


def _build_gaps(
    document: LogicalDocument, start: int, end: int, depths: dict[str, int]
) -> tuple[ReviewGap, ...]:
    """Segments [start, end) that no chunk covers, one gap per part they fall in"""
    parts: tuple[tuple[DocumentPart, int, int], ...] = (
        ("front_matter", 0, document.body_start),
        ("body", document.body_start, document.body_end),
        ("end_matter", document.body_end, len(document.segments)),
    )

    gaps: list[ReviewGap] = []
    for part, part_start, part_end in parts:
        gap_start, gap_end = max(start, part_start), min(end, part_end)
        if gap_start < gap_end:
            gaps.append(
                ReviewGap(
                    id=f"gap-{gap_start}",
                    part=part,
                    segments=tuple(
                        _build_segment(document, index, None, depths)
                        for index in range(gap_start, gap_end)
                    ),
                )
            )

    return tuple(gaps)


def _build_segment(
    document: LogicalDocument,
    index: int,
    table_rows: tuple[int, int] | None,
    depths: dict[str, int],
) -> ReviewSegment:

    segment = document.segments[index]
    if isinstance(segment, TextSegment):
        review_segment: ReviewSegment = ReviewText(
            index=index, block_id=segment.block_id, text=segment.text
        )
    elif isinstance(segment, ImageSegment):
        review_segment = ReviewImage(index=index, block_id=segment.block_id)
    elif isinstance(segment, HeadingSegment):
        node = document.get_node(segment.node_id)
        review_segment = ReviewHeading(
            index=index,
            block_id=segment.block_id,
            label=_node_label(node),
            level=node.level,
            depth=depths[segment.node_id],
        )
    elif isinstance(segment, TableSegment):
        table = segment.table
        if table_rows is not None:
            start, end = table_rows
        else:
            start, end = table.n_header_rows, table.n_rows
        review_segment = ReviewTable(
            index=index,
            block_id=segment.block_id,
            n_header_rows=table.n_header_rows,
            rows=_build_table_rows(table, start, end),
        )
    else:
        assert_never(segment)
    return review_segment


def _build_table_rows(
    table: TableInfo, start: int, end: int
) -> tuple[tuple[ReviewCell, ...], ...]:
    """The header rows, then body rows [start, end), as render_table_rows show them"""

    shown_rows = [*range(table.n_header_rows), *range(start, end)]

    cells_by_row: dict[int, list[tuple[int, ReviewCell]]] = {
        row: [] for row in shown_rows
    }

    for cell in table.cells:
        cell_end = cell.row + cell.row_span
        if cell.row < table.n_header_rows:
            first_row, row_span = cell.row, cell.row_span
        elif cell.row < end and start < cell_end:
            first_row = max(cell.row, start)
            row_span = min(cell_end, end) - first_row
        else:
            continue
        cells_by_row[first_row].append(
            (
                cell.col,
                ReviewCell(text=cell.text(), row_span=row_span, col_span=cell.col_span),
            )
        )

    return tuple(
        tuple(
            review_cell
            for _, review_cell in sorted(cells_by_row[row], key=lambda pair: pair[0])
        )
        for row in shown_rows
    )


def _node_label(node: LogicalNode) -> str:
    return heading_label(node.marker, node.title) or f"[{node.level}]"


def _node_depths(nodes: Sequence[LogicalNode]) -> dict[str, int]:
    """Depth by the parent chain, roots at 0; nodes in pre-order, so a parent comes before its children"""
    depths: dict[str, int] = {}
    for node in nodes:
        depths[node.id] = (
            depths[node.parent_id] + 1 if node.parent_id is not None else 0
        )
    return depths
