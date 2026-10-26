from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

from document_processor.chunking import Chunk
from document_processor.core.models import DocumentPart, LogicalDocument, heading_label

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
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class ReviewData:
    summary: ResultSummary
    outline: tuple[ReviewNode, ...]
    # document order, covering every segment
    items: tuple[ReviewItem, ...]


def build_review_data(result: PipelineResult) -> ReviewData:
    summary = result.summarize()
    return ReviewData(
        summary=summary,

    )


def format_outline(
    document: LogicalDocument, sections_only: bool = False, indent: int = 2
) -> str:
    outline_builder: list[str] = []

    if document.body_start == 0:
        outline_builder.append("front matter  none")
    else:
        outline_builder.append(f"front matter  [0, {document.body_start})")

    depth: dict[str | None, int] = {None: -1}

    for node in document.nodes:
        depth[node.id] = depth[node.parent_id] + 1
        if sections_only and not node.is_section:
            continue
        label = heading_label(node.marker, node.title) or f"[{node.level}]"
        line = f"{indent * depth[node.id] * ' '}{label}  [{node.start}, {node.end})"
        outline_builder.append(line)

    if document.body_end == len(document.segments):
        outline_builder.append("end matter  none")
    else:
        outline_builder.append(
            f"end matter  [{document.body_end}, {len(document.segments)})"
        )

    return "\n".join(outline_builder)
