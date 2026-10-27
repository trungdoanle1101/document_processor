from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from document_processor.chunking.chunk import Chunk, PathStep
from document_processor.chunking.render import (
    TableLayout,
    render_segments,
    render_table_rows,
)
from document_processor.chunking.tokenizer import Tokenizer
from document_processor.core.models import (
    DocumentPart,
    LogicalDocument,
    LogicalNode,
    TableInfo,
    TableSegment,
    TextSegment,
)


@dataclass(frozen=True, kw_only=True)
class _Piece:
    part: DocumentPart
    owner_id: str | None
    start: int
    end: int
    table_rows: tuple[int, int] | None = None
    text_span: tuple[int, int] | None = None


class _ChunkingRun:
    """One chunker applied to one document"""

    def __init__(self, settings: OutlineChunker, document: LogicalDocument) -> None:
        self.settings = settings
        self.document = document

    def chunk(self) -> tuple[Chunk, ...]:
        pieces = [
            *self._split_by_size(
                _Piece(
                    part="front_matter",
                    owner_id=None,
                    start=0,
                    end=self.document.body_start,
                )
            ),
            *self._split_inside(None),
            *self._split_by_size(
                _Piece(
                    part="end_matter",
                    owner_id=None,
                    start=self.document.body_end,
                    end=len(self.document.segments),
                )
            ),
        ]
        kept = [
            piece
            for piece in pieces
            if _has_content(self.document, piece.start, piece.end)
        ]
        return tuple(self._make_chunk(piece, index) for index, piece in enumerate(kept))

    def _split_by_size(self, span: _Piece) -> list[_Piece]:
        """Split a span at segment boundaries, greedily, up to the budget"""
        pieces: list[_Piece] = []
        piece_start = span.start
        for index in range(span.start, span.end):
            if index > piece_start and not self._fits(
                replace(span, start=piece_start, end=index + 1)
            ):
                pieces.append(replace(span, start=piece_start, end=index))
                piece_start = index
        if piece_start < span.end:
            pieces.append(replace(span, start=piece_start))

        result: list[_Piece] = []
        for piece in pieces:
            if piece.end - piece.start == 1 and not self._fits(piece):
                result.extend(self._split_too_big_piece(piece))
            else:
                result.append(piece)
        return result

    def _split_inside(self, owner_id: str | None) -> list[_Piece]:
        """Split a span of body segments belonging to node with id = owner_id"""
        # Determine the range of segments in the document
        if owner_id is None:
            start, end = self.document.body_start, self.document.body_end
        else:
            owner = self.document.get_node(owner_id)
            start, end = owner.start, owner.end

        # Build a list of items consisting of plain segments or children nodes
        child_ids = self.document.get_children(owner_id)
        cursor = start
        items: list[_ChildOrSegment] = []
        for child_id in child_ids:
            child = self.document.get_node(child_id)
            items.extend(
                _ChildOrSegment(start=index, end=index + 1, child=None)
                for index in range(cursor, child.start)
            )
            items.append(_ChildOrSegment(start=child.start, end=child.end, child=child))
            cursor = child.end
        items.extend(
            _ChildOrSegment(start=index, end=index + 1, child=None)
            for index in range(cursor, end)
        )

        pieces: list[_Piece] = []
        group: list[_ChildOrSegment] = []
        for item in items:
            # A section is never grouped: close the group, chunk the section on its own
            if (
                self.settings.keep_sections_apart
                and item.child is not None
                and item.child.is_section
            ):
                if group:
                    pieces.append(_make_piece(group, owner_id))
                    group = []
                pieces.extend(self._chunk_node(item.child))
                continue

            # Extend the open group if it still fits; otherwise close it before processing the new item
            if group:
                piece = _Piece(
                    part="body", owner_id=owner_id, start=group[0].start, end=item.end
                )
                if self._fits(piece):
                    group.append(item)
                    continue

                pieces.append(_make_piece(group, owner_id))
                group = []

            # Start a new group with this item, or handle it alone if it's too big
            # a child is chunked inside; a segment becomes its one piece
            piece = _Piece(
                part="body", owner_id=owner_id, start=item.start, end=item.end
            )
            if self._fits(piece):
                group = [item]
            elif item.child is not None:
                pieces.extend(self._chunk_node(item.child))
            else:
                pieces.extend(self._split_too_big_piece(piece))

        # close the remaining items
        if group:
            pieces.append(_make_piece(group, owner_id))

        return pieces

    def _split_too_big_piece(self, piece: _Piece) -> list[_Piece]:
        """A one-segment piece over budget: tables split by rows, anything else stays whole"""
        segment = self.document.segments[piece.start]
        if isinstance(segment, TableSegment):
            return self._split_table_rows(piece=piece, table=segment.table)
        return [piece]

    def _chunk_node(self, node: LogicalNode) -> list[_Piece]:
        child_ids = self.document.get_children(node.id)
        must_split = self.settings.keep_sections_apart and any(
            self.document.get_node(child_id).is_section for child_id in child_ids
        )

        node_piece = _Piece(
            part="body", owner_id=node.id, start=node.start, end=node.end
        )

        # if not required to split, and satisfies chunk budget
        if not must_split and self._fits(node_piece):
            return [node_piece]

        # otherwise, must split (either to keep sections apart, or not meeting chunk budget)
        # if has no child
        if not child_ids:
            return self._split_by_size(node_piece)

        return self._split_inside(node.id)

    def _split_table_rows(self, piece: _Piece, table: TableInfo) -> list[_Piece]:
        """Greedy groups of body rows; header rows are rendered with every group"""
        first, last = table.n_header_rows, table.n_rows
        if first == last:
            return [piece]
        pieces: list[_Piece] = []
        group_start = first

        for row in range(first, last):
            if row > group_start and not self._fits(
                replace(piece, table_rows=(group_start, row + 1))
            ):
                pieces.append(replace(piece, table_rows=(group_start, row)))
                group_start = row
        pieces.append(replace(piece, table_rows=(group_start, last)))

        return pieces

    def _fits(self, piece: _Piece) -> bool:
        return self._count_piece_tokens(piece) <= self.settings.max_tokens

    def _make_chunk(self, piece: _Piece, chunk_index: int) -> Chunk:
        # unique block_ids, in segment order
        block_ids = tuple(
            dict.fromkeys(
                segment.block_id
                for segment in self.document.segments[piece.start : piece.end]
            )
        )

        return Chunk(
            id=f"c{chunk_index}",
            logical_document_id=self.document.id,
            part=piece.part,
            text=self._render_piece(piece),
            token_count=self._count_piece_tokens(piece),
            path=_path_to(
                self.document,
                piece.owner_id,
                start=piece.start,
                lead_in_words=self.settings.breadcrumb_lead_in_words,
            ),
            start=piece.start,
            end=piece.end,
            table_rows=piece.table_rows,
            text_span=piece.text_span,
            block_ids=block_ids,
        )

    def _count_piece_tokens(self, piece: _Piece) -> int:
        text = self._render_piece(piece)
        if self.settings.include_breadcrumb_in_size:
            path = _path_to(
                self.document,
                piece.owner_id,
                start=piece.start,
                lead_in_words=self.settings.breadcrumb_lead_in_words,
            )
            headings = [step.label for step in path if step.start < piece.start]
            text = "\n".join((*headings, text))
        return self.settings.tokenizer.count_tokens(text)

    def _render_piece(self, piece: _Piece) -> str:
        if piece.table_rows is None:
            return render_segments(
                document=self.document,
                start=piece.start,
                end=piece.end,
                table_layout=self.settings.table_layout,
            )

        table_segment = self.document.segments[piece.start]
        assert isinstance(table_segment, TableSegment)
        return render_table_rows(
            table_segment.table,
            piece.table_rows[0],
            piece.table_rows[1],
            self.settings.table_layout,
        )


@dataclass(frozen=True)
class _ChildOrSegment:
    start: int
    end: int
    # None = a gap segment of the owner
    child: LogicalNode | None


@dataclass(frozen=True, kw_only=True)
class OutlineChunker:
    tokenizer: Tokenizer
    max_tokens: int
    include_breadcrumb_in_size: bool = False
    keep_sections_apart: bool = True
    breadcrumb_lead_in_words: int = 40
    table_layout: TableLayout = "grid"

    def __post_init__(self) -> None:
        error_owner = type(self).__name__
        if self.max_tokens <= 0:
            raise ValueError(
                f"{error_owner}: required max_tokens > 0, got max_tokens={self.max_tokens}"
            )

        if self.breadcrumb_lead_in_words < 0:
            raise ValueError(
                f"{error_owner}: required breadcrumb_lead_in_words >= 0, got breadcrumb_lead_in_words={self.breadcrumb_lead_in_words}"
            )

    def chunk(self, document: LogicalDocument) -> tuple[Chunk, ...]:
        return _ChunkingRun(settings=self, document=document).chunk()


def _make_piece(group: Sequence[_ChildOrSegment], owner_id: str | None) -> _Piece:
    if not group:
        raise ValueError("can only get piece from non-empty group")
    if len(group) == 1 and group[0].child is not None:
        owner_id = group[0].child.id
    return _Piece(
        part="body", owner_id=owner_id, start=group[0].start, end=group[-1].end
    )


def _path_to(
    document: LogicalDocument, node_id: str | None, start: int, lead_in_words: int
) -> tuple[PathStep, ...]:
    """Ancestors of node_id, root first; lead-ins only for nodes whose opening words come before start"""
    steps: list[PathStep] = []
    while node_id is not None:
        node = document.get_node(node_id)
        lead_in = _lead_in(document, node, start, lead_in_words)
        steps.append(
            PathStep(
                node_id=node_id,
                level=node.level,
                number=node.number,
                marker=node.marker,
                title=node.title,
                start=node.start,
                lead_in=lead_in,
            )
        )
        node_id = node.parent_id
    return tuple(reversed(steps))


def _has_content(document: LogicalDocument, start: int, end: int) -> bool:
    """Check if the segment range contains a TextSegment or a TableSegment"""
    for segment in document.segments[start:end]:
        if isinstance(segment, (TextSegment, TableSegment)):
            return True
    return False


def _cap_words(text: str, n: int) -> str:
    words = text.split()
    chosen_words = words[:n]
    suffix = "…" if n < len(words) else ""
    return " ".join(chosen_words) + suffix


def _lead_in(document: LogicalDocument, node: LogicalNode, start: int, words: int) -> str | None:
    """Opening words of a node without a title, when they come before the chunk at start"""
    index = node.start + 1
    if node.title is not None or words == 0 or not (index < node.end and index < start):
        return None
    segment = document.segments[index]
    return _cap_words(segment.text, words) if isinstance(segment, TextSegment) else None