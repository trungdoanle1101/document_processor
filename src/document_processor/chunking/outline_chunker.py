from dataclasses import dataclass
from typing import Literal

from document_processor.chunking.chunk import Chunk, PathStep
from document_processor.chunking.render import render_segments
from document_processor.chunking.tokenizer import Tokenizer
from document_processor.core.models import LogicalDocument, DocumentPart, TableSegment, TextSegment


@dataclass(frozen=True, kw_only=True)
class _Piece:
    part: DocumentPart
    owner_id: str | None
    start: int
    end: int
    table_rows: tuple[int, int] | None = None
    text_span: tuple[int, int] | None = None


@dataclass(frozen=True, kw_only=True)
class OutlineChunker:
    tokenizer: Tokenizer
    max_tokens: int
    include_breadcrumb_in_size: bool = False
    table_rendering: Literal["rows", "rows_with_headers"] = "rows"

    def __post_init__(self) -> None:
        error_owner = type(self).__name__
        if self.max_tokens <= 0:
            raise ValueError(
                f"{error_owner}: required max_tokens > 0, got max_tokens={self.max_tokens}"
            )

    def chunk(self, document: LogicalDocument) -> tuple[Chunk, ...]:
        pieces = [
            *self._chunk_range(document, "front_matter", 0, document.body_start, owner_id=None),
            # *self._chunk_children(document, "body", None, document.body_start, document.body_end, document.root_ids),
            *self._chunk_range(document, "body", document.body_start, document.body_end, owner_id=None),
            *self._chunk_range(document, "end_matter", document.body_end, len(document.segments), owner_id=None)
        ]
        kept = [piece for piece in pieces if _has_content(document, piece.start, piece.end)]
        return tuple(_get_chunk_from_piece(document, piece, index) for index, piece in enumerate(kept))


    def _fits(self, document: LogicalDocument, owner_id: str | None, start: int, end: int) -> bool:
        text = render_segments(document, start, end)
        if self.include_breadcrumb_in_size:
            path = _path_to(document, owner_id)
            headings = [step.label for step in path if step.start < start]
            text = "\n".join((*headings, text))
        return self.tokenizer.count_tokens(text) <= self.max_tokens


    def _chunk_range(
        self,
        document: LogicalDocument,
        part: DocumentPart,
        start: int,
        end: int,
        owner_id: str | None,
    ) -> list[_Piece]:
        pieces: list[_Piece] = []
        piece_start = start
        for index in range(start, end):
            if index > piece_start and not self._fits(document, owner_id, piece_start, index + 1):
                pieces.append(_Piece(
                    part=part,
                    owner_id=owner_id,
                    start=piece_start,
                    end=index
                ))
                piece_start = index
        if piece_start < end:
            pieces.append(_Piece(
                part=part,
                owner_id=owner_id,
                start=piece_start,
                end=end
            ))
        return pieces


def _get_chunk_from_piece(document: LogicalDocument, piece: _Piece, index: int) -> Chunk:
    text = render_segments(document, piece.start, piece.end)
    # unique block_ids, in segment order
    block_ids = tuple(dict.fromkeys(segment.block_id for segment in document.segments[piece.start:piece.end]))

    path = _path_to(document, piece.owner_id)
    
    return Chunk(
        id=f"c{index}",
        logical_document_id=document.id,
        part=piece.part,
        path=path,
        start=piece.start,
        end=piece.end,
        table_rows=piece.table_rows,
        text_span=piece.text_span,
        text=text,
        block_ids=block_ids
    )


def _path_to(document: LogicalDocument, node_id: str | None) -> tuple[PathStep, ...]:

    steps: list[PathStep] = []
    while node_id is not None:
        node = document.get_node(node_id)
        steps.append(PathStep(
            node_id=node_id,
            level=node.level,
            number=node.number,
            marker=node.marker,
            title=node.title,
            start=node.start
        ))
        node_id = node.parent_id
    return tuple(reversed(steps))


def _has_content(document: LogicalDocument, start: int, end: int) -> bool:
    """Check if the segment range contains a TextSegment or a TableSegment
    """
    for segment in document.segments[start:end]:
        if isinstance(segment, (TextSegment, TableSegment)):
            return True
    return False