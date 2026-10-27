from dataclasses import dataclass, field

from document_processor.core.models import DocumentPart, heading_label
from document_processor.core.text_utils import is_invisible


@dataclass(frozen=True, kw_only=True)
class PathStep:
    """One ancestor of a chunk, from the logical document's outline"""

    node_id: str
    # level name as defined in the profile used for the logical document
    level: str
    number: str | None
    marker: str | None
    title: str | None
    start: int
    # opening words of a node without a title, capped; None when it has a title.
    lead_in: str | None = None

    @property
    def label(self) -> str:
        return heading_label(self.marker, self.title or self.lead_in)

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(
                f"PathStep for {self.node_id!r} at level {self.level!r}: required start >= 0, got start={self.start}"
            )
        if self.lead_in is not None:
            if self.title is not None:
                raise ValueError(
                    f"PathStep for {self.node_id!r} at level {self.level!r}: title and lead-in cannot both be set"
                )
            if is_invisible(self.lead_in):
                raise ValueError(
                    f"PathStep for {self.node_id!r} at level {self.level!r}: lead-in is empty or contains invisible text"
                )


@dataclass(frozen=True, kw_only=True)
class Chunk:
    id: str
    logical_document_id: str

    # "front_matter" |  "body" | "end_matter"
    part: DocumentPart

    # 'headings': tuple of path labels displayed to be more readable; derived from 'path'
    headings: tuple[str, ...] = field(init=False, compare=False)

    # text content
    text: str

    # token count according to the tokenizer
    token_count: int

    # root first, owner last
    path: tuple[PathStep, ...]

    # covers segments range [start, end) in LogicalDocument
    # optionally narrowed inside one segment for long segments or table
    start: int
    end: int

    # rows [a, b) of the table segment at 'start'
    table_rows: tuple[int, int] | None = None

    # characters [a,b) of the text at segment 'start'
    text_span: tuple[int, int] | None = None

    # provenance back to original blocks
    block_ids: tuple[str, ...]

    @property
    def node_id(self) -> str | None:
        return self.path[-1].node_id if self.path else None

    def contextualize(self) -> str:
        not_in_text = (step.label for step in self.path if step.start < self.start)
        return "\n".join((*not_in_text, self.text))

    def __post_init__(self) -> None:
        error_owner = f"chunk {self.id!r}"

        if self.token_count < 0:
            raise ValueError(f"{error_owner}: required token_count >= 0, got token_count={self.token_count}")
        
        if not (0 <= self.start < self.end):
            raise ValueError(
                f"{error_owner}: required 0 <= start < end, got start={self.start}, end={self.end}"
            )

        if self.part != "body" and self.path:
            raise ValueError(
                f"{error_owner}: a chunk part = 'front_matter' or 'end_matter' should not have node path defined"
            )

        if not all(
            self.path[i].start < self.path[i + 1].start
            for i in range(len(self.path) - 1)
        ):
            raise ValueError(
                f"{error_owner}: path starts must be in strictly increasing order"
            )

        if self.path and self.path[-1].start > self.start:
            raise ValueError(
                f"{error_owner}: owner must not start after the chunk: owner start={self.path[-1].start} > chunk start={self.start}"
            )

        if self.table_rows is not None and self.text_span is not None:
            raise ValueError(
                f"{error_owner}: a chunk cannot simultaneously have text_span and table_rows"
            )

        if ((self.table_rows is not None) or (self.text_span is not None)) and (
            self.end != self.start + 1
        ):
            raise ValueError(
                f"{error_owner}: end == start + 1 required if table_rows or text_span is defined"
            )

        if self.table_rows is not None and not (
            0 <= self.table_rows[0] < self.table_rows[1]
        ):
            raise ValueError(
                f"{error_owner}: required 0 <= self.table_rows[0] < self.table_rows[1], got [0]={self.table_rows[0]}, [1]={self.table_rows[1]} "
            )

        if self.text_span is not None and not (
            0 <= self.text_span[0] < self.text_span[1]
        ):
            raise ValueError(
                f"{error_owner}: required 0 <= self.text_span[0] < self.text_span[1], got [0]={self.text_span[0]}, [1]={self.text_span[1]} "
            )

        object.__setattr__(
            self,
            "headings",
            tuple(step.label for step in self.path),
        )
