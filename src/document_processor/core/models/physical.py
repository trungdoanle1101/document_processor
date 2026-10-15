from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from document_processor.core.identity import HASH_ALGORITHM
from document_processor.core.text_utils import is_invisible, to_single_line

from .ranges import TextRange

_TABLE_COL_SEP = "\t"
_TABLE_ROW_SEP = "\n"


class DocSourceFormat(Enum):
    DOCX = "docx"
    PDF = "pdf"
    MARKDOWN = "md"
    TXT = "txt"


class BlockType(Enum):
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    IMAGE = "image"
    TABLE = "table"


class Region(Enum):
    BODY = "body"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"


class ParagraphAlignment(Enum):
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    JUSTIFIED = "justified"


class SpanStyle(Enum):
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"
    STRIKETHROUGH = "strikethrough"


@dataclass(frozen=True, kw_only=True)
class DeclaredNumbering:
    """Numbering format as declared by the source (e.g. docx numbering.xml)
    Absent for sources that only have rendered marker text

    Describes the declaration, not what is shown: a level can declare a
    format yet display nothing (e.g. docx with no lvlText). Use
    ListInfo.displayed_marker for what the reader actually sees.
    """
    num_fmt: str # "decimal", "lowerLetter", "bullet", ...
    segment_formats: tuple[str, ...] # One per placeholder, in order

    def __post_init__(self) -> None:
        if not self.num_fmt.strip():
            raise ValueError("Empty num_fmt not allowed")
        if any(not f.strip() for f in self.segment_formats):
            raise ValueError("Empty segment format not allowed")



@dataclass(frozen=True, kw_only=True)
class ListInfo:
    id: str
    is_ordered: bool
    displayed_marker: str | None
    nested_level: int  # Starting at 0
    declared_numbering: DeclaredNumbering | None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Empty ListInfo id not allowed")
        
        if self.nested_level < 0:
            raise ValueError(f"ListInfo {self.id!r}: required: nested_level >= 0") 


IMAGE_MIME_TYPE_PATTERN = re.compile(r"image/[a-z0-9][a-z0-9.+-]*")


@dataclass(frozen=True)
class ImageInfo:
    source_path: str
    mime_type: str
    content_hash: str
    displayed_width_pt: float | None
    displayed_height_pt: float | None

    def __post_init__(self) -> None:
        if is_invisible(self.source_path):
            raise ValueError(f"Invisible source_path: {self.source_path!r}")
        if IMAGE_MIME_TYPE_PATTERN.fullmatch(self.mime_type) is None:
            raise ValueError(
                f"Invalid mime_type: {self.mime_type!r}: expected 'image/<subtype>'"
            )
        if not self.content_hash.startswith(f"{HASH_ALGORITHM}:"):
            raise ValueError(
                f"Invalid content_hash: {self.content_hash!r} - Must start with '{HASH_ALGORITHM}:'"
            )

        if self.displayed_height_pt is not None and self.displayed_height_pt <= 0:
            raise ValueError(
                f"displayed_height_pt must be > 0, got {self.displayed_height_pt}"
            )

        if self.displayed_width_pt is not None and self.displayed_width_pt <= 0:
            raise ValueError(
                f"displayed_width_pt must be > 0, got {self.displayed_width_pt}"
            )


@dataclass(frozen=True, kw_only=True)
class TableCell:
    row: int
    col: int
    row_span: int
    col_span: int
    paragraphs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError(f"row must be >= 0, got {self.row}")
        if self.col < 0:
            raise ValueError(f"col must be >= 0, got {self.col}")
        if self.row_span < 1:
            raise ValueError(f"row_span must be >= 1, got {self.row_span}")
        if self.col_span < 1:
            raise ValueError(f"col_span must be >= 1, got {self.col_span}")
        for i, p in enumerate(self.paragraphs):
            if is_invisible(p):
                raise ValueError(
                    f"TableCell ({self.row}, {self.col}). Paragraph index {i} contains all invisible text."
                )

    def text(self) -> str:
        return " ".join(to_single_line(p) for p in self.paragraphs)


@dataclass(frozen=True, kw_only=True)
class TableInfo:
    n_rows: int
    n_cols: int
    n_header_rows: int
    cells: tuple[TableCell, ...]

    def __post_init__(self) -> None:
        if self.n_rows <= 0:
            raise ValueError(f"n_rows must be > 0, got {self.n_rows}")
        if self.n_cols <= 0:
            raise ValueError(f"n_cols must be > 0, got {self.n_cols}")
        if not (0 <= self.n_header_rows <= self.n_rows):
            raise ValueError(
                f"Invalid number of header rows. Required: 0 <= n_header_rows <= n_rows, got {self.n_header_rows} with n_rows={self.n_rows}"
            )

        # Check valid spanning
        for cell in self.cells:
            if cell.row + cell.row_span > self.n_rows:
                raise ValueError(
                    f"TableCell ({cell.row}, {cell.col}). Cell spanning out of row limit: Required cell.row + cell.row_span <= n_rows, got {cell.row + cell.row_span} > {self.n_rows}"
                )
            if cell.col + cell.col_span > self.n_cols:
                raise ValueError(
                    f"TableCell ({cell.row}, {cell.col}). Cell spanning out of col limit: Required cell.col + cell.col_span <= n_cols, got {cell.col + cell.col_span} > {self.n_cols}"
                )

        # Check cells reading order
        for i in range(len(self.cells) - 1):
            cur = self.cells[i]
            nxt = self.cells[i + 1]
            if (nxt.row, nxt.col) <= (cur.row, cur.col):
                raise ValueError(
                    f"Cells must follow reading order. Got ({cur.row}, {cur.col}) before ({nxt.row}, {nxt.col})"
                )

        # Check coverage
        self.grid()

    def _build_grid(self) -> list[list[TableCell | None]]:
        grid_: list[list[TableCell | None]] = [
            [None for _ in range(self.n_cols)] for _ in range(self.n_rows)
        ]

        for cell in self.cells:
            for i in range(cell.row_span):
                for j in range(cell.col_span):
                    r = cell.row + i
                    c = cell.col + j
                    existing = grid_[r][c]
                    if existing is not None:
                        raise ValueError(
                            f"Overlap detected: cell ({cell.row}, {cell.col}) overlaps cell ({existing.row}, {existing.col}) at position ({r}, {c})"
                        )
                    grid_[r][c] = cell
        return grid_

    def grid(self) -> tuple[tuple[TableCell, ...], ...]:
        grid_ = self._build_grid()
        num_holes = 0
        first_hole: tuple[int, int] | None = None
        grid_temp: list[tuple[TableCell, ...]] = []
        for i, row in enumerate(grid_):
            temp_row: list[TableCell] = []
            for j, slot in enumerate(row):
                if slot is None:
                    num_holes += 1
                    if first_hole is None:
                        first_hole = (i, j)
                else:
                    temp_row.append(slot)
            grid_temp.append(tuple(temp_row))

        if num_holes != 0:
            raise ValueError(
                f"{num_holes} positions not covered. First missing position: {first_hole}"
            )
        return tuple(grid_temp)

    def text(self) -> str:
        grid_ = self.grid()
        builder: list[str] = []
        for i, row in enumerate(grid_):
            row_builder: list[str] = []
            for j, cell in enumerate(row):
                if cell.row == i and cell.col == j:
                    row_builder.append(cell.text())
                else:
                    row_builder.append("")
            row_str = _TABLE_COL_SEP.join(row_builder)
            builder.append(row_str)
        return _TABLE_ROW_SEP.join(builder)


@dataclass(frozen=True)
class ParagraphFormat:
    alignment: ParagraphAlignment | None
    style_name: str | None


@dataclass(frozen=True)
class CharSpan:
    """
    Each span starts at 0 <= start < end <= length of text
    """

    style: SpanStyle
    text_range: TextRange


@dataclass(frozen=True, kw_only=True)
class PhysicalBlock:
    """
    Physical representation of a block
    """

    # Basic information
    id: str
    block_type: BlockType
    text: str

    # Location
    region: Region
    page: int | None  # Starting at 1

    # Formatting
    paragraph_format: ParagraphFormat
    char_spans: tuple[CharSpan, ...]

    # Only if the block is a list item
    list_info: ListInfo | None = None

    # Only if the block is an image
    image_info: ImageInfo | None = None

    # Only if the block is a table
    table_info: TableInfo | None = None

    # Provenance
    source_reference: str

    # Changelog
    changelog: tuple[str, ...]

    def __post_init__(self) -> None:
        # Text must be visible, except for IMAGE
        if self.block_type != BlockType.IMAGE and is_invisible(self.text):
            raise ValueError(f"Block {self.id}: Contains only invisible/empty text")

        # Check orphaned LIST_ITEM block (no list_info)
        if self.block_type == BlockType.LIST_ITEM and self.list_info is None:
            raise ValueError(
                f"Block {self.id}: Invalid {BlockType.LIST_ITEM.value} block: Doesn't contain list_info"
            )

        # list_info must not be present in blocks that are not LIST_ITEM
        if self.block_type != BlockType.LIST_ITEM and self.list_info is not None:
            raise ValueError(
                f"Block {self.id}: Non-LIST_ITEM block must not contain list_info"
            )

        # Check orphaned IMAGE block (no image_info)
        if self.block_type == BlockType.IMAGE and self.image_info is None:
            raise ValueError(
                f"Block {self.id}: Invalid {BlockType.IMAGE.value} block: Doesn't contain image_info"
            )

        # image_info must not be present in blocks that are not IMAGE
        if self.block_type != BlockType.IMAGE and self.image_info is not None:
            raise ValueError(
                f"Block {self.id}: Non-IMAGE block must not contain image_info"
            )

        if self.block_type == BlockType.TABLE:
            # Check orphaned TABLE block (no table_info)
            if self.table_info is None:
                raise ValueError(
                    f"Block {self.id}: Invalid {BlockType.TABLE.value} block: Doesn't contain table_info"
                )
            # Check block's text
            table_info_text = self.table_info.text()
            if self.text != table_info_text:
                raise ValueError(f"Block {self.id}: a table block's text must match")

        # table_info must not be present in blocks that are not TABLE
        if self.block_type != BlockType.TABLE and self.table_info is not None:
            raise ValueError(
                f"Block {self.id}: Non-TABLE block must not contain table_info"
            )

        # Check spans length definition
        for i, span in enumerate(self.char_spans):
            end = span.text_range.end
            if end > len(self.text):
                raise ValueError(
                    f"Block {self.id}: char span {i} ends at {end}, past text length {len(self.text)}"
                )


@dataclass(frozen=True)
class PhysicalDocument:
    """
    Class representing the entire document
    """

    id: str

    source_filename: str
    source_format: DocSourceFormat

    adapter_name: str
    adapter_version: str

    file_hash: str
    blocks: tuple[PhysicalBlock, ...]

    def __post_init__(self) -> None:
        # Check block ids uniqueness
        seen_ids: set[str] = set()
        for i, b in enumerate(self.blocks):
            if b.id in seen_ids:
                raise ValueError(
                    f"PhysicalDocument {self.source_filename!r}: Duplicate block id {b.id!r} at position {i}"
                )
            seen_ids.add(b.id)
