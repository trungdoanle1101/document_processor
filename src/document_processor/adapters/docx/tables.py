from lxml.etree import _Element
from docx.oxml.ns import qn
from ._xml import xpath_str_or_none, xpath_element_or_none, xpath_int_or_none
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass

from document_processor.core.models.physical import TableInfo, TableCell

TR_TBLHEADER_XPATH = "./w:trPr/w:tblHeader"
TR_GRIDBEFORE_VAL_XPATH = "./w:trPr/w:gridBefore/@w:val"
TR_GRIDAFTER_VAL_XPATH = "./w:trPr/w:gridAfter/@w:val"
TC_GRIDSPAN_VAL_XPATH = "./w:tcPr/w:gridSpan/@w:val"
TC_VMERGE_XPATH = "./w:tcPr/w:vMerge"
VAL_XPATH = "./@w:val"

VMERGE_RESTART = "restart"
VMERGE_CONTINUE = "continue"


def read_tblheader(tr: _Element) -> bool:
    tblheader_el = xpath_element_or_none(tr, TR_TBLHEADER_XPATH)
    if tblheader_el is None:
        return False
    val = xpath_str_or_none(tblheader_el, VAL_XPATH)
    if (val is None) or (val in {"1", "true", "on"}):
        return True
    if val in {"0", "false", "off"}:
        return False
    raise ValueError(f"Unrecognized w:tblHeader/@w:val {val!r}")


def read_gridbefore(tr: _Element) -> int:
    # Docx default for gridBefore
    default_val = 0
    val = xpath_int_or_none(tr, TR_GRIDBEFORE_VAL_XPATH)
    if val is None:
        return default_val
    if val < default_val:
        raise ValueError(f"gridBefore must be >= {default_val}, got {val}")
    return val


def read_gridafter(tr: _Element) -> int:
    # Docx default for gridAfter
    default_val = 0
    val = xpath_int_or_none(tr, TR_GRIDAFTER_VAL_XPATH)
    if val is None:
        return default_val
    if val < default_val:
        raise ValueError(f"gridAfter must be >= {default_val}, got {val}")
    return val


def read_gridspan(tc: _Element) -> int:
    # Docx default for gridSpan
    default_val = 1
    val = xpath_int_or_none(tc, TC_GRIDSPAN_VAL_XPATH)
    if val is None:
        return default_val
    if val < default_val:
        raise ValueError(f"gridSpan must be >= {default_val}, got {val}")
    return val


def read_vmerge(tc: _Element) -> str | None:
    vmerge_el = xpath_element_or_none(tc, TC_VMERGE_XPATH)
    if vmerge_el is None:
        return None
    val = xpath_str_or_none(vmerge_el, VAL_XPATH)
    if val is None:
        return VMERGE_CONTINUE
    if val != VMERGE_RESTART and val != VMERGE_CONTINUE:
        raise ValueError(
            f"Unrecognized w:vMerge/@w:val {val!r}; expected {VMERGE_RESTART!r}, {VMERGE_CONTINUE!r} or no value"
        )
    return val


@dataclass(frozen=True)
class _RawCell:
    grid_span: int
    v_merge: str | None
    paragraphs: tuple[str, ...]


def _parse_cell(
    tc: _Element, read_paragraph_func: Callable[[_Element], str | None]
) -> _RawCell:
    paragraphs: list[str] = []
    for el in tc:
        if el.tag == qn("w:p"):
            p_text = read_paragraph_func(el)
            if p_text is not None:
                paragraphs.append(p_text)
        elif el.tag == qn("w:tcPr"):
            continue
        elif el.tag == qn("w:tbl"):
            # Postpone handling Nested Table
            ref = el.getroottree().getpath(el)
            raise NotImplementedError(f"{ref}: Nested tables not supported")
        else:
            ref = el.getroottree().getpath(el)
            raise ValueError(f"{ref}: Unknown tag {el.tag!r}")

    return _RawCell(
        grid_span=read_gridspan(tc),
        v_merge=read_vmerge(tc),
        paragraphs=tuple(paragraphs),
    )


@dataclass(frozen=True)
class _RawRow:
    is_header: bool
    grid_before: int
    grid_after: int
    cells: tuple[_RawCell, ...]


def _parse_row(
    tr: _Element, read_paragraph_func: Callable[[_Element], str | None]
) -> _RawRow:
    cell_list: list[_RawCell] = []
    for el in tr:
        if el.tag == qn("w:tc"):
            parsed = _parse_cell(el, read_paragraph_func)
            cell_list.append(parsed)
        elif el.tag == qn("w:trPr") or el.tag == qn("w:tblPrEx"):
            # Skip deliberately
            continue
        else:
            ref = el.getroottree().getpath(el)
            raise ValueError(f"{ref}: Unknown tag {el.tag!r}")
    return _RawRow(
        is_header=read_tblheader(tr),
        grid_before=read_gridbefore(tr),
        grid_after=read_gridafter(tr),
        cells=tuple(cell_list),
    )


@dataclass
class _CellBuilder:
    row: int
    col: int
    row_span: int
    col_span: int
    paragraphs: tuple[str, ...]

    def build(self) -> TableCell:
        return TableCell(
            row=self.row,
            col=self.col,
            row_span=self.row_span,
            col_span=self.col_span,
            paragraphs=self.paragraphs,
        )


def _build_table_info(raw_rows: tuple[_RawRow, ...]) -> TableInfo | None:
    # Count header rows
    n_header_rows = 0
    for raw_row in raw_rows:
        if not raw_row.is_header:
            break
        n_header_rows += 1

    # Count n_cols
    n_cols = 0
    for raw_row in raw_rows:
        # Number of cells row has
        row_width = raw_row.grid_before + raw_row.grid_after
        for cell in raw_row.cells:
            row_width += cell.grid_span
        if n_cols < row_width:
            n_cols = row_width
    if n_cols == 0:
        return None

    # Builder: list of "mutable" _CellBuilder's
    builders: list[_CellBuilder] = []
    prev_open: list[_CellBuilder | None] = [None] * n_cols
    for r_idx, raw_row in enumerate(raw_rows):
        cur_open: list[_CellBuilder | None] = [None] * n_cols

        # Fill all the cells at position before gridBefore with empty _CellBuilder
        for c in range(raw_row.grid_before):
            builders.append(
                _CellBuilder(row=r_idx, col=c, row_span=1, col_span=1, paragraphs=())
            )

        # First cell's column position starts at gridBefore
        c_idx = raw_row.grid_before
        for raw_cell in raw_row.cells:
            if raw_cell.v_merge is None or raw_cell.v_merge == VMERGE_RESTART:
                # Create new mutable cell
                cell_builder = _CellBuilder(
                    row=r_idx,
                    col=c_idx,
                    row_span=1,
                    col_span=raw_cell.grid_span,
                    paragraphs=raw_cell.paragraphs,
                )

                builders.append(cell_builder)
                if raw_cell.v_merge == VMERGE_RESTART:
                    cur_open[c_idx : c_idx + raw_cell.grid_span] = [
                        cell_builder
                    ] * raw_cell.grid_span

            elif raw_cell.v_merge == VMERGE_CONTINUE:
                prev_open_cell: _CellBuilder | None = prev_open[c_idx]
                if prev_open_cell is None:
                    raise ValueError(
                        f"row {r_idx}, col {c_idx}: encountered vMerge={VMERGE_CONTINUE!r} when at column {c_idx} without open cell"
                    )
                else:
                    if raw_cell.grid_span != prev_open_cell.col_span:
                        raise ValueError(
                            f"row {r_idx}, col {c_idx}: continue vmerging from previous cell with different gridSpan. Previous: {prev_open_cell.col_span}, current: {raw_cell.grid_span}"
                        )
                    prev_open_cell.row_span += 1
                    # Content of a cell with vMerge = 'continue' is deliberately dropped
                    cur_open[c_idx : c_idx + raw_cell.grid_span] = [
                        prev_open_cell
                    ] * raw_cell.grid_span

            # Track the column position of next cell
            c_idx += raw_cell.grid_span

        # Fill all remaining positions with empty _CellBuilder
        for c in range(c_idx, n_cols):
            builders.append(
                _CellBuilder(row=r_idx, col=c, row_span=1, col_span=1, paragraphs=())
            )

        prev_open = cur_open

    builders = sorted(builders, key=lambda b: (b.row, b.col))
    cells = tuple(b.build() for b in builders)

    return TableInfo(
        n_rows=len(raw_rows), n_cols=n_cols, n_header_rows=n_header_rows, cells=cells
    )


def parse_table(
    tbl: _Element, read_paragraph_func: Callable[[_Element], str | None]
) -> TableInfo | None:
    raw_rows: list[_RawRow] = []
    for el in tbl:
        if el.tag == qn("w:tr"):
            parsed = _parse_row(el, read_paragraph_func)
            raw_rows.append(parsed)
        elif el.tag == qn("w:tblPr") or el.tag == qn("w:tblGrid"):
            # Skip deliberately
            continue
        else:
            ref = el.getroottree().getpath(el)
            raise ValueError(f"{ref}: Unknown tag {el.tag!r}")

    return _build_table_info(tuple(raw_rows))
