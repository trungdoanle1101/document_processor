from lxml.etree import _Element
from docx.oxml.ns import qn
from ._xml import xpath_str_or_none, xpath_element_or_none, xpath_int_or_none
from typing import Literal
from collections.abc import Callable
from dataclasses import dataclass

from document_processor.core.models.physical import TableInfo

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
        raise ValueError(f"Unrecognized w:vMerge/@w:val {val!r}; expected {VMERGE_RESTART!r}, {VMERGE_CONTINUE!r} or no value")
    return val



@dataclass(frozen=True)
class _RawCell:
    grid_span: int
    v_merge: str | None
    paragraphs: tuple[str, ...]


def _parse_cell(tc: _Element, read_paragraph_func: Callable[[_Element], str | None]) -> _RawCell:
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
            raise ValueError(f"{ref}: Unknown tag {el.tag}")

    return _RawCell(
        grid_span=read_gridspan(tc),
        v_merge=read_vmerge(tc),
        paragraphs=tuple(paragraphs)
    )


@dataclass(frozen=True)
class _RawRow:
    is_header: bool
    grid_before: int
    grid_after: int
    cells: tuple[_RawCell, ...]


def _parse_row(tr: _Element, read_paragraph_func: Callable[[_Element], str | None]) -> _RawRow:
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
            raise ValueError(f"{ref}: Unknown tag {el.tag}")
    return _RawRow(
        is_header=read_tblheader(tr),
        grid_before=read_gridbefore(tr),
        grid_after=read_gridafter(tr),
        cells=tuple(cell_list)
    )

# def _build_table_info(raw_rows: tuple[_RawRow, ...]) -> TableInfo:
    
#     return TableInfo(
#         n_rows=len(raw_rows),
#         # Pending column counting logic
#         n_header_rows=sum(row.is_header for row in raw_rows)
#         # Pending _RawCell -> TableCell logic
#     )

# def parse_table(tbl: _Element, read_paragraph_func: Callable[[_Element], str | None]) -> TableInfo:
#     raw_rows = tuple(_parse_row(el, read_paragraph_func) for el in tbl if el.tag == qn("w:tr"))
#     return _build_table_info(raw_rows)
