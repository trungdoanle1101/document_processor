"""Tests for reading docx tables into TableInfo.

Tables are built from small XML strings. Cell paragraphs are read with a
simple callback (the run text, or None when invisible), so these tests cover
the table logic only; numbering and real paragraph text are covered by the
adapter tests.
"""

from collections.abc import Callable
from typing import Any

import pytest
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml.etree import _Element

from document_processor.adapters.docx.tables import (
    VMERGE_CONTINUE,
    VMERGE_RESTART,
    parse_table,
    read_gridafter,
    read_gridbefore,
    read_gridspan,
    read_tblheader,
    read_vmerge,
)
from document_processor.core.models.physical import TableCell, TableInfo
from document_processor.core.text_utils import is_invisible

W = nsdecls("w")


# ---------- builders


def read_paragraph(p: _Element) -> str | None:
    """Stand-in for the reader's callback: run text, None when invisible."""
    text = "".join(t.text or "" for t in p.iter(qn("w:t")))
    return None if is_invisible(text) else text


def p(text: str = "") -> str:
    return (
        f"<w:p><w:r><w:t xml:space='preserve'>{text}</w:t></w:r></w:p>"
        if text
        else "<w:p/>"
    )


def tc(
    *paragraphs: str, span: int | None = None, vmerge: str | None = None, raw: str = ""
) -> str:
    """One <w:tc>. `vmerge`: None (absent), "" (bare <w:vMerge/>), or a w:val."""
    props = ""
    if span is not None:
        props += f'<w:gridSpan w:val="{span}"/>'
    if vmerge == "":
        props += "<w:vMerge/>"
    elif vmerge is not None:
        props += f'<w:vMerge w:val="{vmerge}"/>'
    tcpr = f"<w:tcPr>{props}</w:tcPr>" if props else ""
    body = (
        "".join(p(x) for x in paragraphs) or p()
    )  # Word: a cell always has a paragraph
    return f"<w:tc>{tcpr}{body}{raw}</w:tc>"


def tr(
    *cells: str,
    header: str | None = None,
    before: int | None = None,
    after: int | None = None,
) -> str:
    """One <w:tr>. `header`: None (absent), "" (bare <w:tblHeader/>), or a w:val."""
    props = ""
    if before is not None:
        props += f'<w:gridBefore w:val="{before}"/>'
    if after is not None:
        props += f'<w:gridAfter w:val="{after}"/>'
    if header == "":
        props += "<w:tblHeader/>"
    elif header is not None:
        props += f'<w:tblHeader w:val="{header}"/>'
    trpr = f"<w:trPr>{props}</w:trPr>" if props else ""
    return f"<w:tr>{trpr}{''.join(cells)}</w:tr>"


def tbl(*rows: str, extra: str = "") -> _Element:
    return parse_xml(f"<w:tbl {W}><w:tblPr/><w:tblGrid/>{''.join(rows)}{extra}</w:tbl>")


def parse(*rows: str) -> TableInfo:
    result = parse_table(tbl(*rows), read_paragraph)
    assert result is not None
    return result


def el(xml: str) -> _Element:
    """The first child of a wrapper, e.g. el(tr(...)) -> the <w:tr>."""
    return parse_xml(f"<w:tbl {W}>{xml}</w:tbl>")[0]


def spans(table: TableInfo) -> list[tuple[tuple[str, ...], int, int, int, int]]:
    """(paragraphs, row, col, row_span, col_span) for every cell, in order."""
    return [(c.paragraphs, c.row, c.col, c.row_span, c.col_span) for c in table.cells]


# ======================================================================
# Readers
# ======================================================================


def test_tblheader_absent() -> None:
    assert read_tblheader(el(tr(tc("a")))) is False


@pytest.mark.parametrize(
    "value", ["", "1", "true", "on"], ids=["bare", "1", "true", "on"]
)
def test_tblheader_on(value: str) -> None:
    assert read_tblheader(el(tr(tc("a"), header=value))) is True


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_tblheader_off(value: str) -> None:
    assert read_tblheader(el(tr(tc("a"), header=value))) is False


def test_tblheader_unknown_value_rejected() -> None:
    with pytest.raises(ValueError, match="Unrecognized w:tblHeader/@w:val 'maybe'"):
        read_tblheader(el(tr(tc("a"), header="maybe")))


@pytest.mark.parametrize(
    ("reader", "kwarg"),
    [(read_gridbefore, "before"), (read_gridafter, "after")],
    ids=["gridBefore", "gridAfter"],
)
def test_grid_before_after(reader: Callable[[_Element], int], kwarg: str) -> None:
    def row_with(value: int) -> _Element:
        opts: dict[str, Any] = {kwarg: value}
        return el(tr(tc("a"), **opts))

    assert reader(el(tr(tc("a")))) == 0
    assert reader(row_with(0)) == 0
    assert reader(row_with(2)) == 2
    with pytest.raises(ValueError, match=">= 0, got -1"):
        reader(row_with(-1))


def test_gridspan() -> None:
    assert read_gridspan(el(tr(tc("a")))[0]) == 1
    assert read_gridspan(el(tr(tc("a", span=3)))[0]) == 3
    with pytest.raises(ValueError, match="gridSpan must be >= 1, got 0"):
        read_gridspan(el(tr(tc("a", span=0)))[0])


def test_gridspan_not_a_number() -> None:
    cell = parse_xml(
        f'<w:tc {W}><w:tcPr><w:gridSpan w:val="two"/></w:tcPr>{p("a")}</w:tc>'
    )
    with pytest.raises(ValueError, match="'two'"):
        read_gridspan(cell)


@pytest.mark.parametrize(
    ("vmerge", "expected"),
    [
        (None, None),
        ("", VMERGE_CONTINUE),  # bare <w:vMerge/> means continue
        ("restart", VMERGE_RESTART),
        ("continue", VMERGE_CONTINUE),
    ],
    ids=["absent", "bare", "restart", "explicit-continue"],
)
def test_vmerge(vmerge: str | None, expected: str | None) -> None:
    assert read_vmerge(el(tr(tc("a", vmerge=vmerge)))[0]) == expected


def test_vmerge_without_tcpr() -> None:
    cell = parse_xml(f"<w:tc {W}>{p('a')}</w:tc>")
    assert read_vmerge(cell) is None


def test_vmerge_unknown_value_rejected() -> None:
    with pytest.raises(ValueError, match="Unrecognized w:vMerge/@w:val 'sideways'"):
        read_vmerge(el(tr(tc("a", vmerge="sideways")))[0])


# ======================================================================
# Simple grids
# ======================================================================


def test_simple_grid() -> None:
    t = parse(tr(tc("TT"), tc("Nội dung")), tr(tc("1"), tc("Phạm vi")))
    assert (t.n_rows, t.n_cols, t.n_header_rows) == (2, 2, 0)
    assert t.text() == "TT\tNội dung\n1\tPhạm vi"


def test_single_cell() -> None:
    t = parse(tr(tc("only")))
    assert t.cells == (
        TableCell(row=0, col=0, row_span=1, col_span=1, paragraphs=("only",)),
    )


def test_empty_cells_are_kept_with_no_paragraphs() -> None:
    t = parse(tr(tc("a"), tc(), tc("c")))
    assert [c.paragraphs for c in t.cells] == [("a",), (), ("c",)]
    assert t.text() == "a\t\tc"


def test_multiple_paragraphs_in_a_cell() -> None:
    t = parse(tr(tc("Kiểm tra?", "Lưu ý: không.")))
    assert t.cells[0].paragraphs == ("Kiểm tra?", "Lưu ý: không.")


def test_invisible_paragraphs_in_a_cell_are_dropped() -> None:
    t = parse(tr(tc("a", "   ", "b")))
    assert t.cells[0].paragraphs == ("a", "b")


def test_callback_sees_every_cell_paragraph_in_order() -> None:
    seen: list[str] = []

    def spy(p_el: _Element) -> str | None:
        text = read_paragraph(p_el)
        seen.append(text or "<empty>")
        return text

    parse_table(tbl(tr(tc("a", "b"), tc()), tr(tc("c"), tc("d"))), spy)
    # Empty paragraphs are still passed to the callback (numbering must see them).
    assert seen == ["a", "b", "<empty>", "c", "d"]


# ======================================================================
# Horizontal merges (gridSpan)
# ======================================================================


def test_gridspan_moves_later_cells_right() -> None:
    t = parse(tr(tc("THAD", span=3)), tr(tc("A"), tc("B"), tc("C")))
    assert spans(t) == [
        (("THAD",), 0, 0, 1, 3),
        (("A",), 1, 0, 1, 1),
        (("B",), 1, 1, 1, 1),
        (("C",), 1, 2, 1, 1),
    ]
    assert t.text() == "THAD\t\t\nA\tB\tC"


def test_gridspan_in_the_middle_of_a_row() -> None:
    t = parse(tr(tc("a"), tc("b", span=2), tc("c")))
    assert [(c.col, c.col_span) for c in t.cells] == [(0, 1), (1, 2), (3, 1)]
    assert t.n_cols == 4


# ======================================================================
# Vertical merges (vMerge)
# ======================================================================


def test_vertical_merge() -> None:
    #   A  B  C
    #   ↑A D  E (restart)
    #   ↑A F  ↑E
    #   G  H  I
    t = parse(
        tr(tc("A", vmerge="restart"), tc("B"), tc("C")),
        tr(tc(vmerge=""), tc("D"), tc("E", vmerge="restart")),
        tr(tc(vmerge=""), tc("F"), tc(vmerge="")),
        tr(tc("G"), tc("H"), tc("I")),
    )
    assert spans(t) == [
        (("A",), 0, 0, 3, 1),
        (("B",), 0, 1, 1, 1),
        (("C",), 0, 2, 1, 1),
        (("D",), 1, 1, 1, 1),
        (("E",), 1, 2, 2, 1),
        (("F",), 2, 1, 1, 1),
        (("G",), 3, 0, 1, 1),
        (("H",), 3, 1, 1, 1),
        (("I",), 3, 2, 1, 1),
    ]


def test_merge_in_both_directions() -> None:
    t = parse(
        tr(tc("B", span=2, vmerge="restart"), tc("c")),
        tr(tc(span=2, vmerge=""), tc("d")),
    )
    assert spans(t) == [
        (("B",), 0, 0, 2, 2),
        (("c",), 0, 2, 1, 1),
        (("d",), 1, 2, 1, 1),
    ]


def test_explicit_continue_value_works_like_bare() -> None:
    t = parse(tr(tc("A", vmerge="restart")), tr(tc(vmerge="continue")))
    assert spans(t) == [(("A",), 0, 0, 2, 1)]


def test_restart_right_after_a_run_starts_a_new_cell() -> None:
    t = parse(
        tr(tc("A", vmerge="restart")),
        tr(tc(vmerge="")),
        tr(tc("B", vmerge="restart")),
        tr(tc(vmerge="")),
    )
    assert spans(t) == [(("A",), 0, 0, 2, 1), (("B",), 2, 0, 2, 1)]


def test_text_in_a_continue_cell_is_dropped() -> None:
    # Word does not render it; only the first cell's content is shown.
    t = parse(tr(tc("A", vmerge="restart")), tr(tc("hidden", vmerge="")))
    assert t.cells[0].paragraphs == ("A",)
    assert "hidden" not in t.text()


def test_merged_text_appears_once_in_text() -> None:
    t = parse(
        tr(tc("1"), tc("Định danh", vmerge="restart"), tc("Số CIF")),
        tr(tc("2"), tc(vmerge=""), tc("Tên KH")),
    )
    assert t.text() == "1\tĐịnh danh\tSố CIF\n2\t\tTên KH"


def test_continue_without_open_cell_rejected() -> None:
    with pytest.raises(ValueError, match="row 0, col 0: .*without open cell"):
        parse(tr(tc(vmerge="")))


def test_plain_cell_ends_a_run() -> None:
    # A plain cell between a restart and a continue breaks the merge.
    with pytest.raises(ValueError, match="row 2, col 0: .*without open cell"):
        parse(
            tr(tc("A", vmerge="restart")),
            tr(tc("B")),
            tr(tc(vmerge="")),
        )


def test_continue_below_a_plain_cell_rejected() -> None:
    with pytest.raises(ValueError, match="row 1, col 0: .*without open cell"):
        parse(tr(tc("A")), tr(tc(vmerge="")))


def test_continue_with_different_span_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=r"row 1, col 0: .*different gridSpan\. Previous: 2, current: 1",
    ):
        parse(
            tr(tc("A", span=2, vmerge="restart")),
            tr(tc(vmerge=""), tc("b")),
        )


# ======================================================================
# Uneven rows: gridBefore, gridAfter, short rows
# ======================================================================


def test_grid_before_gets_empty_filler() -> None:
    t = parse(tr(tc("a"), tc("b")), tr(tc("x"), before=1))
    assert spans(t) == [
        (("a",), 0, 0, 1, 1),
        (("b",), 0, 1, 1, 1),
        ((), 1, 0, 1, 1),
        (("x",), 1, 1, 1, 1),
    ]


def test_grid_after_gets_empty_filler() -> None:
    t = parse(tr(tc("a"), tc("b")), tr(tc("x"), after=1))
    assert spans(t)[-1] == ((), 1, 1, 1, 1)


def test_short_row_gets_empty_filler() -> None:
    t = parse(tr(tc("a"), tc("b"), tc("c")), tr(tc("x")))
    assert [c.paragraphs for c in t.cells if c.row == 1] == [("x",), (), ()]
    assert t.text() == "a\tb\tc\nx\t\t"


def test_width_is_the_widest_row() -> None:
    t = parse(tr(tc("a")), tr(tc("b"), tc("c"), tc("d")), tr(tc("e", span=2)))
    assert t.n_cols == 3


def test_width_counts_grid_before_and_after() -> None:
    t = parse(tr(tc("a"), before=2, after=1))
    assert t.n_cols == 4
    assert t.text() == "\t\ta\t"


def test_filler_cannot_be_continued() -> None:
    # Column 0 of row 1 is a gridBefore gap, so nothing is open there in row 2.
    with pytest.raises(ValueError, match="row 2, col 0: .*without open cell"):
        parse(tr(tc("a"), tc("b")), tr(tc("x"), before=1), tr(tc(vmerge=""), tc("y")))


# ======================================================================
# Header rows
# ======================================================================


def test_header_rows_counted_from_the_top() -> None:
    t = parse(tr(tc("h1"), header=""), tr(tc("h2"), header=""), tr(tc("d")))
    assert t.n_header_rows == 2


def test_late_header_row_not_counted() -> None:
    t = parse(tr(tc("h1"), header=""), tr(tc("d")), tr(tc("x"), header=""))
    assert t.n_header_rows == 1


def test_explicit_off_header_not_counted() -> None:
    t = parse(tr(tc("h1"), header="0"), tr(tc("d")))
    assert t.n_header_rows == 0


def test_two_level_header_from_the_real_table() -> None:
    t = parse(
        tr(
            tc("TT", vmerge="restart"),
            tc("Nhóm", vmerge="restart"),
            tc("THAD", span=2),
            header="",
        ),
        tr(tc(vmerge=""), tc(vmerge=""), tc("Báo cáo"), tc("Kiểm thử"), header=""),
        tr(tc("1"), tc("Định danh", vmerge="restart"), tc(), tc("x")),
        tr(tc("2"), tc(vmerge=""), tc("x"), tc("x")),
        tr(tc("3"), tc("Thẻ"), tc("x"), tc()),
    )
    assert (t.n_rows, t.n_cols, t.n_header_rows) == (5, 4, 2)
    assert t.text().split("\n") == [
        "TT\tNhóm\tTHAD\t",
        "\t\tBáo cáo\tKiểm thử",
        "1\tĐịnh danh\t\tx",
        "2\t\tx\tx",
        "3\tThẻ\tx\t",
    ]


# ======================================================================
# Empty tables
# ======================================================================


def test_table_without_rows_is_none() -> None:
    assert parse_table(tbl(), read_paragraph) is None


def test_rows_without_cells_is_none() -> None:
    assert parse_table(tbl("<w:tr/>", "<w:tr/>"), read_paragraph) is None


def test_table_of_empty_cells_is_valid() -> None:
    # Valid TableInfo; the reader decides to skip it because its text is invisible.
    t = parse(tr(tc(), tc()))
    assert is_invisible(t.text())


# ======================================================================
# Unknown and unsupported content
# ======================================================================


def test_nested_table_not_supported() -> None:
    nested = f"<w:tbl>{tr(tc('inner'))}</w:tbl>"
    with pytest.raises(
        NotImplementedError,
        match=r"/w:tbl/w:tr/w:tc/w:tbl: Nested tables not supported",
    ):
        parse(tr(tc("outer", raw=nested + p())))


def test_unknown_element_in_cell_rejected() -> None:
    sdt = f"<w:sdt><w:sdtContent>{p('x')}</w:sdtContent></w:sdt>"
    with pytest.raises(ValueError, match=r"/w:tbl/w:tr/w:tc/w:sdt: Unknown tag"):
        parse(tr(tc("a", raw=sdt)))


def test_unknown_element_in_row_rejected() -> None:
    with pytest.raises(ValueError, match=r"/w:tbl/w:tr/w:sdt: Unknown tag"):
        parse(tr(tc("a"), "<w:sdt/>"))


def test_unknown_element_in_table_rejected() -> None:
    with pytest.raises(ValueError, match=r"/w:tbl/w:sdt: Unknown tag"):
        parse_table(tbl(tr(tc("a")), extra="<w:sdt/>"), read_paragraph)


def test_row_properties_are_skipped() -> None:
    row = f"<w:tr><w:tblPrEx/><w:trPr/>{tc('a')}</w:tr>"
    assert parse(row).text() == "a"
