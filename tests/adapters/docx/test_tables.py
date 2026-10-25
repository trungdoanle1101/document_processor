"""Tests for reading docx tables into TableInfo.

Tables are built from small XML strings. Cell paragraphs are read with a
simple callback (the run text, or None when invisible), so these tests cover
the table logic only; numbering and real paragraph text are covered by the
adapter tests.

Header rows: the leading w:tblHeader rows, or (with "marked_or_bold_first") a
bold first row when no row is marked (whether a run is bold is the reader's
callback, Stylist.resolve_run_is_bold; here it is a Stylist without styles,
so direct w:b only); then moved down past any merged cell
that crosses the boundary; and 0 if no body row would be left.
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
    HeaderRowRule,
    parse_table,
    read_gridafter,
    read_gridbefore,
    read_gridspan,
    read_row_is_bold,
    read_run_text,
    read_tblheader,
    read_vmerge,
)
from document_processor.adapters.docx.styles import Stylist
from document_processor.core.models.physical import TableCell, TableInfo
from document_processor.core.text_utils import is_invisible

W = nsdecls("w")

DIRECT_BOLD = Stylist(None).resolve_run_is_bold


# ---------- builders


def read_paragraph(p: _Element) -> str | None:
    """Stand-in for the reader's callback: run text, None when invisible."""
    text = "".join(t.text or "" for t in p.iter(qn("w:t")))
    return None if is_invisible(text) else text


def r(text: str, bold: str | None = None) -> str:
    """One <w:r>. `bold`: None (absent), "" (bare <w:b/>), or a w:val."""
    if bold is None:
        rpr = ""
    elif bold == "":
        rpr = "<w:rPr><w:b/></w:rPr>"
    else:
        rpr = f'<w:rPr><w:b w:val="{bold}"/></w:rPr>'
    return f"<w:r>{rpr}<w:t xml:space='preserve'>{text}</w:t></w:r>"


def p(text: str = "", bold: str | None = None) -> str:
    return f"<w:p>{r(text, bold)}</w:p>" if text else "<w:p/>"


def tc(
    *paragraphs: str,
    span: int | None = None,
    vmerge: str | None = None,
    bold: str | None = None,
    raw: str = "",
) -> str:
    """One <w:tc>. `vmerge`: None (absent), "" (bare <w:vMerge/>), or a w:val. `bold` applies to every run."""
    props = ""
    if span is not None:
        props += f'<w:gridSpan w:val="{span}"/>'
    if vmerge == "":
        props += "<w:vMerge/>"
    elif vmerge is not None:
        props += f'<w:vMerge w:val="{vmerge}"/>'
    tcpr = f"<w:tcPr>{props}</w:tcPr>" if props else ""
    body = "".join(p(x, bold) for x in paragraphs) or p()  # Word: a cell always has a paragraph
    return f"<w:tc>{tcpr}{body}{raw}</w:tc>"


def tr(*cells: str, header: str | None = None, before: int | None = None, after: int | None = None) -> str:
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


def parse(*rows: str, rule: HeaderRowRule = "marked_or_bold_first") -> TableInfo:
    result = parse_table(tbl(*rows), read_paragraph, DIRECT_BOLD, rule)
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


@pytest.mark.parametrize("value", ["", "1", "true", "on"], ids=["bare", "1", "true", "on"])
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
    cell = parse_xml(f'<w:tc {W}><w:tcPr><w:gridSpan w:val="two"/></w:tcPr>{p("a")}</w:tc>')
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


# ---------- bold


def run_el(xml: str) -> _Element:
    run = el(tr(f"<w:tc><w:p>{xml}</w:p></w:tc>")).find(f"{qn('w:tc')}/{qn('w:p')}/{qn('w:r')}")
    assert run is not None
    return run


def test_run_text_joins_its_text_elements() -> None:
    run = run_el("<w:r><w:t xml:space='preserve'>Nội </w:t><w:t>dung</w:t></w:r>")
    assert read_run_text(run) == "Nội dung"


def row_el(*cells: str) -> _Element:
    return el(tr(*cells))


def row_is_bold(*cells: str) -> bool:
    return read_row_is_bold(row_el(*cells), DIRECT_BOLD)


def test_the_callback_decides_for_each_run() -> None:
    seen: list[str] = []

    def everything_bold(run: _Element) -> bool:
        seen.append(read_run_text(run))
        return True

    # No w:b anywhere, but the callback says bold.
    assert read_row_is_bold(row_el(tc("TT"), tc("Nội dung")), everything_bold) is True
    assert seen == ["TT", "Nội dung"]


def test_blank_runs_are_not_passed_to_the_callback() -> None:
    seen: list[str] = []

    def spy(run: _Element) -> bool:
        seen.append(read_run_text(run))
        return True

    cell = f"<w:tc><w:p>{r('Nội')}{r(' ')}{r('dung')}</w:p></w:tc>"
    read_row_is_bold(row_el(cell, tc()), spy)
    assert seen == ["Nội", "dung"]


def test_row_is_bold_when_every_run_is_bold() -> None:
    assert row_is_bold(tc("TT", bold=""), tc("Nội dung", bold="")) is True


def test_row_with_one_plain_cell_is_not_bold() -> None:
    assert row_is_bold(tc("TT", bold=""), tc("Nội dung")) is False


def test_row_with_one_plain_run_is_not_bold() -> None:
    cell = f"<w:tc><w:p>{r('Nội ', bold='')}{r('dung')}</w:p></w:tc>"
    assert row_is_bold(tc("TT", bold=""), cell) is False


def test_row_with_bold_switched_off_is_not_bold() -> None:
    assert row_is_bold(tc("TT", bold="0")) is False


def test_blank_plain_run_does_not_count() -> None:
    cell = f"<w:tc><w:p>{r('Nội', bold='')}{r(' ')}{r('dung', bold='')}</w:p></w:tc>"
    assert row_is_bold(cell) is True


def test_empty_cell_does_not_count() -> None:
    assert row_is_bold(tc("TT", bold=""), tc()) is True


def test_row_without_text_is_not_bold() -> None:
    assert row_is_bold(tc(), tc()) is False


def test_run_inside_a_hyperlink_counts() -> None:
    plain_link = f"<w:tc><w:p><w:hyperlink>{r('xem')}</w:hyperlink></w:p></w:tc>"
    bold_link = f"<w:tc><w:p><w:hyperlink>{r('xem', bold='')}</w:hyperlink></w:p></w:tc>"
    assert row_is_bold(tc("TT", bold=""), plain_link) is False
    assert row_is_bold(tc("TT", bold=""), bold_link) is True


def test_paragraph_mark_bold_is_not_run_bold() -> None:
    # w:pPr/w:rPr formats the paragraph mark, not the text.
    cell = f"<w:tc><w:p><w:pPr><w:rPr><w:b/></w:rPr></w:pPr>{r('TT')}</w:p></w:tc>"
    assert row_is_bold(cell) is False


# ======================================================================
# Simple grids
# ======================================================================


def test_simple_grid() -> None:
    t = parse(tr(tc("TT"), tc("Nội dung")), tr(tc("1"), tc("Phạm vi")))
    assert (t.n_rows, t.n_cols, t.n_header_rows) == (2, 2, 0)
    assert t.text() == "TT\tNội dung\n1\tPhạm vi"


def test_single_cell() -> None:
    t = parse(tr(tc("only")))
    assert t.cells == (TableCell(row=0, col=0, row_span=1, col_span=1, paragraphs=("only",)),)


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

    parse_table(tbl(tr(tc("a", "b"), tc()), tr(tc("c"), tc("d"))), spy, DIRECT_BOLD, "marked_or_bold_first")
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
    assert spans(t) == [(("B",), 0, 0, 2, 2), (("c",), 0, 2, 1, 1), (("d",), 1, 2, 1, 1)]


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
    with pytest.raises(ValueError, match=r"row 1, col 0: .*different gridSpan\. Previous: 2, current: 1"):
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
        tr(tc("TT", vmerge="restart"), tc("Nhóm", vmerge="restart"), tc("THAD", span=2), header=""),
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


def test_merge_from_the_header_into_the_body_moves_the_marked_boundary() -> None:
    # Only row 0 is marked, but TT is merged down into row 1: the header is both rows.
    t = parse(
        tr(tc("TT", vmerge="restart"), tc("Diện tích", span=2), header=""),
        tr(tc(vmerge=""), tc("Tối thiểu"), tc("Tối đa")),
        tr(tc("1"), tc("50"), tc("100")),
    )
    assert t.n_header_rows == 2


def test_every_row_marked_leaves_no_header() -> None:
    t = parse(tr(tc("h1"), header=""), tr(tc("h2"), header=""))
    assert t.n_header_rows == 0


# ---------- a bold first row


def bold_row(*texts: str) -> str:
    return tr(*(tc(text, bold="") for text in texts))


def plain_row(*texts: str) -> str:
    return tr(*(tc(text) for text in texts))


def test_bold_first_row_is_the_header() -> None:
    t = parse(bold_row("TT", "Nội dung"), plain_row("1", "Phạm vi"))
    assert t.n_header_rows == 1


def test_bold_first_row_ignored_with_the_marked_rule() -> None:
    t = parse(bold_row("TT", "Nội dung"), plain_row("1", "Phạm vi"), rule="marked")
    assert t.n_header_rows == 0


def test_marked_rows_win_over_bold() -> None:
    # Rows 0 and 1 are marked; the bold of row 0 changes nothing.
    t = parse(
        tr(tc("TT", bold=""), header=""),
        tr(tc("Nhóm"), header=""),
        plain_row("1"),
    )
    assert t.n_header_rows == 2


def test_marked_rule_still_reads_marked_rows() -> None:
    t = parse(tr(tc("TT"), header=""), plain_row("1"), rule="marked")
    assert t.n_header_rows == 1


def test_plain_first_row_is_not_a_header() -> None:
    t = parse(plain_row("TT"), bold_row("I. NHÓM"), plain_row("1"))
    assert t.n_header_rows == 0


def test_partly_bold_first_row_is_not_a_header() -> None:
    t = parse(tr(tc("TT", bold=""), tc("Nội dung")), plain_row("1", "Phạm vi"))
    assert t.n_header_rows == 0


def test_first_row_without_text_is_not_a_header() -> None:
    t = parse(plain_row("", ""), plain_row("1", "Phạm vi"))
    assert t.n_header_rows == 0


def test_empty_cell_in_a_bold_first_row_is_fine() -> None:
    t = parse(bold_row("TT", ""), plain_row("1", "Phạm vi"))
    assert t.n_header_rows == 1


def test_only_the_first_bold_row_without_merges() -> None:
    # A bold group row right under the header stays a body row.
    t = parse(
        bold_row("TT", "Loại đất"),
        bold_row("I", "ĐẤT NÔNG NGHIỆP"),
        plain_row("1", "Đất trồng lúa"),
        bold_row("II", "ĐẤT PHI NÔNG NGHIỆP"),
        plain_row("1", "Đất ở"),
    )
    assert t.n_header_rows == 1


def test_all_bold_table_has_no_header() -> None:
    t = parse(bold_row("TT", "Nội dung"), bold_row("1", "Phạm vi"), bold_row("2", "Đối tượng"))
    assert t.n_header_rows == 0


def test_bold_single_row_table_has_no_header() -> None:
    assert parse(bold_row("TT", "Nội dung")).n_header_rows == 0


def test_a_row_without_text_is_not_bold_for_the_all_bold_check() -> None:
    # The textless row is not "bold", so not every row is bold: row 0 is the header.
    t = parse(bold_row("TT"), plain_row(""), bold_row("1"))
    assert t.n_header_rows == 1


# ---------- merges across the boundary


def test_two_level_bold_header() -> None:
    #   TT | Loại đất | Diện tích (m²)          TT, Loại đất merged down; Diện tích across
    #      |          | Tối thiểu | Tối đa
    #   1  | Đất ở    | 50        | 100
    t = parse(
        tr(tc("TT", vmerge="restart", bold=""), tc("Loại đất", vmerge="restart", bold=""), tc("Diện tích (m²)", span=2, bold="")),
        tr(tc(vmerge=""), tc(vmerge=""), tc("Tối thiểu", bold=""), tc("Tối đa", bold="")),
        plain_row("1", "Đất ở", "50", "100"),
    )
    assert t.n_header_rows == 2


def test_merge_decides_even_when_the_second_row_is_plain() -> None:
    t = parse(
        tr(tc("TT", vmerge="restart", bold=""), tc("Nội dung", bold="")),
        tr(tc(vmerge=""), tc("ghi chú")),
        plain_row("1", "Phạm vi"),
    )
    assert t.n_header_rows == 2


def test_boundary_moves_again_for_a_chained_merge() -> None:
    # A spans rows 0-1; B starts in row 1 and spans rows 1-2: the header is rows 0-2.
    t = parse(
        tr(tc("A", vmerge="restart", bold=""), tc("x", bold="")),
        tr(tc(vmerge=""), tc("B", vmerge="restart")),
        tr(tc("y"), tc(vmerge="")),
        plain_row("1", "2"),
    )
    assert t.n_header_rows == 3


def test_merge_down_to_the_last_row_leaves_no_header() -> None:
    t = parse(
        tr(tc("Nhóm", vmerge="restart", bold=""), tc("TT", bold="")),
        tr(tc(vmerge=""), tc("1")),
        tr(tc(vmerge=""), tc("2")),
    )
    assert t.n_header_rows == 0


def test_merge_inside_the_body_does_not_move_the_boundary() -> None:
    t = parse(
        bold_row("TT", "Nhóm"),
        tr(tc("1"), tc("Định danh", vmerge="restart")),
        tr(tc("2"), tc(vmerge="")),
    )
    assert t.n_header_rows == 1


def test_merge_ending_at_the_boundary_does_not_move_it() -> None:
    t = parse(
        tr(tc("TT", vmerge="restart"), tc("Nhóm"), header=""),
        tr(tc(vmerge=""), tc("Báo cáo"), header=""),
        plain_row("1", "x"),
    )
    assert t.n_header_rows == 2


# ======================================================================
# Empty tables
# ======================================================================


def test_table_without_rows_is_none() -> None:
    assert parse_table(tbl(), read_paragraph, DIRECT_BOLD, "marked_or_bold_first") is None


def test_rows_without_cells_is_none() -> None:
    assert parse_table(tbl("<w:tr/>", "<w:tr/>"), read_paragraph, DIRECT_BOLD, "marked_or_bold_first") is None


def test_table_of_empty_cells_is_valid() -> None:
    # Valid TableInfo; the reader decides to skip it because its text is invisible.
    t = parse(tr(tc(), tc()))
    assert is_invisible(t.text())


# ======================================================================
# Unknown and unsupported content
# ======================================================================


def test_nested_table_not_supported() -> None:
    nested = f"<w:tbl>{tr(tc('inner'))}</w:tbl>"
    with pytest.raises(NotImplementedError, match=r"/w:tbl/w:tr/w:tc/w:tbl: Nested tables not supported"):
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
        parse_table(tbl(tr(tc("a")), extra="<w:sdt/>"), read_paragraph, DIRECT_BOLD, "marked_or_bold_first")


def test_row_properties_are_skipped() -> None:
    row = f"<w:tr><w:tblPrEx/><w:trPr/>{tc('a')}</w:tr>"
    assert parse(row).text() == "a"