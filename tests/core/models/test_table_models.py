"""Validation and rendering of TableCell / TableInfo, and the TABLE block rules.

Most tests build small tables with the `cell` helper below. Positions are
(row, col), 0-based; the top-left corner of a merged cell is its position.
"""

from pathlib import Path

import pytest

from document_processor.core.models.physical import (
    BlockType,
    TableCell,
    TableInfo,
)
from document_processor.core.serialization import (
    load_physical_document,
    save_physical_document,
)

from tests.factories import (
    make_block,
    make_document,
    make_table_block,
    make_table_cell,
    make_table_info,
)


def cell(row: int, col: int, *paragraphs: str, rs: int = 1, cs: int = 1) -> TableCell:
    return TableCell(row=row, col=col, row_span=rs, col_span=cs, paragraphs=paragraphs)


def table(n_rows: int, n_cols: int, *cells: TableCell, header: int = 0) -> TableInfo:
    return TableInfo(n_rows=n_rows, n_cols=n_cols, n_header_rows=header, cells=cells)


# The 2x3 example used throughout:
#
#   A  A  C        A spans 2 columns
#   D  E  C        C spans 2 rows
def example_2x3() -> TableInfo:
    return table(
        2,
        3,
        cell(0, 0, "A", cs=2),
        cell(0, 2, "C", rs=2),
        cell(1, 0, "D"),
        cell(1, 1, "E"),
    )


def data_table() -> TableInfo:
    """A cut-down copy of the real 'Trường hợp áp dụng' table.

    Two header rows; TT / Nhóm / Trường span both; "Trường hợp áp dụng" spans 3 columns;
    "Thông tin định danh khách hàng" spans 3 data rows.
    """
    return table(
        5,
        6,
        cell(0, 0, "TT", rs=2),
        cell(0, 1, "Nhóm dữ liệu", rs=2),
        cell(0, 2, "Trường dữ liệu", rs=2),
        cell(0, 3, "Trường hợp áp dụng", cs=3),
        cell(1, 3, "Trên các báo cáo"),
        cell(1, 4, "Môi trường kiểm thử"),
        cell(1, 5, "Cung cấp dữ liệu"),
        cell(2, 0, "1"),
        cell(2, 1, "Thông tin định danh khách hàng", rs=3),
        cell(2, 2, "Số CIF"),
        cell(2, 3),
        cell(2, 4),
        cell(2, 5, "x"),
        cell(3, 0, "2"),
        cell(3, 2, "Tên khách hàng"),
        cell(3, 3, "x"),
        cell(3, 4, "x"),
        cell(3, 5, "x"),
        cell(4, 0, "3"),
        cell(4, 2, "Ngày sinh"),
        cell(4, 3, "x"),
        cell(4, 4, "x"),
        cell(4, 5, "x"),
        header=2,
    )


# ======================================================================
# TableCell
# ======================================================================


def test_cell_defaults_are_valid() -> None:
    assert make_table_cell().paragraphs == ("Nội dung",)


def test_empty_cell_is_valid() -> None:
    assert make_table_cell(paragraphs=()).paragraphs == ()


@pytest.mark.parametrize("field", ["row", "col"])
def test_cell_rejects_negative_position(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be >= 0, got -1"):
        make_table_cell(**{field: -1})


@pytest.mark.parametrize("field", ["row_span", "col_span"])
@pytest.mark.parametrize("value", [0, -1])
def test_cell_rejects_span_below_one(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=f"{field} must be >= 1, got {value}"):
        make_table_cell(**{field: value})


@pytest.mark.parametrize("bad", ["", " ", "\t", "​"], ids=["empty", "space", "tab", "zwsp"])
def test_cell_rejects_invisible_paragraph(bad: str) -> None:
    # An empty cell has exactly one form: paragraphs=().
    with pytest.raises(ValueError, match=r"TableCell \(2, 3\)\. Paragraph index 1"):
        make_table_cell(row=2, col=3, paragraphs=("ok", bad))


@pytest.mark.parametrize(
    ("paragraphs", "expected"),
    [
        ((), ""),
        (("Số CIF",), "Số CIF"),
        (("Kiểm tra dữ liệu?", "Lưu ý: không tích hợp."), "Kiểm tra dữ liệu? Lưu ý: không tích hợp."),
        (("a\tb",), "a b"),
        (("dòng 1\ndòng 2",), "dòng 1 dòng 2"),
        (("x\r\ny", "z"), "x y z"),
    ],
    ids=["empty", "one", "two-joined-by-space", "tab-cleaned", "newline-cleaned", "crlf-then-join"],
)
def test_cell_text(paragraphs: tuple[str, ...], expected: str) -> None:
    assert make_table_cell(paragraphs=paragraphs).text() == expected


def test_cell_text_keeps_paragraphs_unchanged() -> None:
    # Cleaning happens only in the rendering, never in the stored data.
    c = make_table_cell(paragraphs=("a\tb",))
    c.text()
    assert c.paragraphs == ("a\tb",)


# ======================================================================
# TableInfo: validation
# ======================================================================


def test_table_defaults_are_valid() -> None:
    assert make_table_info().n_rows == 2


def test_single_cell_table() -> None:
    t = table(1, 1, cell(0, 0, "only"))
    assert t.text() == "only"


def test_real_table_is_valid() -> None:
    assert data_table().n_header_rows == 2


@pytest.mark.parametrize("field", ["n_rows", "n_cols"])
@pytest.mark.parametrize("value", [0, -1])
def test_table_rejects_empty_size(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=f"{field} must be > 0, got {value}"):
        make_table_info(**{field: value})


@pytest.mark.parametrize("value", [0, 1, 2])
def test_header_rows_within_range_accepted(value: int) -> None:
    assert make_table_info(n_header_rows=value).n_header_rows == value


@pytest.mark.parametrize("value", [-1, 3])
def test_header_rows_out_of_range_rejected(value: int) -> None:
    with pytest.raises(ValueError, match=f"got {value} with n_rows=2"):
        make_table_info(n_header_rows=value)


def test_cell_past_last_row_rejected() -> None:
    with pytest.raises(ValueError, match=r"TableCell \(1, 0\)\. Cell spanning out of row limit.*got 3 > 2"):
        table(2, 1, cell(0, 0, "a"), cell(1, 0, "b", rs=2))


def test_cell_past_last_col_rejected() -> None:
    with pytest.raises(ValueError, match=r"TableCell \(0, 1\)\. Cell spanning out of col limit.*got 3 > 2"):
        table(1, 2, cell(0, 0, "a"), cell(0, 1, "b", cs=2))


def test_cell_crossing_the_header_boundary_rejected() -> None:
    # C spans rows 0-1, but only row 0 is a header row.
    with pytest.raises(
        ValueError, match=r"TableCell \(0, 2\)\. Cell crosses the header boundary: spans rows \[0, 2\), n_header_rows=1"
    ):
        TableInfo(n_rows=2, n_cols=3, n_header_rows=1, cells=example_2x3().cells)


@pytest.mark.parametrize("header", [0, 2], ids=["no-header", "header-holds-the-merge"])
def test_merge_not_crossing_the_header_boundary_accepted(header: int) -> None:
    assert TableInfo(n_rows=2, n_cols=3, n_header_rows=header, cells=example_2x3().cells).n_header_rows == header


def test_merge_starting_at_the_header_boundary_accepted() -> None:
    #   H  H        row 0: header
    #   A  B        A spans rows 1-2
    #   A  C
    t = table(3, 2, cell(0, 0, "H", cs=2), cell(1, 0, "A", rs=2), cell(1, 1, "B"), cell(2, 1, "C"), header=1)
    assert t.n_header_rows == 1


def test_cells_out_of_order_rejected() -> None:
    with pytest.raises(ValueError, match=r"reading order\. Got \(0, 1\) before \(0, 0\)"):
        table(1, 2, cell(0, 1, "b"), cell(0, 0, "a"))


def test_rows_out_of_order_rejected() -> None:
    with pytest.raises(ValueError, match=r"reading order\. Got \(1, 0\) before \(0, 1\)"):
        table(2, 2, cell(0, 0), cell(1, 0), cell(0, 1), cell(1, 1))


def test_same_position_twice_rejected_as_order() -> None:
    with pytest.raises(ValueError, match=r"reading order\. Got \(0, 0\) before \(0, 0\)"):
        table(1, 1, cell(0, 0, "a"), cell(0, 0, "b"))


def test_next_row_may_start_at_lower_column() -> None:
    # (0, 1) then (1, 0) is reading order even though the column goes down.
    t = table(2, 2, cell(0, 0, rs=2), cell(0, 1), cell(1, 1))
    assert len(t.cells) == 3


def test_horizontal_overlap_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=r"Overlap detected: cell \(0, 1\) overlaps cell \(0, 0\) at position \(0, 1\)",
    ):
        table(1, 2, cell(0, 0, "a", cs=2), cell(0, 1, "b"))


def test_vertical_overlap_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=r"Overlap detected: cell \(1, 0\) overlaps cell \(0, 0\) at position \(1, 0\)",
    ):
        table(2, 1, cell(0, 0, "a", rs=2), cell(1, 0, "b"))


def test_hole_rejected_with_count_and_first_position() -> None:
    with pytest.raises(ValueError, match=r"3 positions not covered\. First missing position: \(0, 1\)"):
        table(2, 2, cell(0, 0, "a"))


def test_hole_inside_the_grid_rejected() -> None:
    # Only the middle of the bottom row is missing.
    with pytest.raises(ValueError, match=r"1 positions not covered\. First missing position: \(1, 1\)"):
        table(2, 3, cell(0, 0), cell(0, 1), cell(0, 2), cell(1, 0), cell(1, 2))


def test_no_cells_at_all_rejected() -> None:
    with pytest.raises(ValueError, match=r"4 positions not covered\. First missing position: \(0, 0\)"):
        table(2, 2)


# ======================================================================
# TableInfo: grid()
# ======================================================================


def test_grid_points_each_position_at_its_owner() -> None:
    t = example_2x3()
    a, c, d, e = t.cells
    assert t.grid() == ((a, a, c), (d, e, c))


def test_grid_is_immutable_tuples() -> None:
    g = example_2x3().grid()
    assert isinstance(g, tuple)
    assert all(isinstance(row, tuple) for row in g)


def test_grid_has_table_shape() -> None:
    g = data_table().grid()
    assert len(g) == 5
    assert all(len(row) == 6 for row in g)


def test_grid_of_real_table() -> None:
    g = data_table().grid()
    group = g[2][1]
    assert group.paragraphs == ("Thông tin định danh khách hàng",)
    # The vertically merged group cell covers column 1 of all three data rows.
    assert g[3][1] is group
    assert g[4][1] is group
    # The horizontally merged header covers columns 3-5 of row 0.
    assert g[0][3] is g[0][4] is g[0][5]


# ======================================================================
# TableInfo: text()
# ======================================================================


def test_text_example_2x3() -> None:
    # A's covered position and C's covered position are empty strings.
    assert example_2x3().text() == "A\t\tC\nD\tE\t"


def test_text_block_merge_in_both_directions() -> None:
    #   A  B  B
    #   C  B  B
    #   D  E  F
    t = table(
        3,
        3,
        cell(0, 0, "A"),
        cell(0, 1, "B", rs=2, cs=2),
        cell(1, 0, "C"),
        cell(2, 0, "D"),
        cell(2, 1, "E"),
        cell(2, 2, "F"),
    )
    assert t.text() == "A\tB\t\nC\t\t\nD\tE\tF"


def test_text_full_width_row() -> None:
    t = table(2, 3, cell(0, 0, "Phụ lục I", cs=3), cell(1, 0, "a"), cell(1, 1, "b"), cell(1, 2, "c"))
    assert t.text() == "Phụ lục I\t\t\na\tb\tc"


def test_text_empty_cells_keep_their_place() -> None:
    t = table(1, 3, cell(0, 0), cell(0, 1, "x"), cell(0, 2))
    assert t.text() == "\tx\t"


def test_text_multi_paragraph_cell() -> None:
    t = table(1, 2, cell(0, 0, "b"), cell(0, 1, "Kiểm tra?", "Lưu ý: không."))
    assert t.text() == "b\tKiểm tra? Lưu ý: không."


def test_text_cell_content_cannot_fake_separators() -> None:
    t = table(1, 2, cell(0, 0, "a\tb"), cell(0, 1, "c\nd"))
    assert t.text() == "a b\tc d"


def test_text_every_row_has_same_number_of_columns() -> None:
    t = data_table()
    rows = t.text().split("\n")
    assert len(rows) == t.n_rows
    assert all(row.count("\t") == t.n_cols - 1 for row in rows)


def test_text_of_real_table_first_rows() -> None:
    lines = data_table().text().split("\n")
    assert lines[0] == "TT\tNhóm dữ liệu\tTrường dữ liệu\tTrường hợp áp dụng\t\t"
    assert lines[1] == "\t\t\tTrên các báo cáo\tMôi trường kiểm thử\tCung cấp dữ liệu"
    assert lines[2] == "1\tThông tin định danh khách hàng\tSố CIF\t\t\tx"
    # The merged group name is not repeated in the rows it covers.
    assert lines[3] == "2\t\tTên khách hàng\tx\tx\tx"


# ======================================================================
# TABLE blocks
# ======================================================================


def test_table_block_is_valid() -> None:
    block = make_table_block()
    assert block.block_type is BlockType.TABLE
    assert block.text == "TT\tNội dung\n1\tPhạm vi điều chỉnh"


def test_table_block_requires_table_info() -> None:
    with pytest.raises(ValueError, match=r"Block pb1: Invalid table block: Doesn't contain table_info"):
        make_block(block_type=BlockType.TABLE, text="TT\tNội dung")


def test_non_table_block_rejects_table_info() -> None:
    with pytest.raises(ValueError, match=r"Block pb1: Non-TABLE block must not contain table_info"):
        make_block(table_info=make_table_info())


def test_table_block_text_must_match_table_info() -> None:
    with pytest.raises(ValueError, match=r"Block pb1: a table block's text must match"):
        make_table_block(text="TT | Nội dung")


def test_table_with_only_empty_cells_cannot_be_a_block() -> None:
    # Its text is only separators, which the visible-text rule rejects.
    empty = table(2, 2, cell(0, 0), cell(0, 1), cell(1, 0), cell(1, 1))
    assert empty.text() == "\t\n\t"
    with pytest.raises(ValueError, match="invisible"):
        make_table_block(table_info=empty)


# ======================================================================
# Serialization
# ======================================================================


def test_table_block_round_trip(tmp_path: Path) -> None:
    doc = make_document(
        blocks=(
            make_block(id="pb1"),
            make_table_block(id="pb2", table_info=data_table()),
        )
    )
    path = tmp_path / "doc.json"
    save_physical_document(doc, path)
    loaded = load_physical_document(path)

    assert loaded == doc


def test_round_trip_restores_tuples(tmp_path: Path) -> None:
    # JSON has no tuples; loading must turn the lists back into tuples.
    doc = make_document(blocks=(make_table_block(id="pb1", table_info=example_2x3()),))
    path = tmp_path / "doc.json"
    save_physical_document(doc, path)
    table_info = load_physical_document(path).blocks[0].table_info

    assert table_info is not None
    assert isinstance(table_info.cells, tuple)
    assert all(isinstance(c.paragraphs, tuple) for c in table_info.cells)


def test_round_trip_keeps_empty_cells(tmp_path: Path) -> None:
    doc = make_document(blocks=(make_table_block(id="pb1", table_info=data_table()),))
    path = tmp_path / "doc.json"
    save_physical_document(doc, path)
    table_info = load_physical_document(path).blocks[0].table_info

    assert table_info is not None
    empty = [c for c in table_info.cells if c.paragraphs == ()]
    assert len(empty) == 2  # (2, 3) and (2, 4)