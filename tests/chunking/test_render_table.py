"""Tests for render_table_rows: a table, or some of its body rows, as text.

Both layouts print a merged cell's text once per rendered range: on the first
row of the range it covers (row == max(cell.row, start)), at its own column.

- grid: the header rows, then one line per body row, cells joined by " | ";
  "" wherever nothing is printed, so values stay under their headers.
- key_value: one line per body row, "label: value; ...", labels from the header
  text above each column (several header rows joined with " / "); empty
  values and empty rows left out; no header rows falls back to grid.
"""

import pytest

from document_processor.chunking.render import TableLayout, render_table_rows
from document_processor.core.models import TableCell, TableInfo


def cell(
    row: int, col: int, *paragraphs: str, rows: int = 1, cols: int = 1
) -> TableCell:
    return TableCell(
        row=row, col=col, row_span=rows, col_span=cols, paragraphs=paragraphs
    )


# One header row; "Nhóm A" spans rows 1-2; row 3 has an empty cell; row 4 is one cell across the table.
#
#     STT | Nhóm   | Nội dung
#     1   | Nhóm A | Kiến trúc ứng dụng
#     2   |   ↑    | Kiến trúc dữ liệu
#     3   | Nhóm B |
#     I. TỔNG KẾT ------------------
GROUPS = TableInfo(
    n_rows=5,
    n_cols=3,
    n_header_rows=1,
    cells=(
        cell(0, 0, "STT"),
        cell(0, 1, "Nhóm"),
        cell(0, 2, "Nội dung"),
        cell(1, 0, "1"),
        cell(1, 1, "Nhóm A", rows=2),
        cell(1, 2, "Kiến trúc ứng dụng"),
        cell(2, 0, "2"),
        cell(2, 2, "Kiến trúc dữ liệu"),
        cell(3, 0, "3"),
        cell(3, 1, "Nhóm B"),
        cell(3, 2),
        cell(4, 0, "I. TỔNG KẾT", cols=3),
    ),
)

# Two header rows: "STT" spans both, "Thông tin" spans two columns above "Ngày" and "Số VBCĐ".
#
#     STT | Thông tin ------------
#      ↑  | Ngày       | Số VBCĐ
#     1   | 22/07/2022 | 4188/QyĐ-BIDV
TWO_HEADER_ROWS = TableInfo(
    n_rows=3,
    n_cols=3,
    n_header_rows=2,
    cells=(
        cell(0, 0, "STT", rows=2),
        cell(0, 1, "Thông tin", cols=2),
        cell(1, 1, "Ngày"),
        cell(1, 2, "Số VBCĐ"),
        cell(2, 0, "1"),
        cell(2, 1, "22/07/2022"),
        cell(2, 2, "4188/QyĐ-BIDV"),
    ),
)

NO_HEADER = TableInfo(
    n_rows=2,
    n_cols=2,
    n_header_rows=0,
    cells=(
        cell(0, 0, "Họ và tên:"),
        cell(0, 1, "Hoàng Thị Phương"),
        cell(1, 0, "Chức danh:"),
        cell(1, 1, "Chuyên viên"),
    ),
)


# ============================================================ range checks


@pytest.mark.parametrize(("start", "end"), [(0, 5), (2, 1), (1, 6), (-1, 3)])
def test_rows_outside_the_body_rejected(start: int, end: int) -> None:
    with pytest.raises(ValueError, match=r"1 <= start <= end <= 5"):
        render_table_rows(GROUPS, start, end, "grid")


@pytest.mark.parametrize("layout", ["grid", "key_value"])
def test_a_range_without_body_rows(layout: TableLayout) -> None:
    expected = "STT | Nhóm | Nội dung" if layout == "grid" else ""
    assert render_table_rows(GROUPS, 3, 3, layout) == expected


# ============================================================ grid


def test_grid_whole_table() -> None:
    assert (
        render_table_rows(GROUPS, 1, 5, "grid")
        == "\n".join(
            [
                "STT | Nhóm | Nội dung",
                "1 | Nhóm A | Kiến trúc ứng dụng",
                "2 |  | Kiến trúc dữ liệu",  # the merged "Nhóm A" is printed once, on its first row
                "3 | Nhóm B | ",  # an empty cell keeps its column
                "I. TỔNG KẾT |  | ",  # a cell across the table: text at its first column only
            ]
        )
    )


def test_grid_repeats_the_header_for_part_of_the_body() -> None:
    assert (
        render_table_rows(GROUPS, 3, 5, "grid").splitlines()[0]
        == "STT | Nhóm | Nội dung"
    )


def test_grid_range_starting_inside_a_merge_prints_it_on_its_first_row() -> None:
    assert (
        render_table_rows(GROUPS, 2, 3, "grid")
        == "STT | Nhóm | Nội dung\n2 | Nhóm A | Kiến trúc dữ liệu"
    )


def test_grid_range_ending_inside_a_merge() -> None:
    assert (
        render_table_rows(GROUPS, 1, 2, "grid")
        == "STT | Nhóm | Nội dung\n1 | Nhóm A | Kiến trúc ứng dụng"
    )


def test_grid_with_two_header_rows() -> None:
    assert render_table_rows(TWO_HEADER_ROWS, 2, 3, "grid") == "\n".join(
        [
            "STT | Thông tin | ",
            " | Ngày | Số VBCĐ",  # "STT" spans both header rows: printed once
            "1 | 22/07/2022 | 4188/QyĐ-BIDV",
        ]
    )


def test_grid_without_header_rows() -> None:
    assert (
        render_table_rows(NO_HEADER, 0, 2, "grid")
        == "Họ và tên: | Hoàng Thị Phương\nChức danh: | Chuyên viên"
    )


def test_grid_multi_paragraph_cell_stays_on_one_line() -> None:
    table = TableInfo(
        n_rows=2,
        n_cols=1,
        n_header_rows=1,
        cells=(cell(0, 0, "Mô tả"), cell(1, 0, "Đoạn một.", "Đoạn hai.")),
    )
    assert render_table_rows(table, 1, 2, "grid") == "Mô tả\nĐoạn một. Đoạn hai."


# ============================================================ key_value


def test_key_value_whole_table() -> None:
    assert (
        render_table_rows(GROUPS, 1, 5, "key_value")
        == "\n".join(
            [
                "STT: 1; Nhóm: Nhóm A; Nội dung: Kiến trúc ứng dụng",
                "STT: 2; Nội dung: Kiến trúc dữ liệu",  # the merged "Nhóm A" is printed once per range
                "STT: 3; Nhóm: Nhóm B",  # the empty cell is left out
                "STT: I. TỔNG KẾT",  # a cell across columns, with its first column's label
            ]
        )
    )


def test_key_value_range_starting_inside_a_merge_prints_it_on_its_first_row() -> None:
    assert (
        render_table_rows(GROUPS, 2, 3, "key_value")
        == "STT: 2; Nhóm: Nhóm A; Nội dung: Kiến trúc dữ liệu"
    )


def test_key_value_has_no_header_line() -> None:
    assert "STT | " not in render_table_rows(GROUPS, 1, 5, "key_value")


def test_key_value_labels_join_several_header_rows() -> None:
    assert render_table_rows(TWO_HEADER_ROWS, 2, 3, "key_value") == (
        "STT: 1; Thông tin / Ngày: 22/07/2022; Thông tin / Số VBCĐ: 4188/QyĐ-BIDV"
    )


def test_key_value_without_header_rows_falls_back_to_grid() -> None:
    assert render_table_rows(NO_HEADER, 0, 2, "key_value") == render_table_rows(
        NO_HEADER, 0, 2, "grid"
    )


def test_key_value_column_without_a_header_text_prints_the_value_alone() -> None:
    table = TableInfo(
        n_rows=2,
        n_cols=2,
        n_header_rows=1,
        cells=(
            cell(0, 0, "Tên"),
            cell(0, 1),
            cell(1, 0, "Ban CN"),
            cell(1, 1, "Đầu mối"),
        ),
    )
    assert render_table_rows(table, 1, 2, "key_value") == "Tên: Ban CN; Đầu mối"


def test_key_value_row_with_only_empty_cells_is_left_out() -> None:
    table = TableInfo(
        n_rows=4,
        n_cols=2,
        n_header_rows=1,
        cells=(
            cell(0, 0, "A"),
            cell(0, 1, "B"),
            cell(1, 0, "1"),
            cell(1, 1, "x"),
            cell(2, 0),
            cell(2, 1),
            cell(3, 0, "3"),
            cell(3, 1, "z"),
        ),
    )
    assert render_table_rows(table, 1, 4, "key_value") == "A: 1; B: x\nA: 3; B: z"


def test_a_long_merged_cell_appears_once_per_range_in_both_layouts() -> None:
    long_text = " ".join(["đoạn"] * 50)
    table = TableInfo(
        n_rows=5,
        n_cols=2,
        n_header_rows=1,
        cells=(
            cell(0, 0, "Mô tả"),
            cell(0, 1, "Ngày"),
            cell(1, 0, long_text, rows=4),
            cell(1, 1, "01/2024"),
            cell(2, 1, "02/2024"),
            cell(3, 1, "03/2024"),
            cell(4, 1, "04/2024"),
        ),
    )
    layouts: tuple[TableLayout, ...] = ("grid", "key_value")
    for layout in layouts:
        assert render_table_rows(table, 1, 5, layout).count(long_text) == 1
        assert render_table_rows(table, 3, 5, layout).count(long_text) == 1
