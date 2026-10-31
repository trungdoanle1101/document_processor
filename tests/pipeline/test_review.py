"""Tests for pipeline/review.py: format_outline, build_review_data, build_table_rows.

format_outline: one line per node: indentation by depth, the heading label
(marker and title; "[level]" when both are missing) and the segment range.
Front and end matter get a line each, "none" when empty, in both modes.
sections_only keeps only sections (Chương, Mục, Điều), wherever they sit: Điều
may be a root, under a Chương, or under an optional Mục.

build_review_data(result) is what the review page draws: the summary, the
document, node depths, the outline (one ReviewNode per node, with the chunks
whose range overlaps the node's), and chunks_or_gaps: the result's own chunks
in order with a Gap for every run of segments no chunk covers, together
covering every segment. A gap stays inside one part; it is in_breadcrumb when
every segment in it is a heading whose node a later chunk's breadcrumb shows,
otherwise it was dropped (images, headings no breadcrumb shows).

build_table_rows(table, start, end): the header rows, then body rows
[start, end), placed as the grid layout places them: a merged body cell on the
first row it covers in the range, its row span cut at end; cells in column
order; copies, the table itself unchanged.
"""

from pathlib import Path
from typing import Any

import pytest

from document_processor.chunking import Chunk, OutlineChunker, WordTokenizer
from document_processor.chunking.render import render_table_rows
from document_processor.core.models import (
    LogicalDocument,
    LogicalNode,
    Segment,
    TableCell,
    TableInfo,
)
from document_processor.logical import VN_REGULATION
from document_processor.pipeline import PipelineResult, format_outline
from document_processor.pipeline.review import (
    Gap,
    ReviewData,
    build_review_data,
    build_table_rows,
)
from tests.chunking.test_outline_chunker import Blocks, regulation
from tests.factories import (
    heading_for,
    make_document,
    make_logical_document,
    make_logical_node,
    make_table_block,
    make_text_segment,
)

REGULATION_OUTLINE = """front matter  [0, 1)
Chương I QUY ĐỊNH CHUNG  [1, 15)
  Điều 1. Phạm vi điều chỉnh  [2, 7)
    1.  [5, 7)
  Điều 2. Giải thích từ ngữ  [7, 15)
    1.  [8, 10)
    2.  [10, 15)
      a)  [12, 15)
end matter  [15, 17)"""


def outlined(
    *nodes: LogicalNode, n_segments: int, body: tuple[int, int]
) -> LogicalDocument:
    segments: list[Segment] = []
    for index in range(n_segments):
        starting = [n for n in nodes if n.start == index]
        segments.append(
            heading_for(starting[0])
            if starting
            else make_text_segment(block_id=f"pb{index}", text="x")
        )
    return make_logical_document(
        segments=tuple(segments), nodes=nodes, body_start=body[0], body_end=body[1]
    )


def node(
    node_id: str, parent: str | None, start: int, end: int, **overrides: Any
) -> LogicalNode:
    fields: dict[str, Any] = {
        "id": node_id,
        "parent_id": parent,
        "level": "dieu",
        "number": None,
        "marker": f"Điều {node_id}.",
        "title": None,
        "head_block_ids": (f"h{node_id}",),
        "start": start,
        "end": end,
    }
    fields.update(overrides)
    return make_logical_node(**fields)


# ============================================================ the whole outline


def test_full_outline_of_the_regulation() -> None:
    assert format_outline(regulation()) == REGULATION_OUTLINE


def test_returns_text_without_printing(capsys: pytest.CaptureFixture[str]) -> None:
    text = format_outline(regulation())
    assert capsys.readouterr().out == ""
    assert isinstance(text, str) and not text.endswith("\n")


# ============================================================ depth


def test_sections_only() -> None:
    assert (
        format_outline(regulation(), sections_only=True)
        == """front matter  [0, 1)
Chương I QUY ĐỊNH CHUNG  [1, 15)
  Điều 1. Phạm vi điều chỉnh  [2, 7)
  Điều 2. Giải thích từ ngữ  [7, 15)
end matter  [15, 17)"""
    )


def test_sections_only_keeps_front_and_end_matter() -> None:
    lines = format_outline(regulation(), sections_only=True).splitlines()
    assert (lines[0], lines[-1]) == ("front matter  [0, 1)", "end matter  [15, 17)")


def test_sections_only_finds_dieu_at_any_depth() -> None:
    # Chương with a Mục, Chương without, and a Điều with no Chương above it: every Điều is shown.
    document = outlined(
        node("I", None, 0, 6, level="chuong", marker="Chương I"),
        node("m1", "I", 1, 4, level="muc", marker="Mục 1"),
        node("1", "m1", 2, 4),
        node("1-1", "1", 3, 4, level="khoan", is_section=False, marker="1."),
        node("2", "I", 4, 6),
        node("2-1", "2", 5, 6, level="khoan", is_section=False, marker="1."),
        node("3", None, 6, 8),
        node("3-1", "3", 7, 8, level="khoan", is_section=False, marker="1."),
        n_segments=8,
        body=(0, 8),
    )
    assert format_outline(document, sections_only=True).splitlines()[1:-1] == [
        "Chương I  [0, 6)",
        "  Mục 1  [1, 4)",
        "    Điều 1.  [2, 4)",
        "  Điều 2.  [4, 6)",
        "Điều 3.  [6, 8)",
    ]


def test_sections_only_on_a_document_of_items_only() -> None:
    document = outlined(
        node("1", None, 0, 2, level="gach", is_section=False, marker="-"),
        n_segments=2,
        body=(0, 2),
    )
    assert (
        format_outline(document, sections_only=True)
        == "front matter  none\nend matter  none"
    )


@pytest.mark.parametrize("indent", [1, 4])
def test_indent_width(indent: int) -> None:
    lines = format_outline(regulation(), indent=indent).splitlines()
    assert lines[2] == " " * indent + "Điều 1. Phạm vi điều chỉnh  [2, 7)"
    assert lines[7] == " " * (3 * indent) + "a)  [12, 15)"


def test_siblings_after_a_deep_branch_return_to_their_depth() -> None:
    document = outlined(
        node("1", None, 0, 4),
        node("1a", "1", 1, 3),
        node("1a-i", "1a", 2, 3),
        node("2", None, 4, 5),
        n_segments=5,
        body=(0, 5),
    )
    assert format_outline(document).splitlines()[1:-1] == [
        "Điều 1.  [0, 4)",
        "  Điều 1a.  [1, 3)",
        "    Điều 1a-i.  [2, 3)",
        "Điều 2.  [4, 5)",
    ]


# ============================================================ labels


def test_title_without_a_marker() -> None:
    document = outlined(
        node("1", None, 0, 2, marker=None, title="Lời nói đầu"),
        n_segments=2,
        body=(0, 2),
    )
    assert format_outline(document).splitlines()[1] == "Lời nói đầu  [0, 2)"


def test_neither_marker_nor_title_shows_the_level() -> None:
    document = outlined(
        node("1", None, 0, 2, level="gach", marker=None, title=None),
        n_segments=2,
        body=(0, 2),
    )
    assert format_outline(document).splitlines()[1] == "[gach]  [0, 2)"


# ============================================================ front and end matter


def test_no_front_matter() -> None:
    document = outlined(node("1", None, 0, 2), n_segments=3, body=(0, 2))
    assert format_outline(document).splitlines()[0] == "front matter  none"


def test_no_end_matter() -> None:
    document = outlined(node("1", None, 1, 3), n_segments=3, body=(1, 3))
    assert format_outline(document).splitlines()[-1] == "end matter  none"


def test_document_without_nodes() -> None:
    document = make_logical_document(
        segments=(make_text_segment(block_id="pb1"), make_text_segment(block_id="pb2")),
        body_start=0,
        body_end=2,
    )
    assert format_outline(document) == "front matter  none\nend matter  none"


def test_empty_document() -> None:
    assert (
        format_outline(make_logical_document())
        == "front matter  none\nend matter  none"
    )


# ======================================================================
# build_review_data
# ======================================================================


def review(
    document: LogicalDocument, max_tokens: int = 200
) -> tuple[ReviewData, tuple[Chunk, ...]]:
    chunks = OutlineChunker(tokenizer=WordTokenizer(), max_tokens=max_tokens).chunk(
        document
    )
    result = PipelineResult(
        source=Path("quy_dinh.docx"),
        physical=make_document(),
        logical=document,
        chunks=chunks,
        max_tokens=max_tokens,
        level_order=tuple(level.name for level in VN_REGULATION.levels or ()),
    )
    return build_review_data(result), chunks


def gaps(data: ReviewData) -> list[Gap]:
    return [entry for entry in data.chunks_or_gaps if isinstance(entry, Gap)]


def ranges(data: ReviewData) -> list[tuple[str, int, int]]:
    return [(entry.id, entry.start, entry.end) for entry in data.chunks_or_gaps]


def covered(data: ReviewData) -> list[int]:
    return sorted(
        {
            index
            for entry in data.chunks_or_gaps
            for index in range(entry.start, entry.end)
        }
    )


# ---------- what it holds


def test_summary_and_document_come_from_the_result() -> None:
    document = regulation()
    data, chunks = review(document)
    assert data.document is document
    assert data.summary.chunk_count == len(chunks)
    assert data.summary.max_tokens == 200


def test_depths_follow_the_parent_chain() -> None:
    data, _ = review(regulation())
    assert data.depths == {
        "n0": 0,
        "n1": 1,
        "n2": 2,
        "n3": 1,
        "n4": 2,
        "n5": 2,
        "n6": 3,
    }


# ---------- chunks_or_gaps


@pytest.mark.parametrize("max_tokens", [3, 8, 200])
def test_chunks_and_gaps_cover_every_segment(max_tokens: int) -> None:
    document = regulation()
    data, _ = review(document, max_tokens)
    assert covered(data) == list(range(len(document.segments)))


@pytest.mark.parametrize("max_tokens", [3, 8, 200])
def test_chunks_are_the_results_own_in_order(max_tokens: int) -> None:
    data, chunks = review(regulation(), max_tokens)
    in_data = [entry for entry in data.chunks_or_gaps if isinstance(entry, Chunk)]
    assert len(in_data) == len(chunks)
    assert all(a is b for a, b in zip(in_data, chunks))


@pytest.mark.parametrize("max_tokens", [3, 8, 200])
def test_entries_are_in_document_order_without_overlap(max_tokens: int) -> None:
    data, _ = review(regulation(), max_tokens)
    entries = data.chunks_or_gaps
    for before, after in zip(entries, entries[1:]):
        same_table = (
            isinstance(before, Chunk)
            and isinstance(after, Chunk)
            and before.table_rows is not None
        )
        assert after.start == before.end or (same_table and after.start == before.start)


def test_chapter_heading_is_shown_in_the_breadcrumb() -> None:
    # Chương I holds sections, so its heading is a piece of its own and dropped;
    # the chunks below carry it in their breadcrumb.
    data, chunks = review(regulation())
    (gap,) = gaps(data)
    assert (gap.id, gap.part, gap.start, gap.end, gap.in_breadcrumb) == (
        "gap-1",
        "body",
        1,
        2,
        True,
    )
    assert "Chương I QUY ĐỊNH CHUNG" in chunks[1].headings


def test_gap_ids_are_unique() -> None:
    data, _ = review(regulation(), 3)
    ids = [gap.id for gap in gaps(data)]
    assert len(ids) == len(set(ids)) and all(
        id_ == f"gap-{gap.start}" for id_, gap in zip(ids, gaps(data))
    )


def test_dropped_image_is_not_in_breadcrumb() -> None:
    #   0 (image)                 front matter
    #   1 # Điều 1. Phạm vi       [1, 3)
    #   2 Nội dung
    document = Blocks().image().text("Điều 1. Phạm vi").text("Nội dung").build()
    data, _ = review(document)
    assert [
        (gap.part, gap.start, gap.end, gap.in_breadcrumb) for gap in gaps(data)
    ] == [("front_matter", 0, 1, False)]


def test_heading_only_article_is_dropped() -> None:
    # A typed table of contents entry, or an article with only a title: no chunk, no breadcrumb.
    document = (
        Blocks()
        .text("Điều 1. Phạm vi")
        .text("Điều 2. Đối tượng")
        .text("Nội dung")
        .build()
    )
    data, _ = review(document)
    (gap,) = gaps(data)
    assert (gap.start, gap.end, gap.in_breadcrumb) == (0, 1, False)


def test_a_run_splits_where_breadcrumb_headings_meet_dropped_segments() -> None:
    #   0 (image)                 front matter
    #   1 # Chương I              chuong [1, 5)   heading in the breadcrumb of c?
    #   2 (image)                                 dropped
    #   3 # Điều 1. Phạm vi       dieu   [3, 5)
    #   4 Nội dung
    document = (
        Blocks()
        .image()
        .text("Chương I")
        .image()
        .text("Điều 1. Phạm vi")
        .text("Nội dung")
        .build()
    )
    data, _ = review(document)
    assert [
        (gap.id, gap.part, gap.start, gap.end, gap.in_breadcrumb) for gap in gaps(data)
    ] == [
        ("gap-0", "front_matter", 0, 1, False),
        ("gap-1", "body", 1, 2, True),
        ("gap-2", "body", 2, 3, False),
    ]
    assert covered(data) == list(range(len(document.segments)))


def test_split_table_pieces_follow_each_other_without_a_gap() -> None:
    blocks = Blocks().text("Điều 1. Phạm vi").text("Bảng:")
    blocks.blocks.append(make_table_block(id="pb3", table_info=MERGED))
    data, _ = review(blocks.build(), 6)
    pieces = [
        entry
        for entry in data.chunks_or_gaps
        if isinstance(entry, Chunk) and entry.table_rows is not None
    ]
    assert len(pieces) > 1 and len({piece.start for piece in pieces}) == 1
    first = data.chunks_or_gaps.index(pieces[0])
    assert list(data.chunks_or_gaps[first : first + len(pieces)]) == pieces
    assert covered(data) == list(range(len(data.document.segments)))


def test_document_without_chunks_is_all_gaps() -> None:
    document = Blocks().image().image().build()
    data, chunks = review(document)
    assert chunks == ()
    assert ranges(data) == [("gap-0", 0, 2)]
    assert gaps(data)[0].in_breadcrumb is False


def test_empty_document_has_no_chunks_gaps_or_outline() -> None:
    data, _ = review(make_logical_document())
    assert data.chunks_or_gaps == () and data.outline == ()


# ---------- outline


def test_one_review_node_per_node_in_pre_order() -> None:
    document = regulation()
    data, _ = review(document)
    assert [n.node_id for n in data.outline] == [node.id for node in document.nodes]
    assert [(n.label, n.level, n.depth, n.is_section) for n in data.outline] == [
        ("Chương I QUY ĐỊNH CHUNG", "chuong", 0, True),
        ("Điều 1. Phạm vi điều chỉnh", "dieu", 1, True),
        ("1.", "khoan", 2, False),
        ("Điều 2. Giải thích từ ngữ", "dieu", 1, True),
        ("1.", "khoan", 2, False),
        ("2.", "khoan", 2, False),
        ("a)", "diem", 3, False),
    ]


def test_outline_depths_match_data_depths() -> None:
    data, _ = review(regulation())
    assert all(data.depths[n.node_id] == n.depth for n in data.outline)


def test_item_grouped_into_its_parents_chunk_lists_that_chunk() -> None:
    # At 200 words each Điều is one chunk owned by the Điều; its khoản own nothing.
    data, chunks = review(regulation())
    by_id = {n.node_id: n for n in data.outline}
    dieu_1_chunk = next(c for c in chunks if c.path and c.path[-1].node_id == "n1")
    assert by_id["n2"].overlap_chunk_ids == (dieu_1_chunk.id,)
    assert by_id["n1"].overlap_chunk_ids == (dieu_1_chunk.id,)


def test_a_split_node_lists_every_chunk_it_spans() -> None:
    document = regulation()
    data, chunks = review(document, 3)
    dieu_2 = document.get_node("n3")
    expected = tuple(
        c.id for c in chunks if c.start < dieu_2.end and dieu_2.start < c.end
    )
    by_id = {n.node_id: n for n in data.outline}
    assert len(expected) > 1
    assert by_id["n3"].overlap_chunk_ids == expected


def test_a_chunk_ending_where_a_node_starts_does_not_overlap_it() -> None:
    # Ranges are half-open: a chunk that ends at khoản 2's first segment holds none of it.
    document = regulation()
    data, chunks = review(document, 3)
    khoan_2 = document.get_node("n5")
    (touching,) = [c for c in chunks if c.end == khoan_2.start]
    by_id = {n.node_id: n for n in data.outline}
    assert touching.id not in by_id["n5"].overlap_chunk_ids


def test_every_node_of_the_regulation_lists_a_chunk() -> None:
    data, _ = review(regulation())
    assert all(n.overlap_chunk_ids for n in data.outline)


def test_label_falls_back_to_the_level() -> None:
    bare = node(
        "1", None, 0, 2, marker=None, title=None, level="gach", is_section=False
    )
    data, _ = review(outlined(bare, n_segments=2, body=(0, 2)))
    assert data.outline[0].label == "[gach]"


# ======================================================================
# build_table_rows
# ======================================================================


def cell(
    row: int, col: int, text: str = "", rs: int = 1, cs: int = 1, *more: str
) -> TableCell:
    paragraphs = (text, *more) if text else ()
    return TableCell(row=row, col=col, row_span=rs, col_span=cs, paragraphs=paragraphs)


# The running table: one header row; "A" merged down body rows 1-3 in column 0.
#
#   0  TT | Nội dung          header
#   1  A  | x1
#   2  ↑  | x2
#   3  ↑  | x3
#   4  B  | x4
MERGED = TableInfo(
    n_rows=5,
    n_cols=2,
    n_header_rows=1,
    cells=(
        cell(0, 0, "TT"),
        cell(0, 1, "Nội dung"),
        cell(1, 0, "A", 3),
        cell(1, 1, "x1"),
        cell(2, 1, "x2"),
        cell(3, 1, "x3"),
        cell(4, 0, "B"),
        cell(4, 1, "x4"),
    ),
)


def shape(
    rows: tuple[tuple[TableCell, ...], ...],
) -> list[list[tuple[int, int, str, int]]]:
    """(row, col, text, row_span) per cell, row by row."""
    return [[(c.row, c.col, c.text(), c.row_span) for c in row] for row in rows]


def test_whole_table() -> None:
    assert shape(build_table_rows(MERGED, 1, 5)) == [
        [(0, 0, "TT", 1), (0, 1, "Nội dung", 1)],
        [(1, 0, "A", 3), (1, 1, "x1", 1)],
        [(2, 1, "x2", 1)],
        [(3, 1, "x3", 1)],
        [(4, 0, "B", 1), (4, 1, "x4", 1)],
    ]


@pytest.mark.parametrize(
    ("start", "end", "first_body_row"),
    [
        (1, 2, [(1, 0, "A", 1), (1, 1, "x1", 1)]),  # cut at the end of the piece
        (2, 4, [(2, 0, "A", 2), (2, 1, "x2", 1)]),  # carried down, cut at both sides
        (
            3,
            5,
            [(3, 0, "A", 1), (3, 1, "x3", 1)],
        ),  # carried down to the last row it covers
    ],
    ids=["cut-at-end", "carried-and-cut", "carried-to-its-last-row"],
)
def test_merged_cell_in_a_piece(
    start: int, end: int, first_body_row: list[tuple[int, int, str, int]]
) -> None:
    rows = shape(build_table_rows(MERGED, start, end))
    assert rows[0] == [
        (0, 0, "TT", 1),
        (0, 1, "Nội dung", 1),
    ]  # header first, every piece
    assert rows[1] == first_body_row
    assert len(rows) == 1 + (end - start)


def test_piece_after_a_merge_does_not_repeat_it() -> None:
    assert shape(build_table_rows(MERGED, 4, 5)) == [
        [(0, 0, "TT", 1), (0, 1, "Nội dung", 1)],
        [(4, 0, "B", 1), (4, 1, "x4", 1)],
    ]


def test_carried_cell_lands_in_its_own_column() -> None:
    # The merged cell is in column 1, so it is met before row 2's column-0 cell.
    table = TableInfo(
        n_rows=4,
        n_cols=2,
        n_header_rows=1,
        cells=(
            cell(0, 0, "TT"),
            cell(0, 1, "Nhóm"),
            cell(1, 0, "1"),
            cell(1, 1, "G", 3),
            cell(2, 0, "2"),
            cell(3, 0, "3"),
        ),
    )
    assert shape(build_table_rows(table, 2, 4))[1] == [(2, 0, "2", 1), (2, 1, "G", 2)]


def test_header_cells_keep_their_spans() -> None:
    # Two header rows: TT merged down both, "Diện tích" across two columns.
    table = TableInfo(
        n_rows=3,
        n_cols=3,
        n_header_rows=2,
        cells=(
            cell(0, 0, "TT", 2),
            cell(0, 1, "Diện tích", 1, 2),
            cell(1, 1, "Tối thiểu"),
            cell(1, 2, "Tối đa"),
            cell(2, 0, "1"),
            cell(2, 1, "50"),
            cell(2, 2, "100"),
        ),
    )
    rows = build_table_rows(table, 2, 3)
    assert [(c.text(), c.row_span, c.col_span) for c in rows[0]] == [
        ("TT", 2, 1),
        ("Diện tích", 1, 2),
    ]
    assert [c.text() for c in rows[1]] == ["Tối thiểu", "Tối đa"]


def test_table_without_header_rows() -> None:
    table = TableInfo(
        n_rows=2, n_cols=1, n_header_rows=0, cells=(cell(0, 0, "a"), cell(1, 0, "b"))
    )
    assert shape(build_table_rows(table, 1, 2)) == [[(1, 0, "b", 1)]]


def test_empty_range_is_the_header_alone() -> None:
    assert shape(build_table_rows(MERGED, 3, 3)) == [
        [(0, 0, "TT", 1), (0, 1, "Nội dung", 1)]
    ]


def test_cells_are_copies_and_the_table_is_unchanged() -> None:
    before = MERGED.cells
    rows = build_table_rows(MERGED, 2, 4)
    assert MERGED.cells == before
    assert rows[1][0] is not MERGED.cells[2]  # the clipped copy of "A"
    assert MERGED.cells[2].row_span == 3


def test_paragraphs_are_kept() -> None:
    table = TableInfo(
        n_rows=2,
        n_cols=1,
        n_header_rows=1,
        cells=(cell(0, 0, "Nội dung"), cell(1, 0, "a. Một", 1, 1, "b. Hai")),
    )
    assert build_table_rows(table, 1, 2)[1][0].paragraphs == ("a. Một", "b. Hai")


@pytest.mark.parametrize(("start", "end"), [(1, 5), (1, 2), (2, 4), (3, 5), (4, 5)])
def test_same_placement_as_the_grid_layout(start: int, end: int) -> None:
    # The page and the embedded text put each cell's text on the same row and column.
    grid_lines = render_table_rows(MERGED, start, end, "grid").split("\n")
    from_grid = [
        [(col, text) for col, text in enumerate(line.split(" | ")) if text]
        for line in grid_lines
    ]
    from_rows = [
        [(c.col, c.text()) for c in row] for row in build_table_rows(MERGED, start, end)
    ]
    assert from_rows == from_grid
