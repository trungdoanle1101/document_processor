"""Tests for chunking/render.py: a range of segments as plain text.

A heading segment reads as "marker title", a text segment as its text, a table
as its tab-separated text (step 1; real rendering comes with row splitting),
and an image as nothing. Lines are joined with "\\n", except that an item's
marker and the rest of its own paragraph share a line ("1. Nội dung"): a text
segment right after an untitled heading with a marker, from the same block, is
joined to it with a space.

The running example:

    0  CHÍNH PHỦ                     front matter
    1  # Điều 1. Phạm vi điều chỉnh  dieu-1  [1, 6)
    2  Nghị định này quy định về:
    3  (image)
    4  # 1.                          khoan-1 [4, 6)
    5  (table) TT | Nội dung
    6  Nơi nhận:                     end matter
"""

import pytest

from document_processor.chunking.render import render_segments
from document_processor.core.models import BlockType, HeadingSegment, LogicalDocument, LogicalNode, Segment
from document_processor.logical import VN_REGULATION, build_logical_document
from tests.chunking.test_outline_chunker import regulation
from tests.factories import (
    heading_for,
    make_block,
    make_document,
    make_heading_segment,
    make_image_segment,
    make_list_info,
    make_logical_document,
    make_logical_node,
    make_table_segment,
    make_text_segment,
)

DIEU = make_logical_node(id="dieu-1", head_block_ids=("pb2",), start=1, end=6)
KHOAN = make_logical_node(
    id="khoan-1", parent_id="dieu-1", level="khoan", marker="1.", title=None, head_block_ids=("pb5",), start=4, end=6
)
SEGMENTS: tuple[Segment, ...] = (
    make_text_segment(block_id="pb1", text="CHÍNH PHỦ"),
    heading_for(DIEU),
    make_text_segment(block_id="pb3", text="Nghị định này quy định về:"),
    make_image_segment(block_id="pb4"),
    heading_for(KHOAN),
    make_table_segment(block_id="pb6"),
    make_text_segment(block_id="pb7", text="Nơi nhận:"),
)
TABLE_TEXT = "TT\tNội dung\n1\tPhạm vi điều chỉnh"


def document(segments: tuple[Segment, ...] = SEGMENTS, nodes: tuple[LogicalNode, ...] = (DIEU, KHOAN)) -> LogicalDocument:
    return make_logical_document(segments=segments, nodes=nodes, body_start=1, body_end=len(segments) - 1)


# ---------- what each kind renders to


def test_whole_document() -> None:
    assert render_segments(document(), 0, 7) == "\n".join(
        ["CHÍNH PHỦ", "Điều 1. Phạm vi điều chỉnh", "Nghị định này quy định về:", "1.", TABLE_TEXT, "Nơi nhận:"]
    )


def test_heading_reads_as_marker_and_title() -> None:
    assert render_segments(document(), 1, 2) == "Điều 1. Phạm vi điều chỉnh"


def test_heading_without_a_title_is_its_marker() -> None:
    assert render_segments(document(), 4, 5) == "1."


def test_heading_with_neither_marker_nor_title_adds_no_line() -> None:
    bare = make_logical_node(id="x", level="x", marker=None, title=None, head_block_ids=("pb2",), start=1, end=3)
    segments = (
        make_text_segment(block_id="pb1", text="A"),
        heading_for(bare),
        make_text_segment(block_id="pb3", text="B"),
        make_text_segment(block_id="pb4", text="C"),
    )
    assert render_segments(document(segments, (bare,)), 0, 3) == "A\nB"  # not "A\n\nB"


def test_text_is_kept_as_written() -> None:
    assert render_segments(document(), 2, 3) == "Nghị định này quy định về:"


def test_table_renders_its_text() -> None:
    assert render_segments(document(), 5, 6) == TABLE_TEXT


def test_image_renders_nothing() -> None:
    assert render_segments(document(), 3, 4) == ""
    assert render_segments(document(), 2, 5) == "Nghị định này quy định về:\n1."  # no blank line for the image


def test_a_node_range_starts_with_its_heading() -> None:
    assert render_segments(document(), KHOAN.start, KHOAN.end) == "1.\n" + TABLE_TEXT


# ---------- ranges


def test_empty_range_is_empty_text() -> None:
    assert render_segments(document(), 3, 3) == ""
    assert render_segments(document(), 7, 7) == ""


@pytest.mark.parametrize(("start", "end"), [(-1, 2), (3, 2), (0, 8), (8, 8)])
def test_range_outside_the_segments_rejected(start: int, end: int) -> None:
    with pytest.raises(ValueError, match=r"0 <= start <= end <= 7"):
        render_segments(document(), start, end)


def test_heading_segment_factory_renders_like_the_node() -> None:
    # The chunk breadcrumb (PathStep.label) and the rendered heading use the same heading_label.
    segments = (make_heading_segment(block_id="pb1", node_id="dieu-1"), make_text_segment(block_id="pb2"))
    node = make_logical_node(start=0, end=2)
    rendered = render_segments(make_logical_document(segments=segments, nodes=(node,), body_start=0, body_end=2), 0, 1)
    assert rendered == "Điều 1. Phạm vi điều chỉnh"



# ============================================================ an item's marker and its text share a line
#
# regulation(): segment 5 is khoản 1's heading "1." and segment 6 the rest of the
# same paragraph, "Áp dụng cho toàn hệ thống".


def test_marker_and_its_text_share_a_line() -> None:
    assert render_segments(regulation(), 5, 7) == "1. Áp dụng cho toàn hệ thống"


def test_inside_a_longer_range() -> None:
    lines = render_segments(regulation(), 2, 7).splitlines()
    assert lines == [
        "Điều 1. Phạm vi điều chỉnh",
        "Quy định này quy định về quản lý kiến trúc",
        "1. Áp dụng cho toàn hệ thống",
    ]


def test_text_starting_the_range_stands_alone() -> None:
    # The marker is outside the range (in the chunk's breadcrumb): nothing to join with.
    assert render_segments(regulation(), 6, 7) == "Áp dụng cho toàn hệ thống"


def test_marker_alone_at_the_end_of_the_range() -> None:
    assert render_segments(regulation(), 5, 6) == "1."


def test_a_list_generated_marker_joins_too() -> None:
    # The marker comes from Word's numbering; the text segment is the whole paragraph, same block.
    blocks = (
        make_block(id="pb1", text="Điều 1. Phạm vi"),
        make_block(
            id="pb2",
            block_type=BlockType.LIST_ITEM,
            text="Nội dung khoản",
            list_info=make_list_info(displayed_marker="1."),
        ),
    )
    document = build_logical_document(VN_REGULATION, make_document(blocks=blocks))
    assert render_segments(document, 0, len(document.segments)) == "Điều 1. Phạm vi\n1. Nội dung khoản"


def _hand_made(heading: HeadingSegment, *after: Segment, end: int) -> LogicalDocument:
    node = make_logical_node(
        id=heading.node_id, marker=heading.marker, title=heading.title, head_block_ids=(heading.block_id,), start=0, end=end
    )
    return make_logical_document(segments=(heading, *after), nodes=(node,), body_start=0, body_end=end)


def test_a_titled_heading_keeps_its_own_line() -> None:
    heading = make_heading_segment(block_id="pb1", node_id="a", marker="Điều 1.", title="Phạm vi")
    document = _hand_made(heading, make_text_segment(block_id="pb1", text="Nội dung"), end=2)
    assert render_segments(document, 0, 2) == "Điều 1. Phạm vi\nNội dung"


def test_text_from_another_block_keeps_its_own_line() -> None:
    heading = make_heading_segment(block_id="pb1", node_id="a", marker="1.", title=None)
    document = _hand_made(heading, make_text_segment(block_id="pb2", text="Đoạn khác"), end=2)
    assert render_segments(document, 0, 2) == "1.\nĐoạn khác"


def test_a_heading_without_marker_or_title_adds_nothing_to_join_onto() -> None:
    heading = make_heading_segment(block_id="pb1", node_id="a", marker=None, title=None)
    document = _hand_made(heading, make_text_segment(block_id="pb1", text="Nội dung"), end=2)
    assert render_segments(document, 0, 2) == "Nội dung"


def test_only_the_first_text_after_the_marker_joins() -> None:
    heading = make_heading_segment(block_id="pb1", node_id="a", marker="1.", title=None)
    document = _hand_made(
        heading, make_text_segment(block_id="pb1", text="Nội dung"), make_text_segment(block_id="pb2", text="Tiếp"), end=3
    )
    assert render_segments(document, 0, 3) == "1. Nội dung\nTiếp"