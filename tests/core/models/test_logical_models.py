"""Validation rules of the logical models: segments, nodes, and the outline.

The running example (see ``outline_document``) is a small decree:

    segments                                   outline
    0  CHÍNH PHỦ                  front matter
    1  Nghị định này quy định về: body         chuong-I            [1, 4)
    2  Kiến trúc tổng thể.                       chuong-I/dieu-1   [1, 4)
    3  Kiến trúc dữ liệu.                          .../khoan-1     [2, 3)
    4  Nơi nhận:                  end matter         .../khoan-2   [3, 4)

Chương I and Điều 1 start at the same segment because their markers and titles
were consumed by the builder; segment 1 is Điều 1's own lead-in.
"""

import dataclasses
from typing import Any

import pytest

from document_processor.core.models.logical import (
    ImageSegment,
    LogicalDocument,
    LogicalNode,
    TableSegment,
    TextSegment,
)
from tests.factories import (
    make_image_info,
    make_image_segment,
    make_logical_document,
    make_logical_node,
    make_table_info,
    make_table_segment,
    make_text_segment,
)

INVISIBLE = ["", "   ", "\t\n", "​"]
INVISIBLE_IDS = ["empty", "spaces", "whitespace", "zero-width"]

CHUONG = "chuong-I"
DIEU = "chuong-I/dieu-1"
KHOAN_1 = "chuong-I/dieu-1/khoan-1"
KHOAN_2 = "chuong-I/dieu-1/khoan-2"


def text(block_id: str, value: str) -> TextSegment:
    return make_text_segment(block_id=block_id, text=value)


SEGMENTS = (
    text("pb1", "CHÍNH PHỦ"),
    text("pb5", "Nghị định này quy định về:"),
    text("pb6", "Kiến trúc tổng thể."),
    text("pb7", "Kiến trúc dữ liệu."),
    text("pb8", "Nơi nhận:"),
)


def node(node_id: str, parent_id: str | None, start: int, end: int, **overrides: Any) -> LogicalNode:
    fields: dict[str, Any] = {
        "id": node_id,
        "parent_id": parent_id,
        "level": "x",
        "number": None,
        "marker": None,
        "title": None,
        "head_block_ids": (f"head-{node_id}",),
        "start": start,
        "end": end,
    }
    fields.update(overrides)
    return make_logical_node(**fields)


OUTLINE = (
    node(CHUONG, None, 1, 4, level="chuong", number="I", marker="Chương I",
         title="QUY ĐỊNH CHUNG", head_block_ids=("pb2", "pb3")),
    node(DIEU, CHUONG, 1, 4, level="dieu", number="1", marker="Điều 1.",
         title="Phạm vi điều chỉnh", head_block_ids=("pb4",)),
    node(KHOAN_1, DIEU, 2, 3, level="khoan", number="1", marker="1.", head_block_ids=("pb6",)),
    node(KHOAN_2, DIEU, 3, 4, level="khoan", number="2", marker="2.", head_block_ids=("pb7",)),
)


def outline_document(nodes: tuple[LogicalNode, ...] = OUTLINE, **overrides: Any) -> LogicalDocument:
    fields: dict[str, Any] = {
        "segments": SEGMENTS,
        "nodes": nodes,
        "body_start": 1,
        "body_end": 4,
    }
    fields.update(overrides)
    return make_logical_document(**fields)


# ---------- segments


def test_text_segment_is_accepted() -> None:
    segment = make_text_segment(block_id="pb5", text="Nghị định này quy định về:")
    assert segment.block_id == "pb5"
    assert segment.text == "Nghị định này quy định về:"


def test_image_and_table_segments_carry_their_payload() -> None:
    image = make_image_info()
    table = make_table_info()
    assert make_image_segment(image=image).image == image
    assert make_table_segment(table=table).table == table


@pytest.mark.parametrize("block_id", INVISIBLE, ids=INVISIBLE_IDS)
@pytest.mark.parametrize(
    "make",
    [make_text_segment, make_image_segment, make_table_segment],
    ids=["text", "image", "table"],
)
def test_every_segment_rejects_blank_block_id(make: Any, block_id: str) -> None:
    with pytest.raises(ValueError, match="block_id"):
        make(block_id=block_id)


@pytest.mark.parametrize(
    ("make", "name"),
    [(make_text_segment, "TextSegment"), (make_image_segment, "ImageSegment"), (make_table_segment, "TableSegment")],
)
def test_segment_error_names_the_concrete_class(make: Any, name: str) -> None:
    with pytest.raises(ValueError, match=rf"^{name}\b"):
        make(block_id="")


@pytest.mark.parametrize("value", INVISIBLE, ids=INVISIBLE_IDS)
def test_text_segment_rejects_invisible_text(value: str) -> None:
    # A cut heading that leaves nothing produces no segment at all.
    with pytest.raises(ValueError, match="text"):
        make_text_segment(text=value)


def test_text_segment_keeps_text_as_given() -> None:
    assert make_text_segment(text="  a  b ").text == "  a  b "


def test_segments_are_frozen() -> None:
    segment = make_text_segment()
    with pytest.raises(dataclasses.FrozenInstanceError):
        segment.text = "changed"  # type: ignore[misc]


def test_segments_require_keyword_arguments() -> None:
    with pytest.raises(TypeError):
        TextSegment("pb1", "text")  # type: ignore[call-arg]


def test_segment_factories_build_the_right_kinds() -> None:
    assert isinstance(make_text_segment(), TextSegment)
    assert isinstance(make_image_segment(), ImageSegment)
    assert isinstance(make_table_segment(), TableSegment)


# ---------- LogicalNode


def test_valid_node_is_accepted() -> None:
    n = make_logical_node()
    assert n.id == "dieu-1"
    assert n.parent_id is None
    assert (n.start, n.end) == (0, 1)


def test_optional_fields_may_be_none() -> None:
    n = make_logical_node(number=None, marker=None, title=None)
    assert (n.number, n.marker, n.title) == (None, None, None)


def test_empty_range_is_allowed() -> None:
    # A heading with no content of its own.
    n = make_logical_node(start=3, end=3)
    assert n.start == n.end == 3


def test_head_may_span_several_blocks() -> None:
    # "Chương I" with its title on the next block.
    assert make_logical_node(head_block_ids=("pb2", "pb3")).head_block_ids == ("pb2", "pb3")


@pytest.mark.parametrize("node_id", INVISIBLE, ids=INVISIBLE_IDS)
def test_node_rejects_blank_id(node_id: str) -> None:
    with pytest.raises(ValueError, match="'id'"):
        make_logical_node(id=node_id)


@pytest.mark.parametrize("field", ["level"])
@pytest.mark.parametrize("value", INVISIBLE, ids=INVISIBLE_IDS)
def test_node_rejects_blank_level(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=f"'{field}'"):
        make_logical_node(**{field: value})


@pytest.mark.parametrize("field", ["parent_id", "number", "marker", "title"])
@pytest.mark.parametrize("value", INVISIBLE, ids=INVISIBLE_IDS)
def test_node_rejects_blank_optional_fields(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=f"'{field}'"):
        make_logical_node(**{field: value})


def test_node_cannot_be_its_own_parent() -> None:
    with pytest.raises(ValueError, match="parent_id"):
        make_logical_node(id="dieu-1", parent_id="dieu-1")


def test_node_requires_a_head_block() -> None:
    with pytest.raises(ValueError, match="head_block_ids"):
        make_logical_node(head_block_ids=())


@pytest.mark.parametrize("block_id", INVISIBLE, ids=INVISIBLE_IDS)
def test_node_rejects_blank_head_block_id(block_id: str) -> None:
    with pytest.raises(ValueError, match="head_block_ids"):
        make_logical_node(head_block_ids=("pb1", block_id))


def test_node_rejects_duplicate_head_block_ids() -> None:
    with pytest.raises(ValueError, match="duplicate value 'pb2'"):
        make_logical_node(head_block_ids=("pb2", "pb3", "pb2"))


@pytest.mark.parametrize(("start", "end"), [(-1, 1), (2, 1), (-2, -1)])
def test_node_rejects_invalid_range(start: int, end: int) -> None:
    with pytest.raises(ValueError, match=rf"got start={start}, end={end}"):
        make_logical_node(start=start, end=end)


def test_node_error_names_the_node() -> None:
    with pytest.raises(ValueError, match=r"^LogicalNode 'dieu-7'"):
        make_logical_node(id="dieu-7", level="")


def test_node_is_frozen() -> None:
    n = make_logical_node()
    with pytest.raises(dataclasses.FrozenInstanceError):
        n.start = 5  # type: ignore[misc]


# ---------- LogicalDocument: identity and body range


def test_empty_document_is_accepted() -> None:
    doc = make_logical_document()
    assert doc.segments == ()
    assert doc.nodes == ()
    assert doc.root_ids == ()


@pytest.mark.parametrize(
    "field", ["id", "physical_document_id", "source_file_hash", "profile_name", "profile_version"]
)
@pytest.mark.parametrize("value", INVISIBLE, ids=INVISIBLE_IDS)
def test_document_rejects_blank_identity_fields(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=f"'{field}'"):
        make_logical_document(**{field: value})


def test_document_without_hierarchy_is_accepted() -> None:
    # Only paragraphs, no headings: segments with an empty outline.
    doc = make_logical_document(segments=SEGMENTS, body_start=0, body_end=len(SEGMENTS))
    assert doc.nodes == ()
    assert doc.root_ids == ()
    assert doc.get_children(None) == ()


@pytest.mark.parametrize(
    ("body_start", "body_end"),
    [(0, 0), (0, 5), (1, 4), (5, 5), (2, 2)],
    ids=["all-front", "all-body", "front-body-end", "all-front-at-end", "empty-body"],
)
def test_body_range_inside_the_segments_is_accepted(body_start: int, body_end: int) -> None:
    doc = make_logical_document(segments=SEGMENTS, body_start=body_start, body_end=body_end)
    assert (doc.body_start, doc.body_end) == (body_start, body_end)


@pytest.mark.parametrize(
    ("body_start", "body_end"),
    [(-1, 2), (3, 2), (0, 6), (6, 6)],
    ids=["negative", "start-after-end", "end-past-segments", "both-past-segments"],
)
def test_body_range_outside_the_segments_is_rejected(body_start: int, body_end: int) -> None:
    with pytest.raises(ValueError, match="body_start"):
        make_logical_document(segments=SEGMENTS, body_start=body_start, body_end=body_end)


def test_document_is_frozen() -> None:
    doc = outline_document()
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.body_start = 0  # type: ignore[misc]


# ---------- LogicalDocument: the outline example


def test_outline_example_is_accepted() -> None:
    doc = outline_document()
    assert doc.root_ids == (CHUONG,)
    assert doc.get_children(None) == (CHUONG,)
    assert doc.get_children(CHUONG) == (DIEU,)
    assert doc.get_children(DIEU) == (KHOAN_1, KHOAN_2)
    assert doc.get_children(KHOAN_1) == ()


def test_get_node() -> None:
    doc = outline_document()
    assert doc.get_node(DIEU).title == "Phạm vi điều chỉnh"


def test_get_node_rejects_unknown_id() -> None:
    with pytest.raises(KeyError, match="dieu-9"):
        outline_document().get_node("dieu-9")


def test_get_children_rejects_unknown_id() -> None:
    with pytest.raises(KeyError, match="dieu-9"):
        outline_document().get_children("dieu-9")


def test_derived_lookups_are_not_compared_or_shown() -> None:
    first = outline_document()
    second = outline_document()
    assert first == second
    assert "_node_by_id" not in repr(first)
    assert "_children_by_id" not in repr(first)


def test_derived_lookups_are_read_only() -> None:
    doc = outline_document()
    with pytest.raises(TypeError):
        doc._children_by_id[None] = ()  # type: ignore[index]


# ---------- outline: shapes that are allowed


def test_several_roots() -> None:
    nodes = (node("a", None, 1, 2), node("b", None, 2, 4))
    assert outline_document(nodes).root_ids == ("a", "b")


def test_parent_and_first_child_may_share_a_start() -> None:
    nodes = (node("a", None, 1, 4), node("a/1", "a", 1, 2), node("a/2", "a", 2, 4))
    assert outline_document(nodes).get_children("a") == ("a/1", "a/2")


def test_body_text_outside_any_node_is_allowed() -> None:
    # Segment 1 (a preamble) and segment 3 belong to the document itself.
    nodes = (node("a", None, 2, 3),)
    assert outline_document(nodes).root_ids == ("a",)


def test_gaps_between_siblings_are_allowed() -> None:
    nodes = (node("a", None, 1, 4), node("a/1", "a", 1, 2), node("a/2", "a", 3, 4))
    assert outline_document(nodes).get_children("a") == ("a/1", "a/2")


def test_empty_nodes_at_the_same_position() -> None:
    # Two headings with nothing between them.
    nodes = (node("a", None, 2, 2), node("b", None, 2, 2), node("c", None, 2, 4))
    assert outline_document(nodes).root_ids == ("a", "b", "c")


def test_empty_child_at_its_parents_end() -> None:
    nodes = (node("a", None, 1, 3), node("a/1", "a", 3, 3))
    assert outline_document(nodes).get_children("a") == ("a/1",)


def test_deep_nesting_and_return_to_a_shallower_level() -> None:
    nodes = (
        node("a", None, 1, 4),
        node("a/1", "a", 1, 3),
        node("a/1/x", "a/1", 1, 2),
        node("a/1/x/i", "a/1/x", 1, 2),
        node("a/2", "a", 3, 4),          # back up two levels
    )
    doc = outline_document(nodes)
    assert doc.get_children("a") == ("a/1", "a/2")
    assert doc.get_children("a/1/x") == ("a/1/x/i",)


def test_root_after_a_deep_branch() -> None:
    nodes = (node("a", None, 1, 3), node("a/1", "a", 1, 3), node("a/1/x", "a/1", 2, 3), node("b", None, 3, 4))
    assert outline_document(nodes).root_ids == ("a", "b")


# ---------- outline: shapes that are rejected


def test_duplicate_node_ids_rejected() -> None:
    nodes = (node("a", None, 1, 2), node("a", None, 2, 4))
    with pytest.raises(ValueError, match="duplicate value 'a'"):
        outline_document(nodes)


def test_unknown_parent_rejected() -> None:
    nodes = (node("a", None, 1, 4), node("a/1", "missing", 1, 2))
    with pytest.raises(ValueError, match="unknown parent 'missing'"):
        outline_document(nodes)


def test_child_listed_before_its_parent_rejected() -> None:
    nodes = (OUTLINE[0], OUTLINE[2], OUTLINE[1], OUTLINE[3])
    with pytest.raises(ValueError, match="not an open ancestor"):
        outline_document(nodes)


def test_child_listed_after_its_parents_branch_closed_rejected() -> None:
    # a/2 belongs to a, but b (a root) came in between: not pre-order.
    nodes = (node("a", None, 1, 2), node("b", None, 2, 3), node("a/2", "a", 1, 2))
    with pytest.raises(ValueError, match="not an open ancestor"):
        outline_document(nodes)


def test_child_outside_its_parent_rejected() -> None:
    nodes = (node("a", None, 1, 3), node("a/1", "a", 2, 4))
    with pytest.raises(ValueError, match="outside parent's range"):
        outline_document(nodes)


def test_child_starting_before_its_parent_rejected() -> None:
    nodes = (node("a", None, 2, 4), node("a/1", "a", 1, 3))
    with pytest.raises(ValueError, match="outside parent's range"):
        outline_document(nodes)


@pytest.mark.parametrize(("start", "end"), [(0, 2), (3, 5), (0, 5)], ids=["front", "end", "both"])
def test_root_outside_the_body_rejected(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="outside parent's range"):
        outline_document((node("a", None, start, end),))


def test_node_beyond_the_segments_rejected() -> None:
    with pytest.raises(ValueError, match="end <= 5"):
        outline_document((node("a", None, 1, 9),), body_end=5)


def test_overlapping_siblings_rejected() -> None:
    nodes = (node("a", None, 1, 4), node("a/1", "a", 1, 3), node("a/2", "a", 2, 4))
    with pytest.raises(ValueError, match="overlaps or precedes"):
        outline_document(nodes)


def test_siblings_out_of_order_rejected() -> None:
    nodes = (node("a", None, 1, 4), node("a/2", "a", 3, 4), node("a/1", "a", 1, 2))
    with pytest.raises(ValueError, match="overlaps or precedes"):
        outline_document(nodes)


def test_overlapping_roots_rejected() -> None:
    nodes = (node("a", None, 1, 3), node("b", None, 2, 4))
    with pytest.raises(ValueError, match="overlaps or precedes"):
        outline_document(nodes)


def test_outline_error_names_document_node_and_position() -> None:
    nodes = (node("a", None, 1, 4), node("a/1", "a", 1, 3), node("a/2", "a", 2, 4))
    with pytest.raises(ValueError, match=r"^LogicalDocument 'ld1', node 'a/2' at pos 2"):
        outline_document(nodes)