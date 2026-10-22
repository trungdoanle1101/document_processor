"""Tests for chunking/outline_chunker.py, slice 1: sizing without the outline walk.

Front matter, body and end matter are chunked separately. Inside each, segments
are grouped greedily at segment boundaries up to max_tokens; a segment too big
on its own is a chunk by itself. Pieces without a text or table segment (only
images or headings) give no chunk, and ids are numbered after that, so they
have no gaps.

The size tests use documents without headings, so they keep holding once the
body is walked by structure (_chunk_children) instead of as one range. Sizes
use WordTokenizer: one token per whitespace-separated word.
"""

from typing import Any

import pytest

from document_processor.chunking.chunk import Chunk, PathStep
from document_processor.chunking.chunker import Chunker
from document_processor.chunking.outline_chunker import OutlineChunker, _path_to
from document_processor.chunking.tokenizer import WordTokenizer
from document_processor.core.models import (
    BlockType,
    LogicalDocument,
    PhysicalBlock,
    Segment,
    TableSegment,
    TextSegment,
)
from document_processor.logical.build import build_logical_document
from document_processor.logical.profiles.vn_regulation import VN_REGULATION
from tests.factories import (
    make_block,
    make_document,
    make_image_info,
    make_image_segment,
    make_logical_document,
    make_table_block,
    make_text_segment,
)


def chunker(max_tokens: int, **options: Any) -> OutlineChunker:
    """The one place that builds chunkers: a new required option changes only this."""
    return OutlineChunker(tokenizer=WordTokenizer(), max_tokens=max_tokens, **options)


def plain(*items: str | None, body: tuple[int, int] | None = None) -> LogicalDocument:
    """A document without headings: a string is a paragraph, None an image.

    The whole document is body unless ``body`` gives (body_start, body_end).
    """
    segments: list[Segment] = [
        make_image_segment(block_id=f"pb{i}") if item is None else make_text_segment(block_id=f"pb{i}", text=item)
        for i, item in enumerate(items, start=1)
    ]
    body_start, body_end = body if body is not None else (0, len(segments))
    return make_logical_document(segments=tuple(segments), body_start=body_start, body_end=body_end)


def ranges(chunks: tuple[Chunk, ...]) -> list[tuple[int, int]]:
    return [(c.start, c.end) for c in chunks]


class Blocks:
    """Physical blocks with ids pb1, pb2, … in the order they are made."""

    def __init__(self) -> None:
        self.blocks: list[PhysicalBlock] = []

    def _next_id(self) -> str:
        return f"pb{len(self.blocks) + 1}"

    def text(self, text: str) -> "Blocks":
        self.blocks.append(make_block(id=self._next_id(), text=text))
        return self

    def table(self) -> "Blocks":
        self.blocks.append(make_table_block(id=self._next_id()))
        return self

    def image(self) -> "Blocks":
        self.blocks.append(
            make_block(id=self._next_id(), block_type=BlockType.IMAGE, text="", image_info=make_image_info())
        )
        return self

    def build(self) -> LogicalDocument:
        return build_logical_document(VN_REGULATION, make_document(blocks=tuple(self.blocks)))


def regulation() -> LogicalDocument:
    """A small regulation built by the real profile.

        0  BAN HÀNH QUY ĐỊNH                              front matter
        1  # Chương I QUY ĐỊNH CHUNG                      n0 chuong [1, 15)
        2  # Điều 1. Phạm vi điều chỉnh                   n1 dieu   [2, 7)
        3  Quy định này quy định về quản lý kiến trúc
        4  (image)
        5  # 1.                                           n2 khoan  [5, 7)
        6  Áp dụng cho toàn hệ thống
        7  # Điều 2. Giải thích từ ngữ                    n3 dieu   [7, 15)
        8  # 1.                                           n4 khoan  [8, 10)
        9  Hệ thống là tập hợp các thành phần
        10 # 2.                                           n5 khoan  [10, 15)
        11 Kiến trúc gồm:
        12 # a)                                           n6 diem   [12, 15)
        13 Kiến trúc ứng dụng
        14 (table)                                       (tables never close a node)
        15 Nơi nhận:                                      end matter
        16 Lưu VT
    """
    return (
        Blocks()
        .text("BAN HÀNH QUY ĐỊNH")
        .text("Chương I")
        .text("QUY ĐỊNH CHUNG")
        .text("Điều 1. Phạm vi điều chỉnh")
        .text("Quy định này quy định về quản lý kiến trúc")
        .image()
        .text("1. Áp dụng cho toàn hệ thống")
        .text("Điều 2. Giải thích từ ngữ")
        .text("1. Hệ thống là tập hợp các thành phần")
        .text("2. Kiến trúc gồm:")
        .text("a) Kiến trúc ứng dụng")
        .table()
        .text("Nơi nhận:")
        .text("Lưu VT")
        .build()
    )


# ============================================================ configuration


def test_is_a_chunker() -> None:
    usable: Chunker = chunker(10)
    assert usable.chunk(plain("a")) == chunker(10).chunk(plain("a"))


@pytest.mark.parametrize("max_tokens", [0, -5])
def test_max_tokens_must_be_positive(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens > 0"):
        chunker(max_tokens)


def test_breadcrumb_is_not_counted_by_default() -> None:
    assert chunker(10).include_breadcrumb_in_size is False


# ============================================================ _path_to


def test_no_owner_no_path() -> None:
    assert _path_to(regulation(), None) == ()


def test_path_of_a_root() -> None:
    assert _path_to(regulation(), "n0") == (
        PathStep(node_id="n0", level="chuong", number="I", marker="Chương I", title="QUY ĐỊNH CHUNG", start=1),
    )


def test_path_is_root_first_owner_last() -> None:
    path = _path_to(regulation(), "n6")
    assert [(s.node_id, s.level, s.number, s.marker, s.start) for s in path] == [
        ("n0", "chuong", "I", "Chương I", 1),
        ("n3", "dieu", "2", "Điều 2.", 7),
        ("n5", "khoan", "2", "2.", 10),
        ("n6", "diem", "a", "a)", 12),
    ]
    assert [s.label for s in path] == ["Chương I QUY ĐỊNH CHUNG", "Điều 2. Giải thích từ ngữ", "2.", "a)"]


def test_unknown_owner_raises() -> None:
    with pytest.raises(KeyError, match="n99"):
        _path_to(regulation(), "n99")


# ============================================================ _fits


def test_fits_counts_the_rendered_text() -> None:
    document = plain("a b", "c d e")
    assert chunker(5)._fits(document, None, 0, 2)  # exactly at the budget
    assert not chunker(4)._fits(document, None, 0, 2)


def test_fits_counts_headings_inside_the_range() -> None:
    # Segment 9's khoản starts at 8: rendering [8, 10) gives "1." plus 8 words.
    document = regulation()
    assert chunker(9)._fits(document, "n4", 8, 10)
    assert not chunker(8)._fits(document, "n4", 8, 10)


def test_fits_ignores_the_breadcrumb_by_default() -> None:
    assert chunker(8)._fits(regulation(), "n4", 9, 10)  # 8 words; the 12-word breadcrumb is not counted


def test_fits_counts_the_breadcrumb_when_asked() -> None:
    # Range [9, 10) is after khoản 1's heading, so the breadcrumb is
    # "Chương I QUY ĐỊNH CHUNG" (5) + "Điều 2. Giải thích từ ngữ" (6) + "1." (1) = 12, plus 8 words.
    document = regulation()
    assert chunker(20, include_breadcrumb_in_size=True)._fits(document, "n4", 9, 10)
    assert not chunker(19, include_breadcrumb_in_size=True)._fits(document, "n4", 9, 10)


def test_fits_does_not_count_the_owners_heading_twice() -> None:
    # [8, 10) starts with khoản 1's heading, so the breadcrumb leaves "1." out: 11 + 9 = 20, not 21.
    document = regulation()
    assert chunker(20, include_breadcrumb_in_size=True)._fits(document, "n4", 8, 10)
    assert not chunker(19, include_breadcrumb_in_size=True)._fits(document, "n4", 8, 10)


# ============================================================ grouping by size


def test_everything_that_fits_is_one_chunk() -> None:
    chunks = chunker(100).chunk(plain("a b", "c d", "e f"))
    assert ranges(chunks) == [(0, 3)]
    assert chunks[0].text == "a b\nc d\ne f"


def test_greedy_split_at_segment_boundaries() -> None:
    document = plain("a b", "c d", "e f")
    assert ranges(chunker(4).chunk(document)) == [(0, 2), (2, 3)]
    assert ranges(chunker(3).chunk(document)) == [(0, 1), (1, 2), (2, 3)]


def test_no_chunk_goes_over_the_budget_when_segments_fit() -> None:
    document = plain(*(" ".join(["w"] * n) for n in (3, 1, 4, 1, 5, 2, 6, 3)))
    chunks = chunker(7).chunk(document)
    assert all(len(c.text.split()) <= 7 for c in chunks)
    assert ranges(chunks) == [(0, 2), (2, 4), (4, 6), (6, 7), (7, 8)]


def test_oversized_segment_is_a_chunk_by_itself() -> None:
    chunks = chunker(3).chunk(plain("a", "b c d e f", "g"))
    assert ranges(chunks) == [(0, 1), (1, 2), (2, 3)]
    assert chunks[1].text == "b c d e f"


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        (("b c d e f", "a"), [(0, 1), (1, 2)]),  # first
        (("a", "b c d e f"), [(0, 1), (1, 2)]),  # last
        (("b c d e f",), [(0, 1)]),  # alone
        (("b c d e f", "g h i j k"), [(0, 1), (1, 2)]),  # two in a row
    ],
    ids=["first", "last", "alone", "two-in-a-row"],
)
def test_oversized_segment_anywhere(items: tuple[str, ...], expected: list[tuple[int, int]]) -> None:
    assert ranges(chunker(3).chunk(plain(*items))) == expected


def test_images_cost_nothing_and_stay_with_their_neighbours() -> None:
    chunks = chunker(4).chunk(plain("a b", None, "c d"))
    assert ranges(chunks) == [(0, 3)]
    assert chunks[0].text == "a b\nc d"
    assert chunks[0].block_ids == ("pb1", "pb2", "pb3")


# ============================================================ pieces without content


def test_image_only_part_gives_no_chunk() -> None:
    # A logo as the whole front matter.
    chunks = chunker(10).chunk(plain(None, "a b", body=(1, 2)))
    assert [(c.part, c.start, c.end) for c in chunks] == [("body", 1, 2)]


def test_ids_are_numbered_after_dropping_pieces() -> None:
    # The image-only front matter would have been c0.
    chunks = chunker(10).chunk(plain(None, "a", "b", None, "Lưu VT", body=(1, 3)))
    assert [(c.id, c.part) for c in chunks] == [("c0", "body"), ("c1", "end_matter")]


def test_empty_document_gives_no_chunks() -> None:
    assert chunker(10).chunk(make_logical_document()) == ()


# ============================================================ parts


def test_parts_are_never_mixed() -> None:
    # Everything would fit in one chunk, but each part is chunked on its own.
    chunks = chunker(100).chunk(plain("CHÍNH PHỦ", "Nội dung", "Nơi nhận:", body=(1, 2)))
    assert [(c.part, c.start, c.end) for c in chunks] == [
        ("front_matter", 0, 1),
        ("body", 1, 2),
        ("end_matter", 2, 3),
    ]


def test_front_and_end_matter_have_no_path() -> None:
    chunks = chunker(100).chunk(regulation())
    assert [(c.part, c.path, c.headings) for c in chunks if c.part != "body"] == [
        ("front_matter", (), ()),
        ("end_matter", (), ()),
    ]


# ============================================================ chunk fields


def test_chunk_fields() -> None:
    document = plain("a b", "c d", body=(0, 2))
    (chunk,) = chunker(10).chunk(document)
    assert chunk.id == "c0"
    assert chunk.logical_document_id == document.id
    assert (chunk.part, chunk.start, chunk.end) == ("body", 0, 2)
    assert (chunk.table_rows, chunk.text_span) == (None, None)
    assert chunk.text == "a b\nc d"
    assert chunk.block_ids == ("pb1", "pb2")


def test_block_ids_are_unique_and_in_order() -> None:
    # "1. Áp dụng…" gives a heading segment and a text segment from the same block.
    document = regulation()
    body = [c for c in chunker(100).chunk(document) if c.part == "body"]
    all_ids = [block_id for c in body for block_id in c.block_ids]
    assert len(all_ids) == len(set(all_ids))
    assert body[0].block_ids[:4] == ("pb2", "pb4", "pb5", "pb6")  # pb3 (the title) is in head_block_ids only


def test_text_keeps_headings_where_they_are() -> None:
    (body,) = [c for c in chunker(100).chunk(regulation()) if c.part == "body"]
    assert body.text.splitlines()[:4] == [
        "Chương I QUY ĐỊNH CHUNG",
        "Điều 1. Phạm vi điều chỉnh",
        "Quy định này quy định về quản lý kiến trúc",
        "1.",
    ]


# ============================================================ the whole document


@pytest.mark.parametrize("max_tokens", [1, 5, 12, 30, 1000])
@pytest.mark.parametrize("include_breadcrumb_in_size", [False, True])
def test_chunks_cover_every_content_segment_once_and_in_order(
    max_tokens: int, include_breadcrumb_in_size: bool
) -> None:
    document = regulation()
    chunks = chunker(max_tokens, include_breadcrumb_in_size=include_breadcrumb_in_size).chunk(document)

    # in order, no overlap
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:]))
    assert [c.id for c in chunks] == [f"c{i}" for i in range(len(chunks))]

    # every text or table segment in exactly one chunk, in the part it belongs to
    def part_of(index: int) -> str:
        if index < document.body_start:
            return "front_matter"
        return "body" if index < document.body_end else "end_matter"

    for index, segment in enumerate(document.segments):
        if isinstance(segment, (TextSegment, TableSegment)):
            owners = [c for c in chunks if c.start <= index < c.end]
            assert len(owners) == 1, index
            assert owners[0].part == part_of(index)


@pytest.mark.parametrize("max_tokens", [1, 3, 5, 12])
def test_only_single_segments_go_over_the_budget(max_tokens: int) -> None:
    # A heading or image before a too-big segment is closed off on its own (and dropped), so an
    # over-budget chunk is always exactly that one segment.
    for chunk in chunker(max_tokens).chunk(regulation()):
        if len(chunk.text.split()) > max_tokens:
            assert chunk.end - chunk.start == 1, (chunk.start, chunk.end, chunk.text)