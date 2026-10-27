"""Tests for chunking/outline_chunker.py: structure first, then size.

Front matter, body and end matter are chunked separately. In the body, a node
that fits is one chunk. One that does not is chunked over its children and its
own segments (heading, lead-in, gaps, text after the last child), grouped
greedily without ever cutting a child: a group of exactly one child is owned by
that child, any other group by the node; a child too big alone is chunked the
same way, one level down. With keep_sections_apart (the default), a section
child (Chương, Mục, Điều) is never grouped with its siblings: it is chunked on
its own, and only items (khoản, điểm, gạch) and the node's own segments group. A leaf that does not fit is split by size inside its
own range. Body text
outside every root is owned by nothing. Size splitting groups segments greedily
at segment boundaries up to max_tokens; a segment too big on its own is a
chunk by itself. Pieces without a text or table segment (only images or
headings) give no chunk, and ids are numbered after that, so they have no gaps.

The size tests use documents without headings, where the body is one range.
Sizes use WordTokenizer: one token per whitespace-separated word.
"""

from typing import Any

import pytest

from document_processor.chunking.chunk import Chunk, PathStep
from document_processor.chunking.chunker import Chunker
from document_processor.chunking.outline_chunker import OutlineChunker, _ChunkingRun, _path_to, _Piece
from document_processor.chunking.tokenizer import WordTokenizer
from document_processor.core.models import (
    BlockType,
    TableCell,
    TableInfo,
    LogicalDocument,
    PhysicalBlock,
    Segment,
    TableSegment,
    TextSegment,
)
from document_processor.logical.build import build_logical_document
from document_processor.logical.profiles.vn_regulation import VN_REGULATION
from tests.factories import (
    heading_for,
    make_block,
    make_document,
    make_image_info,
    make_image_segment,
    make_logical_document,
    make_logical_node,
    make_table_block,
    make_table_segment,
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


def fits(chunker: OutlineChunker, document: LogicalDocument, owner_id: str | None, start: int, end: int) -> bool:
    """Whether segments [start, end), owned by owner_id, fit the chunker's budget."""
    piece = _Piece(part="body", owner_id=owner_id, start=start, end=end)
    return _ChunkingRun(chunker, document)._fits(piece)


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


class LimitedTokenizer(WordTokenizer):
    """Counts words, for a model that reads at most `limit` of them."""

    def __init__(self, limit: int | None) -> None:
        self._limit = limit

    @property
    def model_max_tokens(self) -> int | None:
        return self._limit


def test_budget_at_the_models_limit_is_accepted() -> None:
    assert OutlineChunker(tokenizer=LimitedTokenizer(100), max_tokens=100).max_tokens == 100


def test_budget_above_the_models_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_tokens=101 is above the tokenizer's model limit of 100"):
        OutlineChunker(tokenizer=LimitedTokenizer(100), max_tokens=101)


def test_a_tokenizer_without_a_limit_allows_any_budget() -> None:
    assert OutlineChunker(tokenizer=LimitedTokenizer(None), max_tokens=10**6).max_tokens == 10**6


def test_breadcrumb_is_not_counted_by_default() -> None:
    assert chunker(10).include_breadcrumb_in_size is False


# ============================================================ _path_to


def path_without_lead_ins(document: LogicalDocument, node_id: str | None) -> tuple[PathStep, ...]:
    """The path structure alone: lead-ins off, start past the end."""
    return _path_to(document, node_id, start=len(document.segments), lead_in_words=0)


def test_no_owner_no_path() -> None:
    assert path_without_lead_ins(regulation(), None) == ()


def test_path_of_a_root() -> None:
    assert path_without_lead_ins(regulation(), "n0") == (
        PathStep(node_id="n0", level="chuong", number="I", marker="Chương I", title="QUY ĐỊNH CHUNG", start=1),
    )


def test_path_is_root_first_owner_last() -> None:
    path = path_without_lead_ins(regulation(), "n6")
    assert [(s.node_id, s.level, s.number, s.marker, s.start) for s in path] == [
        ("n0", "chuong", "I", "Chương I", 1),
        ("n3", "dieu", "2", "Điều 2.", 7),
        ("n5", "khoan", "2", "2.", 10),
        ("n6", "diem", "a", "a)", 12),
    ]
    assert [s.label for s in path] == ["Chương I QUY ĐỊNH CHUNG", "Điều 2. Giải thích từ ngữ", "2.", "a)"]


def test_unknown_owner_raises() -> None:
    with pytest.raises(KeyError, match="n99"):
        path_without_lead_ins(regulation(), "n99")


# ============================================================ _fits


def test_fits_counts_the_rendered_text() -> None:
    document = plain("a b", "c d e")
    assert fits(chunker(5), document, None, 0, 2)  # exactly at the budget
    assert not fits(chunker(4), document, None, 0, 2)


def test_fits_counts_headings_inside_the_range() -> None:
    # Segment 9's khoản starts at 8: rendering [8, 10) gives "1." plus 8 words.
    document = regulation()
    assert fits(chunker(9), document, "n4", 8, 10)
    assert not fits(chunker(8), document, "n4", 8, 10)


def test_fits_ignores_the_breadcrumb_by_default() -> None:
    assert fits(chunker(8), regulation(), "n4", 9, 10)  # 8 words; the 12-word breadcrumb is not counted


def test_fits_counts_the_breadcrumb_when_asked() -> None:
    # Range [9, 10) is after khoản 1's heading, so the breadcrumb is
    # "Chương I QUY ĐỊNH CHUNG" (5) + "Điều 2. Giải thích từ ngữ" (6) + "1." (1) = 12, plus 8 words.
    document = regulation()
    assert fits(chunker(20, include_breadcrumb_in_size=True), document, "n4", 9, 10)
    assert not fits(chunker(19, include_breadcrumb_in_size=True), document, "n4", 9, 10)


def test_fits_does_not_count_the_owners_heading_twice() -> None:
    # [8, 10) starts with khoản 1's heading, so the breadcrumb leaves "1." out: 11 + 9 = 20, not 21.
    document = regulation()
    assert fits(chunker(20, include_breadcrumb_in_size=True), document, "n4", 8, 10)
    assert not fits(chunker(19, include_breadcrumb_in_size=True), document, "n4", 8, 10)


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
    body = [c for c in chunker(100, keep_sections_apart=False).chunk(document) if c.part == "body"]
    all_ids = [block_id for c in body for block_id in c.block_ids]
    assert len(all_ids) == len(set(all_ids))
    assert body[0].block_ids[:4] == ("pb2", "pb4", "pb5", "pb6")  # pb3 (the title) is in head_block_ids only


def test_text_keeps_headings_where_they_are() -> None:
    (body,) = [c for c in chunker(100, keep_sections_apart=False).chunk(regulation()) if c.part == "body"]
    assert body.text.splitlines()[:4] == [
        "Chương I QUY ĐỊNH CHUNG",
        "Điều 1. Phạm vi điều chỉnh",
        "Quy định này quy định về quản lý kiến trúc",
        "1. Áp dụng cho toàn hệ thống",  # an item's marker and its text on one line
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



# ============================================================ structure
#
# Node sizes in regulation(), rendered with their headings (words):
#   n0 Chương I 62 · n1 Điều 1 23 · n2 khoản 1 7 · n3 Điều 2 34 · n4 khoản 1 9 · n5 khoản 2 19 · n6 điểm a 15
#   (the table renders as a grid, "TT | Nội dung": each "|" counts as a word)


def body_chunks(max_tokens: int, **options: Any) -> list[tuple[tuple[int, int], str | None, tuple[str, ...]]]:
    """(range, owner, context) of each body chunk of regulation().

    context = the headings contextualize() puts in front of the text: the
    breadcrumb minus the owner's heading when the text already starts with it.
    """
    return [
        ((c.start, c.end), c.node_id, context(c))
        for c in chunker(max_tokens, **options).chunk(regulation())
        if c.part == "body"
    ]


def context(chunk: Chunk) -> tuple[str, ...]:
    return tuple(step.label for step in chunk.path if step.start < chunk.start)


CHUONG = "Chương I QUY ĐỊNH CHUNG"
DIEU_1 = "Điều 1. Phạm vi điều chỉnh"
DIEU_2 = "Điều 2. Giải thích từ ngữ"


def test_a_root_that_fits_is_one_chunk() -> None:
    assert body_chunks(62, keep_sections_apart=False) == [((1, 15), "n0", ())]


def test_a_node_holding_sections_is_split_even_when_it_fits() -> None:
    # The whole Chương fits in 62, but its Điều are sections: one chunk each, never one for both.
    assert body_chunks(62) == [((2, 7), "n1", (CHUONG,)), ((7, 15), "n3", (CHUONG,))]


def test_a_root_that_does_not_fit_is_chunked_as_its_children() -> None:
    # The Chương (62) does not fit in 40. Each Điều is a section, so each is chunked on its own, and the
    # Chương's heading, alone, gives no chunk: it is in their breadcrumbs.
    assert body_chunks(40) == [
        ((2, 7), "n1", (CHUONG,)),
        ((7, 15), "n3", (CHUONG,)),
    ]


def test_packing_across_sections_when_they_are_not_kept_apart() -> None:
    # The Chương's heading and Điều 1 fit together (29); adding Điều 2 would not.
    assert body_chunks(40, keep_sections_apart=False) == [
        ((1, 7), "n0", ()),
        ((7, 15), "n3", (CHUONG,)),
    ]


def test_descends_only_where_needed() -> None:
    # In 20, neither Điều fits whole. Inside Điều 1 the lead-in and khoản 1 do not fit together;
    # inside Điều 2 the heading and khoản 1 do (15), khoản 2 goes alone (19).
    assert body_chunks(20) == [
        ((2, 5), "n1", (CHUONG,)),  # Điều 1's heading, its lead-in and the image
        ((5, 7), "n2", (CHUONG, DIEU_1)),
        ((7, 10), "n3", (CHUONG,)),  # Điều 2's heading merged with khoản 1, owned by Điều 2
        ((10, 15), "n5", (CHUONG, DIEU_2)),
    ]


def test_lead_in_is_owned_by_the_parent() -> None:
    (lead_in, *_) = chunker(20).chunk(regulation())[1:]
    assert lead_in.node_id == "n1"
    assert lead_in.text == "Điều 1. Phạm vi điều chỉnh\nQuy định này quy định về quản lý kiến trúc"


def test_a_heading_that_fits_nothing_else_is_only_in_breadcrumbs() -> None:
    # In 12, Điều 2's heading (6) cannot join khoản 1 (9): alone it has no content, so it gives no chunk,
    # and every chunk under Điều 2 carries it in its breadcrumb.
    chunks = chunker(12).chunk(regulation())
    assert all("Điều 2." not in c.text for c in chunks)
    assert all(DIEU_2 in context(c) for c in chunks if c.start >= 7 and c.part == "body")


def test_a_heading_merged_with_its_first_child_stays_in_the_text() -> None:
    chunks = chunker(20).chunk(regulation())
    (merged,) = [c for c in chunks if c.start == 7]
    assert merged.text.splitlines()[:2] == [DIEU_2, "1. Hệ thống là tập hợp các thành phần"]
    assert context(merged) == (CHUONG,)
    assert merged.headings == (CHUONG, DIEU_2)  # the breadcrumb still names the owner


def test_a_leaf_that_does_not_fit_is_split_inside_its_own_range() -> None:
    # Điểm a (15) has no children: split by size, both pieces owned by it.
    assert body_chunks(12)[-2:] == [
        ((12, 14), "n6", (CHUONG, DIEU_2, "2. Kiến trúc gồm:")),  # starts with its own heading "a)"
        ((14, 15), "n6", (CHUONG, DIEU_2, "2. Kiến trúc gồm:", "a) Kiến trúc ứng dụng")),  # the table
    ]


def test_a_heading_split_from_its_text_moves_to_the_breadcrumb() -> None:
    # Khoản 1 of Điều 2 (9) does not fit in 8: "1." alone gives no chunk, the text carries it as a heading.
    assert ((9, 10), "n4", (CHUONG, DIEU_2, "1.")) in body_chunks(8)


def outlined(
    *layout: tuple[str, str | None, int, int], n_segments: int, sections: tuple[str, ...] = ()
) -> LogicalDocument:
    """A document whose nodes are (id, parent, start, end); headings at node starts, two-word text elsewhere.

    Nodes are items unless named in ``sections``. The whole document is body.
    """
    nodes = tuple(
        make_logical_node(id=node_id, parent_id=parent, level="x", is_section=node_id in sections, number=None,
                          marker=f"[{node_id}]", title=None, head_block_ids=(f"h-{node_id}",), start=start, end=end)
        for node_id, parent, start, end in layout
    )
    segments: list[Segment] = []
    for index in range(n_segments):
        starting = [n for n in nodes if n.start == index]
        segments.append(heading_for(starting[0]) if starting else make_text_segment(block_id=f"pb{index}", text=f"w{index} w{index}"))
    return make_logical_document(segments=tuple(segments), nodes=nodes, body_start=0, body_end=n_segments)


def test_text_after_the_last_child_is_owned_by_the_parent() -> None:
    #   0 [a]  1 text  2 [a/1]  3 text  4 text (closing remark of a)
    document = outlined(("a", None, 0, 5), ("a/1", "a", 2, 4), n_segments=5)
    chunks = chunker(4).chunk(document)
    assert [((c.start, c.end), c.node_id, context(c)) for c in chunks] == [
        ((0, 2), "a", ()),
        ((2, 4), "a/1", ("[a] w1 w1",)),  # a's lead-in (segment 1) comes before the chunk
        ((4, 5), "a", ("[a] w1 w1",)),  # after a/1, so a's heading is in the breadcrumb
    ]


def test_text_between_children_is_owned_by_the_parent() -> None:
    #   0 [a]  1 [a/1]  2 text  3 text (between)  4 [a/2]  5 text
    document = outlined(("a", None, 0, 6), ("a/1", "a", 1, 3), ("a/2", "a", 4, 6), n_segments=6)
    assert [((c.start, c.end), c.node_id) for c in chunker(4).chunk(document)] == [
        ((0, 3), "a"),  # a's heading with a/1 (4 words)
        ((3, 4), "a"),  # the text between; adding a/2 would make 5
        ((4, 6), "a/2"),
    ]


def test_body_text_outside_every_root_is_owned_by_nothing() -> None:
    #   0 text  1 [a]  2 text  3 text (after the root)  4 [b]  5 text
    document = outlined(("a", None, 1, 3), ("b", None, 4, 6), n_segments=6)
    assert [((c.start, c.end), c.node_id, context(c)) for c in chunker(3).chunk(document)] == [
        ((0, 1), None, ()),
        ((1, 3), "a", ()),
        ((3, 4), None, ()),
        ((4, 6), "b", ()),
    ]


def test_small_siblings_are_merged_under_their_parent() -> None:
    #   0 [a]  1 [a/1]  2 text  3 [a/2]  4 text  5 [a/3]  6 text      each child: 3 words, a: 10
    document = outlined(("a", None, 0, 7), ("a/1", "a", 1, 3), ("a/2", "a", 3, 5), ("a/3", "a", 5, 7), n_segments=7)
    assert [((c.start, c.end), c.node_id) for c in chunker(7).chunk(document)] == [
        ((0, 5), "a"),  # heading + a/1 + a/2 = 7
        ((5, 7), "a/3"),  # alone: owned by it
    ]


def test_a_group_never_cuts_a_child() -> None:
    # a/2 (5 words) does not fit after [a] + a/1 (4) in 6, so the group closes before it, never inside it.
    document = outlined(("a", None, 0, 6), ("a/1", "a", 1, 3), ("a/2", "a", 3, 6), n_segments=6)
    assert [((c.start, c.end), c.node_id) for c in chunker(6).chunk(document)] == [
        ((0, 3), "a"),
        ((3, 6), "a/2"),
    ]


# ============================================================ structure invariants


@pytest.mark.parametrize("max_tokens", [1, 3, 8, 12, 20, 40, 60])
def test_chunks_never_cut_through_a_node(max_tokens: int) -> None:
    # A node is inside one chunk, or holds the chunk, or is disjoint from it.
    document = regulation()
    for chunk in chunker(max_tokens).chunk(document):
        for node in document.nodes:
            inside = chunk.start <= node.start and node.end <= chunk.end
            holds = node.start <= chunk.start and chunk.end <= node.end
            disjoint = node.end <= chunk.start or chunk.end <= node.start
            assert inside or holds or disjoint, (chunk.id, node.id)


@pytest.mark.parametrize("max_tokens", [1, 3, 8, 12, 20, 40, 60])
def test_owner_is_the_innermost_node_holding_the_chunk(max_tokens: int) -> None:
    document = regulation()
    for chunk in chunker(max_tokens).chunk(document):
        holders = [n for n in document.nodes if n.start <= chunk.start and chunk.end <= n.end]
        expected = holders[-1].id if holders else None  # pre-order: the last holder is the innermost
        assert chunk.node_id == expected, chunk.id


@pytest.mark.parametrize("max_tokens", [3, 8, 12, 20, 25, 40])
def test_contextualized_chunks_fit_when_the_breadcrumb_is_counted(max_tokens: int) -> None:
    for chunk in chunker(max_tokens, include_breadcrumb_in_size=True).chunk(regulation()):
        if chunk.end - chunk.start > 1:
            assert len(chunk.contextualize().split()) <= max_tokens, chunk.id



# ============================================================ sections kept apart
#
#   0 [a]  1 [b]  2 text  3 [c]  4 text  5 text           a: section root; b, c: sections, 3 words each


def sections_document() -> LogicalDocument:
    return outlined(("a", None, 0, 6), ("b", "a", 1, 3), ("c", "a", 3, 6), n_segments=6, sections=("a", "b", "c"))


def test_sections_are_kept_apart_by_default() -> None:
    assert chunker(10).keep_sections_apart is True


def test_small_sections_are_not_merged() -> None:
    # b and c would fit together in 8, but each is a section: one chunk each.
    assert [((c.start, c.end), c.node_id) for c in chunker(8).chunk(sections_document())] == [
        ((1, 3), "b"),
        ((3, 6), "c"),
    ]


def test_small_sections_are_merged_when_not_kept_apart() -> None:
    chunks = chunker(8, keep_sections_apart=False).chunk(sections_document())
    assert [((c.start, c.end), c.node_id) for c in chunks] == [((0, 3), "a"), ((3, 6), "c")]


def test_a_section_of_items_that_fits_is_one_chunk() -> None:
    # b and c are sections holding only text: each fits whole, so each is one chunk.
    assert [((c.start, c.end), c.node_id) for c in chunker(20).chunk(sections_document())] == [
        ((1, 3), "b"),
        ((3, 6), "c"),
    ]


def test_items_still_group_inside_a_section() -> None:
    #   0 [a]  1 [a/1]  2 text  3 [a/2]  4 text  5 [a/3]  6 text     a: section; children: items
    document = outlined(
        ("a", None, 0, 7), ("a/1", "a", 1, 3), ("a/2", "a", 3, 5), ("a/3", "a", 5, 7), n_segments=7, sections=("a",)
    )
    assert [((c.start, c.end), c.node_id) for c in chunker(7).chunk(document)] == [((0, 5), "a"), ((5, 7), "a/3")]


def test_a_lead_in_does_not_join_the_next_section() -> None:
    #   0 [a]  1 text (lead-in)  2 [b]  3 text           a, b: sections
    document = outlined(("a", None, 0, 4), ("b", "a", 2, 4), n_segments=4, sections=("a", "b"))
    assert [((c.start, c.end), c.node_id) for c in chunker(4).chunk(document)] == [((0, 2), "a"), ((2, 4), "b")]


def test_a_section_too_big_alone_is_chunked_inside() -> None:
    #   0 [a]  1 [b]  2 text  3 text  4 text             a, b: sections; b (7 words) does not fit in 4
    document = outlined(("a", None, 0, 5), ("b", "a", 1, 5), n_segments=5, sections=("a", "b"))
    assert [((c.start, c.end), c.node_id) for c in chunker(4).chunk(document)] == [((1, 3), "b"), ((3, 5), "b")]


@pytest.mark.parametrize("keep_sections_apart", [True, False])
@pytest.mark.parametrize("max_tokens", [1, 3, 8, 12, 20, 40, 60])
def test_invariants_hold_with_and_without_keeping_sections_apart(max_tokens: int, keep_sections_apart: bool) -> None:
    document = regulation()
    chunks = chunker(max_tokens, keep_sections_apart=keep_sections_apart).chunk(document)
    for chunk in chunks:
        holders = [n for n in document.nodes if n.start <= chunk.start and chunk.end <= n.end]
        assert chunk.node_id == (holders[-1].id if holders else None), chunk.id
        for node in document.nodes:
            assert (
                (chunk.start <= node.start and node.end <= chunk.end)
                or (node.start <= chunk.start and chunk.end <= node.end)
                or node.end <= chunk.start
                or chunk.end <= node.start
            ), (chunk.id, node.id)


@pytest.mark.parametrize("max_tokens", [1, 8, 20, 40, 60, 1000])
def test_no_chunk_touches_two_sections_of_the_same_parent(max_tokens: int) -> None:
    # Not even when everything fits: a Chương holding Điều always comes out as one chunk per Điều.
    document = regulation()
    for chunk in chunker(max_tokens).chunk(document):
        touched = [n for n in document.nodes if n.is_section and chunk.start < n.end and n.start < chunk.end]
        parents = [n.parent_id for n in touched]
        assert len(parents) == len(set(parents)), (chunk.id, [n.id for n in touched])



# ============================================================ headings are the whole breadcrumb


@pytest.mark.parametrize("max_tokens", [3, 8, 12, 20, 40, 60])
def test_headings_name_every_node_holding_the_chunk(max_tokens: int) -> None:
    document = regulation()
    for chunk in chunker(max_tokens).chunk(document):
        holders = [n for n in document.nodes if n.start <= chunk.start and chunk.end <= n.end]
        assert [step.node_id for step in chunk.path] == [n.id for n in holders], chunk.id
        assert len(chunk.headings) == len(holders)


def test_chunks_inside_one_dieu_share_its_breadcrumb() -> None:
    chunks = [c for c in chunker(12).chunk(regulation()) if c.part == "body" and c.start >= 7]
    assert all(c.headings[:2] == (CHUONG, DIEU_2) for c in chunks)



# ============================================================ token_count


@pytest.mark.parametrize("max_tokens", [3, 12, 60])
def test_token_count_is_what_the_chunker_measured(max_tokens: int) -> None:
    for chunk in chunker(max_tokens).chunk(regulation()):
        assert chunk.token_count == WordTokenizer().count_tokens(chunk.text), chunk.id


@pytest.mark.parametrize("max_tokens", [8, 20, 40])
def test_token_count_includes_the_breadcrumb_when_it_is_counted(max_tokens: int) -> None:
    for chunk in chunker(max_tokens, include_breadcrumb_in_size=True).chunk(regulation()):
        assert chunk.token_count == WordTokenizer().count_tokens(chunk.contextualize()), chunk.id


@pytest.mark.parametrize("max_tokens", [1, 3, 8, 12, 20, 40, 60])
def test_token_count_is_within_budget_except_for_single_segments(max_tokens: int) -> None:
    for chunk in chunker(max_tokens).chunk(regulation()):
        assert chunk.token_count <= max_tokens or chunk.end - chunk.start == 1, chunk.id


def test_a_too_big_segment_reports_its_real_size() -> None:
    (chunk,) = [c for c in chunker(3).chunk(plain("a", "b c d e f", "g")) if c.text == "b c d e f"]
    assert chunk.token_count == 5


# ============================================================ splitting a table that does not fit
#
# A table with a header "STT | Nội dung" (4 words in grid) and body rows "i | từ từ từ" (5 words each).
# With a budget of 15, a piece holds the header and two rows (4 + 5 + 5 = 14).


def big_table(n_body_rows: int = 6, words: int = 3) -> TableInfo:
    cells = [TableCell(row=0, col=0, row_span=1, col_span=1, paragraphs=("STT",)),
             TableCell(row=0, col=1, row_span=1, col_span=1, paragraphs=("Nội dung",))]
    for r in range(1, n_body_rows + 1):
        cells.append(TableCell(row=r, col=0, row_span=1, col_span=1, paragraphs=(str(r),)))
        cells.append(TableCell(row=r, col=1, row_span=1, col_span=1, paragraphs=(" ".join(["từ"] * words),)))
    return TableInfo(n_rows=n_body_rows + 1, n_cols=2, n_header_rows=1, cells=tuple(cells))


def table_document(table: TableInfo, where: str) -> LogicalDocument:
    """The table as the only content of a leaf node ("leaf"), next to a child ("gap"), or in front matter."""
    table_segment = make_table_segment(block_id="pb-table", table=table)
    if where == "front":
        segments: tuple[Segment, ...] = (table_segment, make_text_segment(block_id="pb2", text="Nội dung"))
        return make_logical_document(segments=segments, body_start=1, body_end=2)
    if where == "leaf":
        node = make_logical_node(id="a", level="dieu", marker="Điều 1.", title="Bảng", head_block_ids=("h-a",), start=0, end=2)
        return make_logical_document(segments=(heading_for(node), table_segment), nodes=(node,), body_start=0, body_end=2)
    parent = make_logical_node(id="a", level="dieu", marker="Điều 1.", title="Bảng", head_block_ids=("h-a",), start=0, end=4)
    child = make_logical_node(id="a/1", parent_id="a", level="khoan", is_section=False, marker="1.", title=None,
                              head_block_ids=("h-a1",), start=2, end=4)
    segments = (heading_for(parent), table_segment, heading_for(child), make_text_segment(block_id="pb3", text="Nội dung khoản"))
    return make_logical_document(segments=segments, nodes=(parent, child), body_start=0, body_end=4)


def table_chunks(chunks: tuple[Chunk, ...]) -> list[Chunk]:
    return [c for c in chunks if "pb-table" in c.block_ids]


@pytest.mark.parametrize("where", ["leaf", "gap", "front"])
def test_a_table_that_does_not_fit_is_split_into_row_groups(where: str) -> None:
    pieces = table_chunks(chunker(15).chunk(table_document(big_table(), where)))
    assert [c.table_rows for c in pieces] == [(1, 3), (3, 5), (5, 7)]
    assert all(c.end - c.start == 1 for c in pieces)


def test_every_piece_repeats_the_header_and_stays_within_budget() -> None:
    pieces = table_chunks(chunker(15).chunk(table_document(big_table(), "leaf")))
    assert all(c.text.splitlines()[0] == "STT | Nội dung" for c in pieces)
    assert all(c.token_count <= 15 for c in pieces)
    assert pieces[1].text == "STT | Nội dung\n3 | từ từ từ\n4 | từ từ từ"


def test_row_groups_cover_every_body_row_once_in_order() -> None:
    for max_tokens in (10, 15, 20, 31):
        pieces = table_chunks(chunker(max_tokens).chunk(table_document(big_table(), "leaf")))
        rows = [r for c in pieces for r in range(*(c.table_rows or (0, 0)))]
        assert rows == list(range(1, 7)), max_tokens


def test_a_table_that_fits_is_not_split() -> None:
    (piece,) = table_chunks(chunker(100).chunk(table_document(big_table(), "leaf")))
    assert piece.table_rows is None


def test_a_table_piece_is_owned_by_the_node_holding_the_table() -> None:
    for where, owner in (("leaf", "a"), ("gap", "a")):
        pieces = table_chunks(chunker(15).chunk(table_document(big_table(), where)))
        assert {c.node_id for c in pieces} == {owner}
    front = table_chunks(chunker(15).chunk(table_document(big_table(), "front")))
    assert {(c.part, c.node_id) for c in front} == {("front_matter", None)}


def test_a_row_too_big_on_its_own_is_a_piece_by_itself() -> None:
    table = big_table(n_body_rows=3, words=3)
    cells = list(table.cells)
    cells[5] = TableCell(row=2, col=1, row_span=1, col_span=1, paragraphs=(" ".join(["từ"] * 30),))
    table = TableInfo(n_rows=table.n_rows, n_cols=2, n_header_rows=1, cells=tuple(cells))
    pieces = table_chunks(chunker(15).chunk(table_document(table, "leaf")))
    assert [c.table_rows for c in pieces] == [(1, 2), (2, 3), (3, 4)]
    assert pieces[1].token_count > 15


def test_a_table_of_header_rows_only_stays_whole() -> None:
    table = TableInfo(n_rows=1, n_cols=1, n_header_rows=1,
                      cells=(TableCell(row=0, col=0, row_span=1, col_span=1, paragraphs=(" ".join(["từ"] * 30),)),))
    (piece,) = table_chunks(chunker(15).chunk(table_document(table, "leaf")))
    assert piece.table_rows is None


def test_key_value_pieces() -> None:
    # Each key_value line is 7 words and there is no header line: two rows per piece in a budget of 20.
    pieces = table_chunks(chunker(20, table_layout="key_value").chunk(table_document(big_table(), "leaf")))
    assert [c.table_rows for c in pieces] == [(1, 3), (3, 5), (5, 7)]
    assert pieces[0].text == "STT: 1; Nội dung: từ từ từ\nSTT: 2; Nội dung: từ từ từ"
    assert all(c.token_count <= 20 for c in pieces)


def test_split_table_pieces_have_their_own_ids_in_order() -> None:
    chunks = chunker(15).chunk(table_document(big_table(), "gap"))
    assert [c.id for c in chunks] == [f"c{i}" for i in range(len(chunks))]
    starts = [(c.start, (c.table_rows or (0, 0))[0]) for c in chunks]
    assert starts == sorted(starts)


def test_token_count_of_a_piece_counts_only_its_rows() -> None:
    pieces = table_chunks(chunker(15).chunk(table_document(big_table(), "leaf")))
    assert all(c.token_count == WordTokenizer().count_tokens(c.text) for c in pieces)


def test_a_merge_across_a_split_is_printed_again_in_the_next_piece() -> None:
    cells = [TableCell(row=0, col=0, row_span=1, col_span=1, paragraphs=("Nhóm",)),
             TableCell(row=0, col=1, row_span=1, col_span=1, paragraphs=("Nội dung",)),
             TableCell(row=1, col=0, row_span=4, col_span=1, paragraphs=("Nhóm A",))]
    for r in range(1, 5):
        cells.append(TableCell(row=r, col=1, row_span=1, col_span=1, paragraphs=(" ".join(["từ"] * 3),)))
    table = TableInfo(n_rows=5, n_cols=2, n_header_rows=1, cells=tuple(cells))
    pieces = table_chunks(chunker(15).chunk(table_document(table, "leaf")))
    assert len(pieces) > 1
    assert all("Nhóm A" in c.text for c in pieces)


def test_a_table_in_end_matter_is_split() -> None:
    table_segment = make_table_segment(block_id="pb-table", table=big_table())
    segments: tuple[Segment, ...] = (make_text_segment(block_id="pb1", text="Nội dung"), table_segment)
    document = make_logical_document(segments=segments, body_start=0, body_end=1)
    pieces = table_chunks(chunker(15).chunk(document))
    assert [(c.part, c.table_rows) for c in pieces] == [("end_matter", (1, 3)), ("end_matter", (3, 5)), ("end_matter", (5, 7))]


def test_text_around_a_split_table_is_chunked_as_before() -> None:
    #   0 [Điều 1.]  1 paragraph  2 table (too big)  3 paragraph
    node = make_logical_node(id="a", level="dieu", marker="Điều 1.", title=None, head_block_ids=("h-a",), start=0, end=4)
    segments: tuple[Segment, ...] = (
        heading_for(node),
        make_text_segment(block_id="pb1", text="Đoạn trước bảng"),
        make_table_segment(block_id="pb-table", table=big_table()),
        make_text_segment(block_id="pb3", text="Đoạn sau bảng"),
    )
    document = make_logical_document(segments=segments, nodes=(node,), body_start=0, body_end=4)
    chunks = chunker(15).chunk(document)
    assert [(c.start, c.end, c.table_rows) for c in chunks] == [
        (0, 2, None),  # the heading and the paragraph before the table, together
        (2, 3, (1, 3)),
        (2, 3, (3, 5)),
        (2, 3, (5, 7)),
        (3, 4, None),
    ]
    assert chunks[0].text == "Điều 1.\nĐoạn trước bảng"
    assert chunks[-1].text == "Đoạn sau bảng"


def test_a_table_without_header_rows_is_split_without_a_header() -> None:
    # Header detection is the adapter's job: with n_header_rows == 0 every row is a body row.
    table = big_table()
    table = TableInfo(n_rows=table.n_rows, n_cols=table.n_cols, n_header_rows=0, cells=table.cells)
    pieces = table_chunks(chunker(15).chunk(table_document(table, "leaf")))
    assert pieces[0].table_rows is not None and pieces[0].table_rows[0] == 0
    assert pieces[0].text.splitlines()[0] == "STT | Nội dung"
    assert all("STT | Nội dung" not in c.text for c in pieces[1:])


# ============================================================ breadcrumb lead-ins
#
# A node without a title gets its first segment after the heading, when that is
# a TextSegment inside the node and before the chunk, cut to
# breadcrumb_lead_in_words words (default 40; "…" when cut). Nodes with a title keep it.


def lead_in_document() -> LogicalDocument:
    """
        0  # Điều 1. Phạm vi                 n0 dieu   [0, 7)
        1  # 1.                              n1 khoan  [1, 7)
        2  Các hệ thống phải đáp ứng các yêu cầu sau đây về an toàn thông tin:   (16 words)
        3  # a)                              n2 diem   [3, 7)
        4  Về lưu trữ:
        5  # -                               n3 gach   [5, 7)
        6  Dữ liệu được mã hóa
    """
    return (
        Blocks()
        .text("Điều 1. Phạm vi")
        .text("1. Các hệ thống phải đáp ứng các yêu cầu sau đây về an toàn thông tin:")
        .text("a) Về lưu trữ:")
        .text("- Dữ liệu được mã hóa")
        .build()
    )


def test_default_cap_is_40_words() -> None:
    # The real regulation: 93 lead-ins of nodes with children, median 12 words, p90 43; 40 cuts 10.
    assert chunker(10).breadcrumb_lead_in_words == 40


def test_negative_cap_rejected() -> None:
    with pytest.raises(ValueError, match="breadcrumb_lead_in_words >= 0, got breadcrumb_lead_in_words=-1"):
        chunker(10, breadcrumb_lead_in_words=-1)


def test_deep_chunk_reads_its_ancestors_lead_ins() -> None:
    document = lead_in_document()
    path = _path_to(document, "n3", start=6, lead_in_words=25)
    assert [step.label for step in path] == [
        "Điều 1. Phạm vi",
        "1. Các hệ thống phải đáp ứng các yêu cầu sau đây về an toàn thông tin:",
        "a) Về lưu trữ:",
        "-",  # the owner: its words are the chunk's own text (segment 6), so no lead-in
    ]


def test_titled_nodes_keep_their_title() -> None:
    path = _path_to(lead_in_document(), "n3", start=6, lead_in_words=25)
    assert (path[0].title, path[0].lead_in) == ("Phạm vi", None)


def test_lead_in_is_capped_with_an_ellipsis() -> None:
    path = _path_to(lead_in_document(), "n3", start=6, lead_in_words=5)
    assert path[1].lead_in == "Các hệ thống phải đáp…"
    assert path[1].label == "1. Các hệ thống phải đáp…"


def test_lead_in_of_exactly_the_cap_has_no_ellipsis() -> None:
    path = _path_to(lead_in_document(), "n3", start=6, lead_in_words=16)
    assert path[1].lead_in == "Các hệ thống phải đáp ứng các yêu cầu sau đây về an toàn thông tin:"


def test_lead_in_whitespace_is_collapsed() -> None:
    document = Blocks().text("Điều 1. Phạm vi").text("1. Hệ thống   gồm\tcác  phần").text("a) Một").build()
    path = _path_to(document, "n2", start=len(document.segments), lead_in_words=25)
    assert path[1].lead_in == "Hệ thống gồm các phần"


def test_zero_turns_lead_ins_off() -> None:
    path = _path_to(lead_in_document(), "n3", start=6, lead_in_words=0)
    assert [step.lead_in for step in path] == [None, None, None, None]
    assert [step.label for step in path][1:] == ["1.", "a)", "-"]


def test_no_lead_in_when_the_chunk_starts_at_it() -> None:
    # The chunk's text begins with khoản 1's words: the breadcrumb must not repeat them.
    path = _path_to(lead_in_document(), "n1", start=2, lead_in_words=25)
    assert path[-1].lead_in is None


def test_no_lead_in_when_the_chunk_starts_before_it() -> None:
    path = _path_to(lead_in_document(), "n1", start=1, lead_in_words=25)
    assert path[-1].lead_in is None


def test_regulation_chunk_starting_at_its_lead_in_keeps_the_bare_marker() -> None:
    # At 8 words khoản 1 of Điều 2's heading (8) is split from its text (9).
    (chunk,) = [c for c in chunker(8).chunk(regulation()) if (c.start, c.end) == (9, 10)]
    assert chunk.path[-1].lead_in is None
    assert chunk.headings[-1] == "1."
    assert chunk.contextualize().count("Hệ thống là tập hợp các thành phần") == 1


def test_no_lead_in_when_the_first_segment_is_an_image() -> None:
    document = Blocks().text("Điều 1. Phạm vi").text("1.").image().text("Nội dung").build()
    khoan = next(node for node in document.nodes if node.level == "khoan")
    path = _path_to(document, khoan.id, start=len(document.segments), lead_in_words=25)
    assert path[-1].lead_in is None


def test_no_lead_in_when_the_first_segment_is_a_table() -> None:
    document = Blocks().text("Điều 1. Phạm vi").text("1.").table().text("Nội dung").build()
    khoan = next(node for node in document.nodes if node.level == "khoan")
    path = _path_to(document, khoan.id, start=len(document.segments), lead_in_words=25)
    assert path[-1].lead_in is None


def test_no_lead_in_for_a_node_that_is_only_its_heading() -> None:
    document = Blocks().text("Điều 1. Phạm vi").text("1.").text("2. Nội dung").build()
    first_khoan = next(node for node in document.nodes if node.level == "khoan")
    assert first_khoan.end == first_khoan.start + 1
    path = _path_to(document, first_khoan.id, start=len(document.segments), lead_in_words=25)
    assert path[-1].lead_in is None


def test_lead_in_from_the_next_paragraph_when_the_marker_stands_alone() -> None:
    # "1." alone on its line, its words in the next paragraph: still the node's opening words.
    document = Blocks().text("Điều 1. Phạm vi").text("1.").text("Nội dung khoản một").text("2. Khác").build()
    first_khoan = next(node for node in document.nodes if node.level == "khoan")
    path = _path_to(document, first_khoan.id, start=len(document.segments), lead_in_words=25)
    assert path[-1].label == "1. Nội dung khoản một"


@pytest.mark.parametrize("max_tokens", [3, 8, 12, 60])
def test_no_chunk_has_a_title_and_a_lead_in(max_tokens: int) -> None:
    for chunk in chunker(max_tokens).chunk(regulation()):
        assert all(step.title is None or step.lead_in is None for step in chunk.path)


@pytest.mark.parametrize("max_tokens", [3, 8, 12, 60])
def test_no_lead_in_repeats_text_inside_its_chunk(max_tokens: int) -> None:
    document = regulation()
    for chunk in chunker(max_tokens).chunk(document):
        for step in chunk.path:
            if step.lead_in is not None:
                assert step.start + 1 < chunk.start


def test_counted_breadcrumb_includes_the_lead_ins() -> None:
    document = lead_in_document()
    piece = _Piece(part="body", owner_id="n3", start=6, end=7)
    with_lead_ins = _ChunkingRun(chunker(200, include_breadcrumb_in_size=True), document)._count_piece_tokens(piece)
    without = _ChunkingRun(
        chunker(200, include_breadcrumb_in_size=True, breadcrumb_lead_in_words=0), document
    )._count_piece_tokens(piece)
    # 16 words of khoản 1 and 3 of điểm a; "Dữ liệu được mã hóa" is the chunk's own text, not a lead-in.
    assert with_lead_ins - without == 16 + 3


def test_lead_ins_are_not_counted_by_default() -> None:
    document = lead_in_document()
    piece = _Piece(part="body", owner_id="n3", start=6, end=7)
    assert _ChunkingRun(chunker(200), document)._count_piece_tokens(piece) == 5