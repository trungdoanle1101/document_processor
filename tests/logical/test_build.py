"""Tests for logical/build.py: assembling a LogicalDocument from classified blocks.

The builder never decides what a block is (passes 1-3 did); it turns the
decisions into segments and nodes. Each block gives one segment, except a
heading (its rest of text, if any, per the level's title rule) and a title
block consumed by a next_block heading (none). A node covers [start, end) of
the segments, including its descendants. LogicalDocument itself re-checks the
outline, so a wrong parent or range fails loudly here too.
"""

from typing import Any

import pytest

from document_processor.core.identity import compute_hash
from document_processor.core.models import (
    BlockType,
    ImageSegment,
    LogicalDocument,
    TableSegment,
    TextSegment,
)
from document_processor.logical.build import assemble_document, build_logical_document
from document_processor.logical.candidates import classify_blocks
from document_processor.logical.levels import GlyphMarkerSpec, LevelSpec, TemplateMarkerSpec
from document_processor.logical.profile import Profile
from document_processor.logical.profiles.vn_regulation import VN_REGULATION
from tests.factories import (
    make_block,
    make_declared_numbering,
    make_document,
    make_image_info,
    make_list_info,
    make_table_block,
    make_table_cell,
    make_table_info,
)


def templates(*items: str) -> TemplateMarkerSpec:
    return TemplateMarkerSpec(templates=items)


PROFILE = Profile(
    name="test",
    version="1",
    levels=(
        LevelSpec(name="mo_dau", marker=templates("MỞ ĐẦU"), title="none"),  # a word-only marker
        LevelSpec(name="chuong", marker=templates("Chương {upper_roman}"), title="next_block"),
        LevelSpec(name="dieu", marker=templates("Điều {arabic}."), title="rest_of_paragraph"),
        LevelSpec(name="khoan", marker=templates("{arabic}."), inside=("dieu",)),
        LevelSpec(name="diem", marker=templates("{lower_latin})"), inside=("khoan",)),
        LevelSpec(name="gach", marker=GlyphMarkerSpec(glyphs=("dash",)), closes_on_plain_paragraph=True),
    ),
    end_matter=templates("Nơi nhận:"),
)


class Blocks:
    """Builds blocks with unique ids pb1, pb2, … in the order they are made."""

    def __init__(self) -> None:
        self.count = 0

    def _id(self) -> str:
        self.count += 1
        return f"pb{self.count}"

    def text(self, text: str) -> Any:
        return make_block(id=self._id(), text=text)

    def numbered(self, text: str, displayed: str) -> Any:
        info = make_list_info(displayed_marker=displayed, is_ordered=True)
        return make_block(id=self._id(), block_type=BlockType.LIST_ITEM, text=text, list_info=info)

    def table(self, text: str = "Nội dung") -> Any:
        info = make_table_info(n_rows=1, n_cols=1, n_header_rows=0, cells=(make_table_cell(paragraphs=(text,)),))
        return make_table_block(id=self._id(), table_info=info)

    def image(self) -> Any:
        return make_block(id=self._id(), block_type=BlockType.IMAGE, text="", image_info=make_image_info())


def build(*blocks: Any, profile: Profile = PROFILE) -> LogicalDocument:
    return build_logical_document(profile, make_document(blocks=blocks))


def outline(document: LogicalDocument) -> list[tuple[str, str | None, str, str | None, str | None, str | None, int, int]]:
    """(id, parent_id, level, number, marker, title, start, end) per node, in pre-order."""
    return [(n.id, n.parent_id, n.level, n.number, n.marker, n.title, n.start, n.end) for n in document.nodes]


def texts(document: LogicalDocument) -> list[str]:
    """Each segment as text: its text, or [table] / [image]."""
    out = []
    for segment in document.segments:
        if isinstance(segment, TextSegment):
            out.append(segment.text)
        elif isinstance(segment, TableSegment):
            out.append("[table]")
        else:
            out.append("[image]")
    return out


# ============================================================ no structure, front matter, document fields


def test_no_heading_everything_is_body_text() -> None:
    b = Blocks()
    document = build(b.text("Một"), b.text("Hai"))
    assert texts(document) == ["Một", "Hai"]
    assert document.nodes == ()
    assert (document.body_start, document.body_end) == (0, 2)


def test_front_matter_is_everything_before_the_first_heading() -> None:
    b = Blocks()
    document = build(b.text("NGÂN HÀNG"), b.text("QUY ĐỊNH"), b.text("Điều 1. Phạm vi"), b.text("Nội dung"))
    assert texts(document) == ["NGÂN HÀNG", "QUY ĐỊNH", "Nội dung"]
    assert document.body_start == 2
    assert document.body_end == 3


def test_document_fields() -> None:
    b = Blocks()
    physical = make_document(blocks=(b.text("Điều 1. X"),))
    document = build_logical_document(PROFILE, physical)
    assert document.physical_document_id == physical.id
    assert document.source_file_hash == physical.file_hash
    assert (document.profile_name, document.profile_version) == ("test", "1")


def test_document_id_is_a_hash_of_document_and_profile() -> None:
    b = Blocks()
    physical = make_document(blocks=(b.text("Điều 1. X"),))
    first = build_logical_document(PROFILE, physical)
    again = build_logical_document(PROFILE, physical)
    assert first.id == again.id
    assert first.id.startswith("sha256:")
    other_version = Profile(name="test", version="2", levels=PROFILE.levels, end_matter=PROFILE.end_matter)
    assert build_logical_document(other_version, physical).id != first.id
    assert first.id == compute_hash(f"{physical.id}|test|1".encode())


def test_node_ids_are_pre_order_positions() -> None:
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.text("1. X"), b.text("2. Y"), b.text("Điều 2. B"))
    assert [n.id for n in document.nodes] == ["n0", "n1", "n2", "n3"]


# ============================================================ heading parts and title rules


def test_rest_of_paragraph_is_the_title() -> None:
    b = Blocks()
    document = build(b.text("Điều 9. Phạm vi điều chỉnh"))
    assert outline(document) == [("n0", None, "dieu", "9", "Điều 9.", "Phạm vi điều chỉnh", 0, 0)]
    assert texts(document) == []  # the heading block gives no segment


def test_rest_of_paragraph_with_nothing_after_the_marker_has_no_title() -> None:
    b = Blocks()
    document = build(b.text("Điều 9."), b.text("Nội dung"))
    assert outline(document)[0][5] is None
    assert texts(document) == ["Nội dung"]  # no fallback to the next block


def test_title_none_keeps_the_rest_as_the_first_body_segment() -> None:
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.text("2. Nội dung khoản"))
    assert outline(document)[1] == ("n1", "n0", "khoan", "2", "2.", None, 0, 1)
    assert texts(document) == ["Nội dung khoản"]


def test_generated_marker_keeps_the_whole_text() -> None:
    # Word shows "1."; the text has no marker of its own, so all of it is body.
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.numbered("Quy định này quy định về", "1."))
    assert outline(document)[1] == ("n1", "n0", "khoan", "1", "1.", None, 0, 1)
    assert texts(document) == ["Quy định này quy định về"]


def test_generated_and_typed_marker_drop_the_typed_one() -> None:
    # Word shows "1." and the author typed "1." too: the body starts after the typed marker.
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.numbered("1. Nội dung", "1."))
    assert outline(document)[1][4] == "1."
    assert texts(document) == ["Nội dung"]


def test_multi_numeral_number_is_written_with_its_separators() -> None:
    sub = LevelSpec(name="sub", marker=templates("{arabic}.{arabic}.{arabic}"))
    profile = Profile(name="sub", version="1", levels=(sub,), end_matter=None)
    b = Blocks()
    document = build(b.text("2.3.1 Tầng xử lý"), profile=profile)
    assert outline(document)[0][3:5] == ("2.3.1", "2.3.1")


def test_word_only_marker_has_no_number() -> None:
    b = Blocks()
    document = build(b.text("MỞ ĐẦU"), b.text("Nội dung"))
    assert outline(document)[0][2:6] == ("mo_dau", None, "MỞ ĐẦU", None)


def test_glyph_marker_has_no_number() -> None:
    b = Blocks()
    document = build(b.text("- Nội dung gạch"))
    assert outline(document) == [("n0", None, "gach", None, "-", None, 0, 1)]
    assert texts(document) == ["Nội dung gạch"]


# ---------- next_block


def test_next_block_takes_the_following_text_as_title() -> None:
    b = Blocks()
    chuong, title, dieu = b.text("Chương I"), b.text("QUY ĐỊNH CHUNG"), b.text("Điều 1. Phạm vi")
    document = build(chuong, title, dieu, b.text("Nội dung"))
    assert outline(document)[0] == ("n0", None, "chuong", "I", "Chương I", "QUY ĐỊNH CHUNG", 0, 1)
    assert document.nodes[0].head_block_ids == (chuong.id, title.id)
    assert texts(document) == ["Nội dung"]  # neither the marker nor the title block is a segment


def test_next_block_uses_the_rest_of_its_own_line_first() -> None:
    b = Blocks()
    document = build(b.text("Chương I QUY ĐỊNH CHUNG"), b.text("Đoạn đầu chương"))
    assert outline(document)[0][5] == "QUY ĐỊNH CHUNG"
    assert texts(document) == ["Đoạn đầu chương"]  # not consumed


def test_next_block_followed_by_a_heading_has_no_title() -> None:
    b = Blocks()
    document = build(b.text("Chương I"), b.text("Điều 1. Phạm vi"))
    assert [n.title for n in document.nodes] == [None, "Phạm vi"]


def test_next_block_followed_by_a_table_has_no_title() -> None:
    b = Blocks()
    document = build(b.text("Chương I"), b.table(), b.text("Đoạn"))
    assert outline(document)[0][5] is None
    assert texts(document) == ["[table]", "Đoạn"]


def test_only_the_block_right_after_can_be_the_title() -> None:
    b = Blocks()
    document = build(b.text("Chương I"), b.image(), b.text("QUY ĐỊNH CHUNG"))
    assert outline(document)[0][5] is None
    assert texts(document) == ["[image]", "QUY ĐỊNH CHUNG"]


# ============================================================ tree shape


def test_parents_and_siblings() -> None:
    b = Blocks()
    document = build(
        b.text("Chương I"),
        b.text("QUY ĐỊNH CHUNG"),
        b.text("Điều 1. A"),
        b.text("1. X"),
        b.text("a) điểm"),
        b.text("2. Y"),
        b.text("Điều 2. B"),
        b.text("Chương II"),
        b.text("TỔ CHỨC THỰC HIỆN"),
    )
    assert [(n.id, n.parent_id, n.level) for n in document.nodes] == [
        ("n0", None, "chuong"),
        ("n1", "n0", "dieu"),
        ("n2", "n1", "khoan"),
        ("n3", "n2", "diem"),
        ("n4", "n1", "khoan"),  # a sibling of khoản 1, not its child
        ("n5", "n0", "dieu"),
        ("n6", None, "chuong"),
    ]
    assert document.root_ids == ("n0", "n6")


def test_ranges_include_descendants() -> None:
    b = Blocks()
    document = build(
        b.text("Điều 1. A"), b.text("1. X"), b.text("a) x1"), b.text("a2"), b.text("2. Y"), b.text("Điều 2. B"), b.text("Z")
    )
    assert texts(document) == ["X", "x1", "a2", "Y", "Z"]
    assert [(n.level, n.start, n.end) for n in document.nodes] == [
        ("dieu", 0, 4),
        ("khoan", 0, 3),
        ("diem", 1, 3),
        ("khoan", 3, 4),
        ("dieu", 4, 5),
    ]


def test_empty_nodes_at_the_start() -> None:
    # Headings with nothing under them (as in a typed table of contents) give empty ranges, even at 0.
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.text("Điều 2. B"), b.text("Nội dung"))
    assert [(n.start, n.end) for n in document.nodes] == [(0, 0), (0, 1)]


def test_a_plain_paragraph_goes_to_the_parent_of_closing_items() -> None:
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.text("1. X"), b.text("- gạch"), b.text("Đoạn kết khoản"))
    assert [(n.level, n.start, n.end) for n in document.nodes] == [
        ("dieu", 0, 3),
        ("khoan", 0, 3),  # the closing remark is inside the khoản
        ("gach", 1, 2),  # but not inside the dash item
    ]


def test_tables_and_images_stay_inside_the_open_node() -> None:
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.text("- gạch"), b.table(), b.image(), b.text("Đoạn"))
    assert texts(document) == ["gạch", "[table]", "[image]", "Đoạn"]
    assert [(n.level, n.start, n.end) for n in document.nodes] == [("dieu", 0, 4), ("gach", 0, 3)]
    assert isinstance(document.segments[1], TableSegment)
    assert isinstance(document.segments[2], ImageSegment)


# ============================================================ end matter


def test_end_matter_closes_everything_and_keeps_its_segments() -> None:
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.text("1. X"), b.text("Nơi nhận:"), b.text("- Như Điều 3;"))
    assert texts(document) == ["X", "Nơi nhận:", "- Như Điều 3;"]
    assert (document.body_start, document.body_end) == (0, 1)
    assert [(n.level, n.end) for n in document.nodes] == [("dieu", 1), ("khoan", 1)]


def test_end_matter_in_a_table() -> None:
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.text("Nội dung"), b.table("Nơi nhận: - Như Điều 26; - Lưu: VT"))
    assert (document.body_start, document.body_end) == (0, 1)
    assert texts(document) == ["Nội dung", "[table]"]


def test_end_matter_right_after_an_empty_heading() -> None:
    # The body is empty: it starts and ends at segment 0.
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.text("Nơi nhận:"))
    assert (document.body_start, document.body_end) == (0, 0)
    assert texts(document) == ["Nơi nhận:"]


# ============================================================ assemble_document


def test_assemble_refuses_blocks_from_another_document() -> None:
    b = Blocks()
    blocks = (b.text("Điều 1. A"), b.text("Nội dung"))
    classified = classify_blocks(PROFILE, blocks[:1])
    with pytest.raises(ValueError):
        assemble_document(PROFILE, make_document(blocks=blocks), classified)


def test_build_is_classify_then_assemble() -> None:
    b = Blocks()
    physical = make_document(blocks=(b.text("Điều 1. A"), b.text("1. X"), b.text("Nội dung")))
    assert build_logical_document(PROFILE, physical) == assemble_document(
        PROFILE, physical, classify_blocks(PROFILE, physical.blocks)
    )


# ============================================================ VN_REGULATION, a stretch of the real document


def test_vn_regulation_real_stretch() -> None:
    none_format = make_declared_numbering(num_fmt="none", segment_formats=())
    b = Blocks()

    def heading_list(text: str, level: int) -> Any:
        info = make_list_info(displayed_marker=None, is_ordered=False, nested_level=level, declared_numbering=none_format)
        return make_block(id=b._id(), block_type=BlockType.LIST_ITEM, text=text, list_info=info)

    def decimal(text: str, displayed: str) -> Any:
        info = make_list_info(
            displayed_marker=displayed,
            is_ordered=True,
            declared_numbering=make_declared_numbering(num_fmt="decimal", segment_formats=("decimal",)),
        )
        return make_block(id=b._id(), block_type=BlockType.LIST_ITEM, text=text, list_info=info)

    blocks = (
        b.text("Theo đề nghị của Giám đốc Ban Công nghệ,"),
        heading_list("Chương I", 6),
        heading_list("QUY ĐỊNH CHUNG", 6),
        heading_list("Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng", 0),
        decimal("Quy định này quy định về việc quản lý kiến trúc tổng thể CNTT", "1."),
        decimal("Quy định này được áp dụng cho toàn hệ thống", "2."),
        heading_list("Điều 2. Giải thích từ ngữ", 0),
        decimal("“Cloud native”: Là một cách tiếp cận", "1."),
        b.text("- Kiến trúc tổng thể CNTT bao gồm các kiến trúc thành phần sau:"),
        b.text("+ Kiến trúc ứng dụng"),
        b.text("Các thuật ngữ khác được hiểu theo quy định."),
        b.table("Nơi nhận: - Như Điều 26; - Lưu: VT"),
    )
    document = build(*blocks, profile=VN_REGULATION)
    assert texts(document) == [
        "Theo đề nghị của Giám đốc Ban Công nghệ,",
        "Quy định này quy định về việc quản lý kiến trúc tổng thể CNTT",
        "Quy định này được áp dụng cho toàn hệ thống",
        "“Cloud native”: Là một cách tiếp cận",
        "Kiến trúc tổng thể CNTT bao gồm các kiến trúc thành phần sau:",
        "Kiến trúc ứng dụng",
        "Các thuật ngữ khác được hiểu theo quy định.",
        "[table]",
    ]
    assert (document.body_start, document.body_end) == (1, 7)
    assert outline(document) == [
        ("n0", None, "chuong", "I", "Chương I", "QUY ĐỊNH CHUNG", 1, 7),
        ("n1", "n0", "dieu", "1", "Điều 1.", "Phạm vi điều chỉnh và đối tượng áp dụng", 1, 3),
        ("n2", "n1", "khoan", "1", "1.", None, 1, 2),
        ("n3", "n1", "khoan", "2", "2.", None, 2, 3),
        ("n4", "n0", "dieu", "2", "Điều 2.", "Giải thích từ ngữ", 3, 7),
        ("n5", "n4", "khoan", "1", "1.", None, 3, 7),
        ("n6", "n5", "gach", None, "-", None, 4, 6),  # "+" has no `inside`, so it nests under the dash
        ("n7", "n6", "cong", None, "+", None, 5, 6),
    ]