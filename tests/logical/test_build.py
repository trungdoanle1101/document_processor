"""Tests for logical/build.py: assembling a LogicalDocument from classified blocks.

The builder never decides what a block is (passes 1-3 did); it turns the
decisions into segments and nodes. A heading gives a HeadingSegment (marker and
title) and then its rest of text, if the level's title rule leaves any; a title
block consumed by a next_block heading gives no segment of its own; every
other block gives one segment. A node covers [start, end) of the segments,
starting with its own heading and including its descendants. LogicalDocument
itself re-checks the outline and the headings, so a wrong parent, range or
heading fails loudly here too.

In the expected texts, "# …" is a heading segment.
"""

from typing import Any

import pytest

from document_processor.core.identity import compute_hash
from document_processor.core.models import (
    BlockType,
    HeadingSegment,
    ImageSegment,
    LogicalDocument,
    TableSegment,
    TextSegment,
)
from document_processor.logical.build import assemble_document, build_logical_document
from document_processor.logical.candidates import classify_blocks
from document_processor.logical.levels import (
    GlyphMarkerSpec,
    LevelSpec,
    TemplateMarkerSpec,
)
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
        LevelSpec(
            name="mo_dau", is_section=True, marker=templates("MỞ ĐẦU"), title="none"
        ),  # a word-only marker
        LevelSpec(
            name="chuong",
            is_section=True,
            marker=templates("Chương {upper_roman}"),
            title="next_block",
        ),
        LevelSpec(
            name="dieu",
            is_section=True,
            marker=templates("Điều {arabic}."),
            title="rest_of_paragraph",
        ),
        LevelSpec(
            name="khoan",
            is_section=False,
            marker=templates("{arabic}."),
            inside=("dieu",),
        ),
        LevelSpec(
            name="diem",
            is_section=False,
            marker=templates("{lower_latin})"),
            inside=("khoan",),
        ),
        LevelSpec(
            name="gach",
            is_section=False,
            marker=GlyphMarkerSpec(glyphs=("dash",)),
            closes_on_plain_paragraph=True,
        ),
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
        return make_block(
            id=self._id(), block_type=BlockType.LIST_ITEM, text=text, list_info=info
        )

    def table(self, text: str = "Nội dung") -> Any:
        info = make_table_info(
            n_rows=1,
            n_cols=1,
            n_header_rows=0,
            cells=(make_table_cell(paragraphs=(text,)),),
        )
        return make_table_block(id=self._id(), table_info=info)

    def image(self) -> Any:
        return make_block(
            id=self._id(),
            block_type=BlockType.IMAGE,
            text="",
            image_info=make_image_info(),
        )


def build(*blocks: Any, profile: Profile = PROFILE) -> LogicalDocument:
    return build_logical_document(profile, make_document(blocks=blocks))


def outline(
    document: LogicalDocument,
) -> list[tuple[str, str | None, str, str | None, str | None, str | None, int, int]]:
    """(id, parent_id, level, number, marker, title, start, end) per node, in pre-order."""
    return [
        (n.id, n.parent_id, n.level, n.number, n.marker, n.title, n.start, n.end)
        for n in document.nodes
    ]


def texts(document: LogicalDocument) -> list[str]:
    """Each segment as text: "# marker title" for a heading, its text, or [table] / [image]."""
    out = []
    for segment in document.segments:
        if isinstance(segment, HeadingSegment):
            out.append(
                "# "
                + " ".join(part for part in (segment.marker, segment.title) if part)
            )
        elif isinstance(segment, TextSegment):
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
    document = build(
        b.text("NGÂN HÀNG"),
        b.text("QUY ĐỊNH"),
        b.text("Điều 1. Phạm vi"),
        b.text("Nội dung"),
    )
    assert texts(document) == ["NGÂN HÀNG", "QUY ĐỊNH", "# Điều 1. Phạm vi", "Nội dung"]
    assert document.body_start == 2  # the body starts with the first heading
    assert document.body_end == 4


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
    other_version = Profile(
        name="test", version="2", levels=PROFILE.levels, end_matter=PROFILE.end_matter
    )
    assert build_logical_document(other_version, physical).id != first.id
    assert first.id == compute_hash(f"{physical.id}|test|1".encode())


def test_node_ids_are_pre_order_positions() -> None:
    b = Blocks()
    document = build(
        b.text("Điều 1. A"), b.text("1. X"), b.text("2. Y"), b.text("Điều 2. B")
    )
    assert [n.id for n in document.nodes] == ["n0", "n1", "n2", "n3"]


# ============================================================ heading parts and title rules


def test_rest_of_paragraph_is_the_title() -> None:
    b = Blocks()
    document = build(b.text("Điều 9. Phạm vi điều chỉnh"))
    assert outline(document) == [
        ("n0", None, "dieu", "9", "Điều 9.", "Phạm vi điều chỉnh", 0, 1)
    ]
    assert texts(document) == [
        "# Điều 9. Phạm vi điều chỉnh"
    ]  # the heading only, no body segment


def test_rest_of_paragraph_with_nothing_after_the_marker_has_no_title() -> None:
    b = Blocks()
    document = build(b.text("Điều 9."), b.text("Nội dung"))
    assert outline(document)[0][5] is None
    assert texts(document) == ["# Điều 9.", "Nội dung"]  # no fallback to the next block


def test_title_none_keeps_the_rest_as_the_first_body_segment() -> None:
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.text("2. Nội dung khoản"))
    assert outline(document)[1] == ("n1", "n0", "khoan", "2", "2.", None, 1, 3)
    assert texts(document) == ["# Điều 1. A", "# 2.", "Nội dung khoản"]


def test_generated_marker_keeps_the_whole_text() -> None:
    # Word shows "1."; the text has no marker of its own, so all of it is body.
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.numbered("Quy định này quy định về", "1."))
    assert outline(document)[1] == ("n1", "n0", "khoan", "1", "1.", None, 1, 3)
    assert texts(document) == ["# Điều 1. A", "# 1.", "Quy định này quy định về"]


def test_generated_and_typed_marker_drop_the_typed_one() -> None:
    # Word shows "1." and the author typed "1." too: the body starts after the typed marker.
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.numbered("1. Nội dung", "1."))
    assert outline(document)[1][4] == "1."
    assert texts(document) == ["# Điều 1. A", "# 1.", "Nội dung"]


def test_multi_numeral_number_is_written_with_its_separators() -> None:
    sub = LevelSpec(
        name="sub", is_section=False, marker=templates("{arabic}.{arabic}.{arabic}")
    )
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
    assert outline(document) == [("n0", None, "gach", None, "-", None, 0, 2)]
    assert texts(document) == ["# -", "Nội dung gạch"]


# ---------- next_block


def test_next_block_takes_the_following_text_as_title() -> None:
    b = Blocks()
    chuong, title, dieu = (
        b.text("Chương I"),
        b.text("QUY ĐỊNH CHUNG"),
        b.text("Điều 1. Phạm vi"),
    )
    document = build(chuong, title, dieu, b.text("Nội dung"))
    assert outline(document)[0] == (
        "n0",
        None,
        "chuong",
        "I",
        "Chương I",
        "QUY ĐỊNH CHUNG",
        0,
        3,
    )
    assert document.nodes[0].head_block_ids == (chuong.id, title.id)
    # One heading segment holds both the marker and the consumed title; the title block adds nothing.
    assert texts(document) == [
        "# Chương I QUY ĐỊNH CHUNG",
        "# Điều 1. Phạm vi",
        "Nội dung",
    ]


def test_next_block_uses_the_rest_of_its_own_line_first() -> None:
    b = Blocks()
    document = build(b.text("Chương I QUY ĐỊNH CHUNG"), b.text("Đoạn đầu chương"))
    assert outline(document)[0][5] == "QUY ĐỊNH CHUNG"
    assert texts(document) == [
        "# Chương I QUY ĐỊNH CHUNG",
        "Đoạn đầu chương",
    ]  # not consumed


def test_next_block_followed_by_a_heading_has_no_title() -> None:
    b = Blocks()
    document = build(b.text("Chương I"), b.text("Điều 1. Phạm vi"))
    assert [n.title for n in document.nodes] == [None, "Phạm vi"]


def test_next_block_followed_by_a_table_has_no_title() -> None:
    b = Blocks()
    document = build(b.text("Chương I"), b.table(), b.text("Đoạn"))
    assert outline(document)[0][5] is None
    assert texts(document) == ["# Chương I", "[table]", "Đoạn"]


def test_only_the_block_right_after_can_be_the_title() -> None:
    b = Blocks()
    document = build(b.text("Chương I"), b.image(), b.text("QUY ĐỊNH CHUNG"))
    assert outline(document)[0][5] is None
    assert texts(document) == ["# Chương I", "[image]", "QUY ĐỊNH CHUNG"]


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
        b.text("Điều 1. A"),
        b.text("1. X"),
        b.text("a) x1"),
        b.text("a2"),
        b.text("2. Y"),
        b.text("Điều 2. B"),
        b.text("Z"),
    )
    assert texts(document) == [
        "# Điều 1. A",
        "# 1.",
        "X",
        "# a)",
        "x1",
        "a2",
        "# 2.",
        "Y",
        "# Điều 2. B",
        "Z",
    ]
    assert [(n.level, n.start, n.end) for n in document.nodes] == [
        ("dieu", 0, 8),
        ("khoan", 1, 6),
        ("diem", 3, 6),
        ("khoan", 6, 8),
        ("dieu", 8, 10),
    ]


def test_headings_with_nothing_under_them() -> None:
    # As in a typed table of contents: such a node holds only its own heading segment.
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.text("Điều 2. B"), b.text("Nội dung"))
    assert [(n.start, n.end) for n in document.nodes] == [(0, 1), (1, 3)]


def test_a_plain_paragraph_goes_to_the_parent_of_closing_items() -> None:
    b = Blocks()
    document = build(
        b.text("Điều 1. A"), b.text("1. X"), b.text("- gạch"), b.text("Đoạn kết khoản")
    )
    assert texts(document) == [
        "# Điều 1. A",
        "# 1.",
        "X",
        "# -",
        "gạch",
        "Đoạn kết khoản",
    ]
    assert [(n.level, n.start, n.end) for n in document.nodes] == [
        ("dieu", 0, 6),
        ("khoan", 1, 6),  # the closing remark is inside the khoản
        ("gach", 3, 5),  # but not inside the dash item
    ]


def test_tables_and_images_stay_inside_the_open_node() -> None:
    b = Blocks()
    document = build(
        b.text("Điều 1. A"), b.text("- gạch"), b.table(), b.image(), b.text("Đoạn")
    )
    assert texts(document) == [
        "# Điều 1. A",
        "# -",
        "gạch",
        "[table]",
        "[image]",
        "Đoạn",
    ]
    assert [(n.level, n.start, n.end) for n in document.nodes] == [
        ("dieu", 0, 6),
        ("gach", 1, 5),
    ]
    assert isinstance(document.segments[3], TableSegment)
    assert isinstance(document.segments[4], ImageSegment)


# ============================================================ end matter


def test_end_matter_closes_everything_and_keeps_its_segments() -> None:
    b = Blocks()
    document = build(
        b.text("Điều 1. A"),
        b.text("1. X"),
        b.text("Nơi nhận:"),
        b.text("- Như Điều 3;"),
    )
    assert texts(document) == ["# Điều 1. A", "# 1.", "X", "Nơi nhận:", "- Như Điều 3;"]
    assert (document.body_start, document.body_end) == (0, 3)
    assert [(n.level, n.end) for n in document.nodes] == [("dieu", 3), ("khoan", 3)]


def test_end_matter_in_a_table() -> None:
    b = Blocks()
    document = build(
        b.text("Điều 1. A"),
        b.text("Nội dung"),
        b.table("Nơi nhận: - Như Điều 26; - Lưu: VT"),
    )
    assert (document.body_start, document.body_end) == (0, 2)
    assert texts(document) == ["# Điều 1. A", "Nội dung", "[table]"]


def test_end_matter_right_after_a_heading_with_nothing_under_it() -> None:
    # The body holds only that heading.
    b = Blocks()
    document = build(b.text("Điều 1. A"), b.text("Nơi nhận:"))
    assert (document.body_start, document.body_end) == (0, 1)
    assert texts(document) == ["# Điều 1. A", "Nơi nhận:"]


# ============================================================ assemble_document


def test_assemble_refuses_blocks_from_another_document() -> None:
    b = Blocks()
    blocks = (b.text("Điều 1. A"), b.text("Nội dung"))
    classified = classify_blocks(PROFILE, blocks[:1])
    with pytest.raises(ValueError):
        assemble_document(PROFILE, make_document(blocks=blocks), classified)


def test_build_is_classify_then_assemble() -> None:
    b = Blocks()
    physical = make_document(
        blocks=(b.text("Điều 1. A"), b.text("1. X"), b.text("Nội dung"))
    )
    assert build_logical_document(PROFILE, physical) == assemble_document(
        PROFILE, physical, classify_blocks(PROFILE, physical.blocks)
    )


# ============================================================ VN_REGULATION, a stretch of the real document


def test_vn_regulation_real_stretch() -> None:
    none_format = make_declared_numbering(num_fmt="none", segment_formats=())
    b = Blocks()

    def heading_list(text: str, level: int) -> Any:
        info = make_list_info(
            displayed_marker=None,
            is_ordered=False,
            nested_level=level,
            declared_numbering=none_format,
        )
        return make_block(
            id=b._id(), block_type=BlockType.LIST_ITEM, text=text, list_info=info
        )

    def decimal(text: str, displayed: str) -> Any:
        info = make_list_info(
            displayed_marker=displayed,
            is_ordered=True,
            declared_numbering=make_declared_numbering(
                num_fmt="decimal", segment_formats=("decimal",)
            ),
        )
        return make_block(
            id=b._id(), block_type=BlockType.LIST_ITEM, text=text, list_info=info
        )

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
        "Theo đề nghị của Giám đốc Ban Công nghệ,",  # 0  front matter
        "# Chương I QUY ĐỊNH CHUNG",  # 1
        "# Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng",  # 2
        "# 1.",  # 3
        "Quy định này quy định về việc quản lý kiến trúc tổng thể CNTT",
        "# 2.",  # 5
        "Quy định này được áp dụng cho toàn hệ thống",
        "# Điều 2. Giải thích từ ngữ",  # 7
        "# 1.",  # 8
        "“Cloud native”: Là một cách tiếp cận",
        "# -",  # 10
        "Kiến trúc tổng thể CNTT bao gồm các kiến trúc thành phần sau:",
        "# +",  # 12
        "Kiến trúc ứng dụng",
        "Các thuật ngữ khác được hiểu theo quy định.",  # 14: closes the dash items
        "[table]",  # 15 end matter
    ]
    assert (document.body_start, document.body_end) == (1, 15)
    assert outline(document) == [
        ("n0", None, "chuong", "I", "Chương I", "QUY ĐỊNH CHUNG", 1, 15),
        (
            "n1",
            "n0",
            "dieu",
            "1",
            "Điều 1.",
            "Phạm vi điều chỉnh và đối tượng áp dụng",
            2,
            7,
        ),
        ("n2", "n1", "khoan", "1", "1.", None, 3, 5),
        ("n3", "n1", "khoan", "2", "2.", None, 5, 7),
        ("n4", "n0", "dieu", "2", "Điều 2.", "Giải thích từ ngữ", 7, 15),
        ("n5", "n4", "khoan", "1", "1.", None, 8, 15),
        (
            "n6",
            "n5",
            "gach",
            None,
            "-",
            None,
            10,
            14,
        ),  # "+" has no `inside`, so it nests under the dash
        ("n7", "n6", "cong", None, "+", None, 12, 14),
    ]


# ============================================================ heading segments


def test_every_node_starts_with_its_own_heading_segment() -> None:
    b = Blocks()
    document = build(
        b.text("Chương I"),
        b.text("QUY ĐỊNH CHUNG"),
        b.text("Điều 1. A"),
        b.numbered("Nội dung", "1."),
        b.text("- gạch"),
    )
    for node in document.nodes:
        heading = document.segments[node.start]
        assert isinstance(heading, HeadingSegment)
        assert (heading.node_id, heading.marker, heading.title) == (
            node.id,
            node.marker,
            node.title,
        )


def test_heading_segment_points_to_the_heading_block() -> None:
    b = Blocks()
    chuong, title = b.text("Chương I"), b.text("QUY ĐỊNH CHUNG")
    document = build(chuong, title)
    heading = document.segments[0]
    assert isinstance(heading, HeadingSegment)
    assert (
        heading.block_id == chuong.id
    )  # the marker's block; the title block is in head_block_ids


def test_next_block_title_is_written_into_the_heading_segment() -> None:
    # The heading segment is emitted before the title arrives and rebuilt when it does.
    b = Blocks()
    document = build(b.text("Chương II"), b.text("TỔ CHỨC THỰC HIỆN"), b.text("Đoạn"))
    heading = document.segments[0]
    assert isinstance(heading, HeadingSegment) and heading.title == "TỔ CHỨC THỰC HIỆN"


def test_heading_segments_and_nodes_correspond_one_to_one() -> None:
    b = Blocks()
    document = build(
        b.text("Điều 1. A"), b.text("1. X"), b.text("Điều 2. B"), b.text("Nội dung")
    )
    heading_ids = [
        s.node_id for s in document.segments if isinstance(s, HeadingSegment)
    ]
    assert heading_ids == [n.id for n in document.nodes]


def test_body_text_of_a_heading_follows_its_heading_segment() -> None:
    b = Blocks()
    khoan = b.text("1. Nội dung khoản")
    document = build(b.text("Điều 1. A"), khoan)
    assert [type(s).__name__ for s in document.segments] == [
        "HeadingSegment",
        "HeadingSegment",
        "TextSegment",
    ]
    assert document.segments[1].block_id == document.segments[2].block_id == khoan.id


# ============================================================ sections


def test_nodes_copy_is_section_from_their_level() -> None:
    b = Blocks()
    document = build(
        b.text("Chương I"),
        b.text("QUY ĐỊNH CHUNG"),
        b.text("Điều 1. A"),
        b.text("1. X"),
        b.text("a) Y"),
        b.text("- Z"),
    )
    assert [(n.level, n.is_section) for n in document.nodes] == [
        ("chuong", True),
        ("dieu", True),
        ("khoan", False),
        ("diem", False),
        ("gach", False),
    ]


def test_vn_regulation_marks_chuong_and_dieu_as_sections() -> None:
    b = Blocks()
    physical = make_document(
        blocks=(
            b.text("Chương I"),
            b.text("QUY ĐỊNH CHUNG"),
            b.text("Điều 1. A"),
            b.text("1. X"),
        )
    )
    document = build_logical_document(VN_REGULATION, physical)
    assert [(n.level, n.is_section) for n in document.nodes] == [
        ("chuong", True),
        ("dieu", True),
        ("khoan", False),
    ]
