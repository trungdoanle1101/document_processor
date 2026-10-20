"""Tests for logical/profiles/vn_regulation.py: the VN_REGULATION profile.

Two parts: the declared structure (level order, title rules, `inside`,
closing rules, end matter) and how single lines are read by each level's
marker. The full test matrix (step 7) will extend the line tables with real
documents; `inside` and titles are applied by the builder, so here a "1." line
reads as a khoản even with no Điều around it.
"""

import pytest

from document_processor.logical.glyph_marker import GlyphMatch
from document_processor.logical.levels import GlyphMarkerSpec, TemplateMarkerSpec
from document_processor.logical.profiles.vn_regulation import VN_REGULATION
from document_processor.logical.template_marker import TemplateMatch

LEVEL_NAMES = ("chuong", "muc", "dieu", "khoan", "diem", "gach", "cong")


def levels_reading_text(text: str) -> list[str]:
    """Names of the levels whose marker reads `text` as a typed marker."""
    found = []
    assert VN_REGULATION.levels is not None
    for level in VN_REGULATION.levels:
        result = VN_REGULATION.get_marker(level.name).match_text(text)
        if result:
            found.append(level.name)
    return found


def levels_reading_list_marker(displayed: str) -> list[str]:
    found = []
    assert VN_REGULATION.levels is not None
    for level in VN_REGULATION.levels:
        if VN_REGULATION.get_marker(level.name).match_list_marker(displayed):
            found.append(level.name)
    return found


def template_reading(level: str, text: str) -> TemplateMatch:
    readings = VN_REGULATION.get_marker(level).match_text(text)
    assert len(readings) == 1, readings
    assert isinstance(readings[0], TemplateMatch)
    return readings[0]


# ============================================================ structure


def test_identity() -> None:
    assert VN_REGULATION.name == "vn_regulation"
    assert VN_REGULATION.version == "2"


def test_level_order_is_outermost_first() -> None:
    assert VN_REGULATION.levels is not None
    assert tuple(level.name for level in VN_REGULATION.levels) == LEVEL_NAMES
    assert [VN_REGULATION.get_rank(n) for n in LEVEL_NAMES] == list(range(len(LEVEL_NAMES)))


@pytest.mark.parametrize(
    ("level", "title"),
    [
        ("chuong", "next_block"),
        ("muc", "next_block"),
        ("dieu", "rest_of_paragraph"),
        ("khoan", "none"),
        ("diem", "none"),
        ("gach", "none"),
        ("cong", "none"),
    ],
)
def test_title_rules(level: str, title: str) -> None:
    assert VN_REGULATION.get_level(level).title == title


@pytest.mark.parametrize(
    ("level", "inside"),
    [
        ("chuong", ()),
        ("muc", ()),  # Mục is optional, so nothing may require it
        ("dieu", ()),
        ("khoan", ("dieu",)),
        ("diem", ("dieu",)),  # directly under Điều or under a khoản
        ("gach", ()),
        ("cong", ()),
    ],
)
def test_inside(level: str, inside: tuple[str, ...]) -> None:
    assert VN_REGULATION.get_level(level).inside == inside


@pytest.mark.parametrize(
    ("level", "closes"),
    [("chuong", False), ("muc", False), ("dieu", False), ("khoan", False), ("diem", False), ("gach", True), ("cong", True)],
)
def test_closes_on_plain_paragraph(level: str, closes: bool) -> None:
    assert VN_REGULATION.get_level(level).closes_on_plain_paragraph is closes


def test_marker_kinds() -> None:
    for name in ("chuong", "muc", "dieu", "khoan", "diem"):
        assert isinstance(VN_REGULATION.get_level(name).marker, TemplateMarkerSpec), name
    for name in ("gach", "cong"):
        assert isinstance(VN_REGULATION.get_level(name).marker, GlyphMarkerSpec), name


def test_markers_are_case_insensitive() -> None:
    for name in ("chuong", "muc", "dieu", "khoan", "diem"):
        spec = VN_REGULATION.get_level(name).marker
        assert isinstance(spec, TemplateMarkerSpec)
        assert spec.case_sensitive is False


def test_no_word_bullet_level() -> None:
    assert VN_REGULATION.word_bullet_level is None


def test_default_registries() -> None:
    assert "lower_vn" in VN_REGULATION.counter_styles
    assert "dash" in VN_REGULATION.glyph_styles


# ============================================================ typed markers


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("Chương I", "chuong"),
        ("Chương IV", "chuong"),
        ("CHƯƠNG IV", "chuong"),
        ("chương iv", None),  # words ignore case, numerals don't
        ("Chương 4", "chuong"),
        ("Chương XII QUY ĐỊNH CHUNG", "chuong"),
        ("Mục 1", "muc"),
        ("MỤC II", "muc"),
        ("Điều 1. Phạm vi điều chỉnh", "dieu"),
        ("Điều 10: Hiệu lực thi hành", "dieu"),
        ("ĐIỀU 25. Điều khoản chuyển tiếp", "dieu"),
        ("Điều 9.", "dieu"),
        ("1. Nội dung khoản", "khoan"),
        ("12. Nội dung khoản", "khoan"),
        ("a) Nội dung điểm", "diem"),
        ("b. Nội dung điểm", "diem"),
        ("đ) Nội dung điểm", "diem"),
        ("i) Nội dung điểm", "diem"),
        ("- Nội dung gạch đầu dòng", "gach"),
        ("– Nội dung, gạch ngang dài", "gach"),
        ("+ Nội dung dấu cộng", "cong"),
    ],
)
def test_lines_with_a_marker(text: str, level: str | None) -> None:
    assert levels_reading_text(text) == ([] if level is None else [level])


@pytest.mark.parametrize(
    "text",
    [
        "Chương trình đào tạo",  # word after the literal, no numeral
        "Mục tiêu của dự án",
        "Mục đích",
        "Điều kiện để được cấp phép",
        "Điều 5.Phạm vi",  # no blank after the marker: strict on purpose
        "Điều IV. Sai kiểu số",  # Điều is numbered in arabic only
        "5.000 đồng",  # thousands separator, not a khoản
        "2.5 triệu đồng",
        "1.1. Mục tiêu",  # a typed table-of-contents entry, not a khoản
        "2.3.1 Thiết kế",
        "1) Khoản viết bằng ngoặc",  # not declared yet
        "ii) Nội dung",  # điểm are single Vietnamese letters
        "f) Nội dung",  # f is not in the Vietnamese alphabet
        "w. Nội dung",
        "-5% so với năm trước",
        "--",
        "Nội dung - tiếp theo",
        "Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015;",
        "",
    ],
)
def test_lines_without_a_marker(text: str) -> None:
    assert levels_reading_text(text) == []


def test_every_line_is_read_by_at_most_one_level() -> None:
    lines = ["Chương I", "Mục 1", "Điều 1. X", "1. X", "a) X", "b. X", "- X", "+ X", "i) X", "v) X"]
    for text in lines:
        assert len(levels_reading_text(text)) <= 1, text


# ---------- numbers


@pytest.mark.parametrize(
    ("level", "text", "number_text", "value"),
    [
        ("chuong", "Chương IV", "IV", 4),
        ("chuong", "Chương 12", "12", 12),
        ("muc", "Mục II", "II", 2),
        ("dieu", "Điều 25. X", "25", 25),
        ("khoan", "3. X", "3", 3),
        ("diem", "a) X", "a", 1),
        ("diem", "đ) X", "đ", 5),
        ("diem", "e. X", "e", 6),
    ],
)
def test_numbers(level: str, text: str, number_text: str, value: int) -> None:
    reading = template_reading(level, text)
    assert [(n.text, n.value) for n in reading.numerals] == [(number_text, value)]


def test_chuong_reads_roman_and_arabic_as_separate_styles() -> None:
    assert template_reading("chuong", "Chương IV").numerals[0].style == "upper_roman"
    assert template_reading("chuong", "Chương 4").numerals[0].style == "arabic"


def test_dieu_marker_end_splits_off_the_title() -> None:
    text = "Điều 9. Phạm vi điều chỉnh"
    reading = template_reading("dieu", text)
    assert text[: reading.end] == "Điều 9."
    assert text[reading.end :].strip() == "Phạm vi điều chỉnh"


def test_diem_reports_which_template_matched() -> None:
    assert template_reading("diem", "a) X").template == "{lower_vn})"
    assert template_reading("diem", "b. X").template == "{lower_vn}."


def test_glyph_levels_report_their_style() -> None:
    [gach] = VN_REGULATION.get_marker("gach").match_text("– X")
    [cong] = VN_REGULATION.get_marker("cong").match_text("+ X")
    assert isinstance(gach, GlyphMatch) and gach.style == "dash"
    assert isinstance(cong, GlyphMatch) and cong.style == "plus"


# ============================================================ Word list markers


@pytest.mark.parametrize(
    ("displayed", "level"),
    [
        ("Chương I", "chuong"),
        ("Điều 1.", "dieu"),
        ("1.", "khoan"),
        ("a)", "diem"),
        ("-", "gach"),
        ("–", "gach"),
        ("+", "cong"),
    ],
)
def test_generated_list_markers(displayed: str, level: str) -> None:
    assert levels_reading_list_marker(displayed) == [level]


@pytest.mark.parametrize(
    "displayed",
    [
        "•",  # Word bullets: no word_bullet level in this profile
        "◦",
        "▪",
        "✓",
        "1.1.",  # multi-level numbering
        "Chương I –",  # the whole marker must be the template
        "1",
        "",
    ],
)
def test_list_markers_no_level_reads(displayed: str) -> None:
    assert levels_reading_list_marker(displayed) == []


# ============================================================ end matter


@pytest.mark.parametrize("text", ["Nơi nhận:", "NƠI NHẬN:", "Nơi nhận: - Như Điều 3;"])
def test_end_matter(text: str) -> None:
    assert VN_REGULATION.end_matter_marker is not None
    assert VN_REGULATION.end_matter_marker.match_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "Nơi nhận hồ sơ: Bộ phận một cửa",  # a body paragraph; must not end the body
        "Nơi nhận",  # no colon
        "Gửi nơi nhận:",
    ],
)
def test_not_end_matter(text: str) -> None:
    assert VN_REGULATION.end_matter_marker is not None
    assert not VN_REGULATION.end_matter_marker.match_text(text)


def test_end_matter_is_no_level() -> None:
    assert levels_reading_text("Nơi nhận:") == []