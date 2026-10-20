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

LEVEL_NAMES = ("chuong", "muc", "dieu", "khoan", "khoan_cap_2", "khoan_cap_3", "diem", "gach", "cong")
TEMPLATE_LEVELS = ("chuong", "muc", "dieu", "khoan", "khoan_cap_2", "khoan_cap_3", "diem")


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


def template_readings(level: str, text: str) -> list[TemplateMatch]:
    readings = VN_REGULATION.get_marker(level).match_text(text)
    assert all(isinstance(r, TemplateMatch) for r in readings)
    return [r for r in readings if isinstance(r, TemplateMatch)]


def template_reading(level: str, text: str) -> TemplateMatch:
    [reading] = template_readings(level, text)
    return reading


def numbers_read(level: str, text: str) -> set[tuple[str, str, int]]:
    """(style, numeral as written, value) for every reading's first numeral."""
    return {(r.numerals[0].style, r.numerals[0].text, r.numerals[0].value) for r in template_readings(level, text)}


# ============================================================ structure


def test_identity() -> None:
    assert VN_REGULATION.name == "vn_regulation"
    assert VN_REGULATION.version == "3"


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
        ("khoan_cap_2", "none"),
        ("khoan_cap_3", "none"),
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
        ("khoan_cap_2", ("khoan",)),  # keeps "2.5 triệu đồng" out unless it starts a line inside a khoản
        ("khoan_cap_3", ("khoan",)),
        ("diem", ("dieu",)),  # directly under Điều or under a khoản
        ("gach", ()),
        ("cong", ()),
    ],
)
def test_inside(level: str, inside: tuple[str, ...]) -> None:
    assert VN_REGULATION.get_level(level).inside == inside


@pytest.mark.parametrize(
    ("level", "closes"),
    [(name, False) for name in TEMPLATE_LEVELS] + [("gach", True), ("cong", True)],
)
def test_closes_on_plain_paragraph(level: str, closes: bool) -> None:
    assert VN_REGULATION.get_level(level).closes_on_plain_paragraph is closes


def test_marker_kinds() -> None:
    for name in TEMPLATE_LEVELS:
        assert isinstance(VN_REGULATION.get_level(name).marker, TemplateMarkerSpec), name
    for name in ("gach", "cong"):
        assert isinstance(VN_REGULATION.get_level(name).marker, GlyphMarkerSpec), name


def test_markers_are_case_insensitive() -> None:
    for name in TEMPLATE_LEVELS:
        spec = VN_REGULATION.get_level(name).marker
        assert isinstance(spec, TemplateMarkerSpec)
        assert spec.case_sensitive is False


def test_no_word_bullet_level() -> None:
    assert VN_REGULATION.word_bullet_level is None


def test_default_registries() -> None:
    assert "lower_vn" in VN_REGULATION.counter_styles
    assert "lower_latin" in VN_REGULATION.counter_styles
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
        ("2.1 Nguồn dữ liệu (Source): Là tập hợp", "khoan_cap_2"),
        ("2.3. Tầng xử lý và lưu trữ dữ liệu", "khoan_cap_2"),
        ("2.3.1 Tầng xử lý dữ liệu", "khoan_cap_3"),
        ("2.3.1. Tầng xử lý dữ liệu", "khoan_cap_3"),
        ("1.1. Mục tiêu", "khoan_cap_2"),  # also a typed table-of-contents entry; the builder handles it
        ("2.5 triệu đồng", "khoan_cap_2"),  # syntax matches; `inside` and pass 2 must reject it
        ("5.000 đồng", "khoan_cap_2"),  # likewise
        ("a) Nội dung điểm", "diem"),
        ("b. Nội dung điểm", "diem"),
        ("đ) Nội dung điểm", "diem"),
        ("i) Nội dung điểm", "diem"),
        ("f) Nội dung điểm", "diem"),  # Latin letters (Word's lowerLetter, internal regulations)
        ("f. Vùng Dữ liệu chủ", "diem"),
        ("w. Nội dung", "diem"),
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
        "5.000đ",  # no blank after the number
        "2.4.Tầng truy cập dữ liệu",  # a typo in the real document: no blank after the marker
        "1.2.3.4 Quá sâu",  # four numbers: no level
        "1) Khoản viết bằng ngoặc",  # not declared yet
        "ii) Nội dung",  # điểm are single letters
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
    lines = ["Chương I", "Mục 1", "Điều 1. X", "1. X", "2.1 X", "2.1. X", "2.3.1 X", "a) X", "b. X", "- X", "+ X", "i) X", "v) X"]
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
    ],
)
def test_numbers(level: str, text: str, number_text: str, value: int) -> None:
    reading = template_reading(level, text)
    assert [(n.text, n.value) for n in reading.numerals] == [(number_text, value)]


def test_sub_clause_numbers() -> None:
    assert [(n.text, n.value) for n in template_reading("khoan_cap_2", "2.3. X").numerals] == [("2", 2), ("3", 3)]
    assert [n.value for n in template_reading("khoan_cap_3", "2.3.1 X").numerals] == [2, 3, 1]


@pytest.mark.parametrize(
    ("text", "readings"),
    [
        # Letters both alphabets share: two readings, equal up to d…
        ("a) X", {("lower_latin", "a", 1), ("lower_vn", "a", 1)}),
        ("d) X", {("lower_latin", "d", 4), ("lower_vn", "d", 4)}),
        # …and different from e on, because đ comes before e in Vietnamese.
        ("e. X", {("lower_latin", "e", 5), ("lower_vn", "e", 6)}),
        ("h) X", {("lower_latin", "h", 8), ("lower_vn", "h", 8)}),
        ("i) X", {("lower_latin", "i", 9), ("lower_vn", "i", 9)}),
        # One alphabet only: one reading.
        ("đ) X", {("lower_vn", "đ", 5)}),
        ("f) X", {("lower_latin", "f", 6)}),
        ("w. X", {("lower_latin", "w", 23)}),
    ],
)
def test_diem_letters_are_read_in_both_alphabets(text: str, readings: set[tuple[str, str, int]]) -> None:
    assert numbers_read("diem", text) == readings


def test_chuong_reads_roman_and_arabic_as_separate_styles() -> None:
    assert template_reading("chuong", "Chương IV").numerals[0].style == "upper_roman"
    assert template_reading("chuong", "Chương 4").numerals[0].style == "arabic"


def test_dieu_marker_end_splits_off_the_title() -> None:
    text = "Điều 9. Phạm vi điều chỉnh"
    reading = template_reading("dieu", text)
    assert text[: reading.end] == "Điều 9."
    assert text[reading.end :].strip() == "Phạm vi điều chỉnh"


def test_diem_reports_which_template_matched() -> None:
    assert {r.template for r in template_readings("diem", "a) X")} == {"{lower_latin|lower_vn})"}
    assert {r.template for r in template_readings("diem", "b. X")} == {"{lower_latin|lower_vn}."}


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
        ("2.1", "khoan_cap_2"),  # Word multi-level numbering, as in the real document
        ("1.1.", "khoan_cap_2"),
        ("a)", "diem"),
        ("f)", "diem"),  # Word's lowerLetter is Latin
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