"""Test matrix: real lines, and the pass 1 candidates each one must give.

Each row is one block: its text, its Word list marker (if any) and the expected
candidates, written as  level[list reading|text reading].  A reading is "-" when
that source gave nothing; otherwise the numerals as written, joined by "." (a
letter carries its alphabet, a/lat or a/vn), or the glyph style. A level that
reads a line in two ways appears twice.

    khoan[1|-]          Word shows "1.", the text has no typed marker
    khoan[1|1]          Word shows "1." and the author typed "1." too: one candidate
    diem[a/lat|-] diem[a/vn|-]   "a)" read as Latin and as Vietnamese; pass 2 picks
    gach[-|dash]        a typed "- "

Rows marked TRAP must give nothing, or give a reading that only `inside`, pass 2
or the builder can reject (the comment says which). When a row fails, pytest
prints both lists, so the failure says what was read instead.

Section 1 comes from a real internal regulation (2025 "Quy định quản lý kiến
trúc tổng thể CNTT", cleaned dump, block ids kept). Section 2 is a small made-up
profile with other conventions, to keep the marker model general.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from document_processor.core.models import BlockType
from document_processor.logical.candidates import Candidate, find_candidates
from document_processor.logical.levels import LevelSpec, MarkerMatch, TemplateMarkerSpec
from document_processor.logical.profile import Profile
from document_processor.logical.profiles.vn_regulation import VN_REGULATION
from document_processor.logical.template_marker import TemplateMatch
from tests.factories import make_block, make_declared_numbering, make_list_info

_ALPHABET_TAG = {
    "lower_latin": "lat",
    "upper_latin": "LAT",
    "lower_vn": "vn",
    "upper_vn": "VN",
}
_UNORDERED_FORMATS = {"bullet", "none"}


@dataclass(frozen=True)
class Row:
    block_id: str
    text: str
    expected: str  # space-separated candidates; "" for none
    marker: str | None = None  # Word's displayed marker
    num_fmt: str | None = None  # Word's declared format; None for a plain paragraph
    nested_level: int = 0

    def block(self) -> Any:
        if self.num_fmt is None:
            return make_block(id=self.block_id, text=self.text)
        info = make_list_info(
            displayed_marker=self.marker,
            is_ordered=self.num_fmt not in _UNORDERED_FORMATS,
            nested_level=self.nested_level,
            declared_numbering=make_declared_numbering(
                num_fmt=self.num_fmt,
                segment_formats=()
                if self.num_fmt in _UNORDERED_FORMATS
                else (self.num_fmt,),
            ),
        )
        return make_block(
            id=self.block_id,
            block_type=BlockType.LIST_ITEM,
            text=self.text,
            list_info=info,
        )


def reading(match: MarkerMatch | None) -> str:
    if match is None:
        return "-"
    if isinstance(match, TemplateMatch):
        parts = []
        for n in match.numerals:
            tag = _ALPHABET_TAG.get(n.style)
            parts.append(f"{n.text}/{tag}" if tag else n.text)
        return ".".join(parts)
    return match.style


def render(candidates: tuple[Candidate, ...]) -> list[str]:
    return [
        f"{c.level}[{reading(c.list_match)}|{reading(c.text_match)}]"
        for c in candidates
    ]


def check(profile: Profile, row: Row) -> None:
    assert render(find_candidates(profile, row.block())) == row.expected.split()


# ============================================================ 1. VN_REGULATION, real document

NONE = "none"  # Word list that numbers nothing: the document's heading list (list id 1)

VN_ROWS = [
    # ---------- front matter: nothing reads as a heading
    Row("pb2", "PHẦN KIỂM SOÁT", ""),
    Row("pb7", "QUY ĐỊNH", "", num_fmt=NONE, nested_level=8),
    Row(
        "pb11",
        "Căn cứ Điều lệ về tổ chức và hoạt động của Ngân hàng TMCP Đầu tư và Phát triển",
        "",
    ),
    Row("pb13", "QUY ĐỊNH:", "", num_fmt=NONE, nested_level=6),
    # ---------- Chương / Mục: typed, inside Word's heading list that displays nothing
    Row("pb14", "Chương I", "chuong[-|I]", num_fmt=NONE, nested_level=6),
    Row(
        "pb15", "QUY ĐỊNH CHUNG", "", num_fmt=NONE, nested_level=6
    ),  # the title on the next block
    Row("pb38", "Chương II", "chuong[-|II]", num_fmt=NONE, nested_level=6),
    Row("pb57", "Mục I", "muc[-|I]", num_fmt=NONE, nested_level=6),
    Row("pb58", "KIẾN TRÚC ỨNG DỤNG", "", num_fmt=NONE, nested_level=6),
    Row(
        "pb550", "ĐIỀU KHOẢN THI HÀNH", "", num_fmt=NONE, nested_level=6
    ),  # TRAP: "ĐIỀU" with no number
    # ---------- Điều
    Row(
        "pb16",
        "Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng",
        "dieu[-|1]",
        num_fmt=NONE,
    ),
    Row("pb20", "Điều 2. Giải thích từ ngữ", "dieu[-|2]", num_fmt=NONE),
    Row("pb551", "Điều 25. Hiệu lực thi hành", "dieu[-|25]", num_fmt=NONE),
    # ---------- khoản: generated by Word, and typed
    Row(
        "pb17",
        "Quy định này quy định về việc quản lý kiến trúc tổng thể CNTT tại BIDV",
        "khoan[1|-]",
        "1.",
        "decimal",
    ),
    Row(
        "pb30",
        "“Hệ thống Log tập trung”: Là hệ thống cho phép tập trung hóa dữ liệu log",
        "khoan[10|-]",
        "10.",
        "decimal",
    ),
    Row(
        "pb172", "Mô hình", "khoan[1|-]", "1.", "decimal", nested_level=3
    ),  # Word level 3, still a khoản
    Row("pb544", "2. Giám sát công tác tuân thủ kiến trúc:", "khoan[-|2]"),
    # ---------- plain paragraphs between items
    Row(
        "pb37",
        "Các thuật ngữ, từ, cụm từ viết tắt khác được sử dụng trong văn bản này",
        "",
    ),
    Row(
        "pb68",
        "Chi tiết năng lực nghiệp vụ của các thành phần thuộc các Lớp nêu tại Phụ lục",
        "",
    ),
    Row("pb333", "(*) SDS xem xét triển khai trong tương lai", ""),
    # ---------- sub-clauses under a khoản: typed with and without a final dot, and generated
    Row(
        "pb175",
        "2.1 Nguồn dữ liệu (Source): Là tập hợp các nguồn dữ liệu trong/ngoài ngân hàng",
        "khoan_cap_2[-|2.1]",
    ),
    Row(
        "pb177",
        "2.3. Tầng xử lý và lưu trữ dữ liệu (Data Processing and Stores):",
        "khoan_cap_2[-|2.3]",
    ),
    Row("pb239", "2.5 Tầng sử dụng dữ liệu (Data Consumption)", "khoan_cap_2[-|2.5]"),
    Row(
        "pb178",
        "2.3.1 Tầng xử lý dữ liệu (Data Processing): Là Khung công nghệ có chức năng",
        "khoan_cap_3[-|2.3.1]",
    ),
    Row(
        "pb237", "2.4.Tầng truy cập dữ liệu (Data Access)", ""
    ),  # TRAP, accepted loss: a typo, no blank after "2.4."
    Row(
        "pb392",
        "Các khối về quy trình:",
        "khoan_cap_2[2.1|-]",
        "2.1",
        "decimal",
        nested_level=1,
    ),
    # ---------- điểm: Word's lowerLetter is Latin; typed letters both ways; "a)" and "b." mix
    Row(
        "pb63",
        "Lớp tiếp thị, thu hút, phục vụ và giữ chân khách hàng",
        "diem[a/lat|-] diem[a/vn|-]",
        "a)",
        "lowerLetter",
    ),
    Row(
        "pb117",
        "Được hỗ trợ, cung cấp tài liệu tốt của các hãng/cộng đồng mã nguồn mở.",
        "diem[f/lat|-]",
        "f)",
        "lowerLetter",
    ),
    Row(
        "pb77",
        "a) Các chức năng phải được thiết kế hỗ trợ xuyên suốt giữa các kênh",
        "diem[-|a/lat] diem[-|a/vn]",
    ),
    Row(
        "pb94",
        "f) Mỗi microservice có thể mở rộng (scaling) linh hoạt với luồng xử lý",
        "diem[-|f/lat]",
    ),
    Row(
        "pb179", "a. Chuẩn bị dữ liệu (Data Preparation).", "diem[-|a/lat] diem[-|a/vn]"
    ),
    Row(
        "pb227",
        "f. Vùng Dữ liệu chủ (MDM - Master Data Management): Là nơi lưu trữ",
        "diem[-|f/lat]",
    ),
    Row(
        "pb548",
        "b. Ban CN trình Phó Tổng Giám đốc phụ trách/Chủ tịch Hội đồng kiến trúc",
        "diem[-|b/lat] diem[-|b/vn]",
    ),
    # ---------- dash and plus items: typed, and Word bullets that show a dash
    Row(
        "pb44",
        "- Kiến trúc tổng thể CNTT bao gồm các kiến trúc thành phần sau:",
        "gach[-|dash]",
    ),
    Row(
        "pb132",
        "- (1) Cổng API (External API Gateway): Là thành phần cho phép các phần mềm",
        "gach[-|dash]",
    ),
    Row(
        "pb48",
        "Về quy định đối với từng thành phần kiến trúc: Nêu tại Chương III quy định",
        "gach[dash|-]",
        "-",
        "bullet",
    ),
    Row(
        "pb409",
        "Phần mềm: bao gồm Web forms app, Web services/APIs,...",
        "gach[dash|-]",
        "-",
        "bullet",
    ),
    Row(
        "pb45",
        "+ Kiến trúc ứng dụng: Gồm Lớp tiếp thị, thu hút, phục vụ và giữ chân khách hàng",
        "cong[-|plus]",
    ),
    # ---------- Word bullets no level claims (VN_REGULATION has no word_bullet level)
    Row(
        "pb194",
        "Ứng dụng sử dụng mô hình AI/ML được nhúng sẵn trong phần mềm thương mại",
        "",
        "✓",
        "bullet",
    ),
    Row(
        "pb217",
        "ODS, SOR: Kho dữ liệu nguyên tố theo mô hình dữ liệu 09 thực thể IBM",
        "",
        "•",
        "bullet",
    ),
]


@pytest.mark.parametrize("row", VN_ROWS, ids=lambda r: f"{r.block_id} {r.text[:30]}")
def test_vn_regulation(row: Row) -> None:
    check(VN_REGULATION, row)


# ============================================================ 2. other conventions

GENERAL = Profile(
    name="general",
    version="1",
    levels=(
        LevelSpec(
            name="part",
            is_section=False,
            marker=TemplateMarkerSpec(templates=("Part {upper_roman}",)),
            title="next_block",
        ),
        LevelSpec(
            name="article",
            is_section=False,
            marker=TemplateMarkerSpec(
                templates=("Article {arabic}", "Article {arabic}er")
            ),
            title="rest_of_paragraph",
        ),
        LevelSpec(
            name="section",
            is_section=False,
            marker=TemplateMarkerSpec(templates=("§ {arabic}", "§{arabic}")),
        ),
        LevelSpec(
            name="annex_item",
            is_section=False,
            marker=TemplateMarkerSpec(templates=("{upper_latin}.{arabic}",)),
        ),
        LevelSpec(
            name="point",
            is_section=False,
            marker=TemplateMarkerSpec(templates=("({lower_roman|lower_latin})",)),
        ),
    ),
    end_matter=TemplateMarkerSpec(templates=("Done at",)),
)

GENERAL_ROWS = [
    Row("g1", "Part IV", "part[-|IV]"),
    Row(
        "g2", "Article 1er Objet", "article[-|1]"
    ),  # French "1er": a literal after the number
    Row("g3", "Article 12 Definitions", "article[-|12]"),
    Row("g4", "Articles 3 and 4 apply", ""),  # TRAP: "Articles" is another word
    Row("g5", "§ 12 Scope", "section[-|12]"),
    Row("g6", "§12 Scope", "section[-|12]"),
    Row("g7", "A.1 Data model", "annex_item[-|A/LAT.1]"),
    Row(
        "g8", "(iv) the fourth point", "point[-|iv]"
    ),  # roman only: letter styles read one letter
    Row(
        "g9", "(c) a lettered point", "point[-|c] point[-|c/lat]"
    ),  # c is also roman 100; pass 2 decides
    Row("g10", "(i) roman or letter", "point[-|i] point[-|i/lat]"),
    Row("g11", "(iv)the text runs on", ""),  # TRAP: no blank after the marker
]


@pytest.mark.parametrize(
    "row", GENERAL_ROWS, ids=lambda r: f"{r.block_id} {r.text[:30]}"
)
def test_other_conventions(row: Row) -> None:
    check(GENERAL, row)
