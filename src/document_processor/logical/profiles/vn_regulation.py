"""Vietnamese legal and regulation documents (luật, nghị định, thông tư, quy chế, quy định nội bộ…).

Hierarchy, outermost first (a level's position in `levels` is its rank):

    Chương I              chapter; title on the next line
      Mục 1               section (optional); title on the next line
        Điều 9. Phạm vi…  article; title after the marker
          1. …            khoản (clause), only inside an Điều
            2.1 …         sub-clause, only inside a khoản ("2.1", "2.1.")
              2.3.1 …     sub-sub-clause, only inside a khoản ("2.3.1", "2.3.1.")
                a) …      điểm (point), only inside an Điều; "a)" and "a." mix in one list
                  - …     dash item
                    + …   plus item, one step deeper than a dash

Điểm letters are read both ways: legal texts use the Vietnamese alphabet
(a b c d đ e g h …), internal regulations and Word's lowerLetter numbering use the
Latin one (… e f g h …). Every common letter has two readings; pass 2 picks one
from the sequence and the declared numbering format.

Words are matched case-insensitively ("CHƯƠNG I" and "Chương I" both count);
numerals are not. The document ends at "Nơi nhận:" (the distribution list).
"""

from document_processor.logical.levels import (
    GlyphMarkerSpec,
    LevelSpec,
    TemplateMarkerSpec,
)
from document_processor.logical.profile import Profile

VN_REGULATION = Profile(
    name="vn_regulation",
    version="4",
    levels=(
        LevelSpec(
            name="chuong",
            is_section=True,
            marker=TemplateMarkerSpec(templates=("Chương {upper_roman|arabic}",)),
            title="next_block",
        ),
        LevelSpec(
            name="muc",
            is_section=True,
            marker=TemplateMarkerSpec(templates=("Mục {upper_roman|arabic}",)),
            title="next_block",
        ),
        LevelSpec(
            name="dieu",
            is_section=True,
            marker=TemplateMarkerSpec(templates=("Điều {arabic}.", "Điều {arabic}:")),
            title="rest_of_paragraph",
        ),
        LevelSpec(
            name="khoan",
            is_section=False,
            marker=TemplateMarkerSpec(templates=("{arabic}.",)),
            inside=("dieu",),
        ),
        LevelSpec(
            # "2.1 Nguồn dữ liệu…" under khoản 2. Also reads "2.5 triệu đồng" at the start
            # of a line; `inside` and pass 2 are what keep that out.
            name="khoan_cap_2",
            is_section=False,
            marker=TemplateMarkerSpec(
                templates=("{arabic}.{arabic}", "{arabic}.{arabic}.")
            ),
            inside=("khoan",),
        ),
        LevelSpec(
            name="khoan_cap_3",
            is_section=False,
            marker=TemplateMarkerSpec(
                templates=("{arabic}.{arabic}.{arabic}", "{arabic}.{arabic}.{arabic}.")
            ),
            inside=("khoan",),
        ),
        LevelSpec(
            name="diem",
            is_section=False,
            marker=TemplateMarkerSpec(
                templates=("{lower_latin|lower_vn})", "{lower_latin|lower_vn}.")
            ),
            inside=("dieu",),
        ),
        LevelSpec(
            name="gach",
            is_section=False,
            marker=GlyphMarkerSpec(glyphs=("dash",)),
            closes_on_plain_paragraph=True,
        ),
        LevelSpec(
            name="cong",
            is_section=False,
            marker=GlyphMarkerSpec(glyphs=("plus",)),
            closes_on_plain_paragraph=True,
        ),
    ),
    # With the colon only: "Nơi nhận hồ sơ: …" (where to file an application) can start
    # a body paragraph, and everything after an end-matter match is cut from the body.
    end_matter=TemplateMarkerSpec(templates=("Nơi nhận:",)),
)
