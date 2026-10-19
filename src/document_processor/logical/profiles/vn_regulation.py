"""Vietnamese legal and regulation documents (luật, nghị định, thông tư, quy chế…).

Hierarchy, outermost first (a level's position in `levels` is its rank):

    Chương I              chapter; title on the next line
      Mục 1               section (optional); title on the next line
        Điều 9. Phạm vi…  article; title after the marker
          1. …            khoản (clause), only inside an Điều
            a) …          điểm (point), only inside an Điều; "a)" and "a." mix in one list
              - …         dash item
                + …       plus item, one step deeper than a dash

Words are matched case-insensitively ("CHƯƠNG I" and "Chương I" both count);
numerals are not. The document ends at "Nơi nhận:" (the distribution list).
"""

from document_processor.logical.levels import GlyphMarkerSpec, LevelSpec, TemplateMarkerSpec
from document_processor.logical.profile import Profile

VN_REGULATION = Profile(
    name="vn_regulation",
    version="2",
    levels=(
        LevelSpec(
            name="chuong",
            marker=TemplateMarkerSpec(templates=("Chương {upper_roman|arabic}",)),
            title="next_block",
        ),
        LevelSpec(
            name="muc",
            marker=TemplateMarkerSpec(templates=("Mục {upper_roman|arabic}",)),
            title="next_block",
        ),
        LevelSpec(
            name="dieu",
            marker=TemplateMarkerSpec(templates=("Điều {arabic}.", "Điều {arabic}:")),
            title="rest_of_paragraph",
        ),
        LevelSpec(
            name="khoan",
            marker=TemplateMarkerSpec(templates=("{arabic}.",)),
            inside=("dieu",),
        ),
        LevelSpec(
            name="diem",
            marker=TemplateMarkerSpec(templates=("{lower_vn})", "{lower_vn}.")),
            inside=("dieu",),
        ),
        LevelSpec(
            name="gach",
            marker=GlyphMarkerSpec(glyphs=("dash",)),
            closes_on_plain_paragraph=True,
        ),
        LevelSpec(
            name="cong",
            marker=GlyphMarkerSpec(glyphs=("plus",)),
            closes_on_plain_paragraph=True,
        ),
    ),
    # With the colon only: "Nơi nhận hồ sơ: …" (where to file an application) can start
    # a body paragraph, and everything after an end-matter match is cut from the body.
    end_matter=TemplateMarkerSpec(templates=("Nơi nhận:",)),
)