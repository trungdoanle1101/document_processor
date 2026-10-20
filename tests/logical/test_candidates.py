"""Tests for logical/candidates.py: pass 1 of the classifier.

find_candidates reads one block with every level of a profile and returns every
reading, in rank order. The block's text and its Word list marker are matched
separately; a list reading and a text reading of the same level and the same
number are one candidate (a generated "1." in front of a typed "1."). Context
(inside, titles, picking one reading) is left to pass 2 and the builder.
"""

import dataclasses
from typing import Any

import pytest

from document_processor.core.models import BlockType, Region
from document_processor.logical.candidates import Candidate, find_candidates
from document_processor.logical.glyph_marker import GlyphMatch
from document_processor.logical.levels import GlyphMarkerSpec, LevelSpec, MarkerMatch, TemplateMarkerSpec
from document_processor.logical.markers import WORD_BULLET
from document_processor.logical.profile import Profile
from document_processor.logical.profiles.vn_regulation import VN_REGULATION
from document_processor.logical.template_marker import TemplateMatch
from tests.factories import make_block, make_declared_numbering, make_image_info, make_list_info, make_table_block


def templates(*items: str) -> TemplateMarkerSpec:
    return TemplateMarkerSpec(templates=items)


def glyphs(*names: str) -> GlyphMarkerSpec:
    return GlyphMarkerSpec(glyphs=names)


DIEU = LevelSpec(name="dieu", marker=templates("Điều {arabic}."))
KHOAN = LevelSpec(name="khoan", marker=templates("{arabic}."), inside=("dieu",))
ITEM = LevelSpec(name="item", marker=templates("{lower_roman|lower_vn})"))  # "i)" has two readings
GACH = LevelSpec(name="gach", marker=glyphs("dash"))


def profile(*levels: LevelSpec) -> Profile:
    return Profile(name="test", version="1", levels=levels or (DIEU, KHOAN, ITEM, GACH), end_matter=None)


PROFILE = profile()


def paragraph(text: str, **overrides: Any) -> Any:
    return make_block(text=text, **overrides)


def numbered(text: str, displayed: str | None) -> Any:
    return make_block(
        block_type=BlockType.LIST_ITEM, text=text, list_info=make_list_info(displayed_marker=displayed, is_ordered=True)
    )


def bulleted(text: str, displayed: str | None) -> Any:
    info = make_list_info(
        displayed_marker=displayed,
        is_ordered=False,
        declared_numbering=make_declared_numbering(num_fmt="bullet", segment_formats=()),
    )
    return make_block(block_type=BlockType.LIST_ITEM, text=text, list_info=info)


def number_of(match: MarkerMatch | None) -> object:
    """A short, comparable summary of a match: the numerals, or the glyph style."""
    if match is None:
        return None
    if isinstance(match, TemplateMatch):
        return tuple((n.style, n.value) for n in match.numerals)
    return match.style


def summary(candidates: tuple[Candidate, ...]) -> list[tuple[str, object, object]]:
    """(level, list reading, text reading) per candidate."""
    return [(c.level, number_of(c.list_match), number_of(c.text_match)) for c in candidates]


ARABIC_1 = (("arabic", 1),)
ARABIC_2 = (("arabic", 2),)


# ============================================================ Candidate


def test_candidate_needs_at_least_one_match() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Candidate(level="dieu", list_match=None, text_match=None)


def test_candidate_list_match_must_come_from_the_list_marker() -> None:
    [from_text] = PROFILE.get_marker("gach").match_text("- x")
    with pytest.raises(ValueError, match="list_match.source must be 'list_marker'"):
        Candidate(level="gach", list_match=from_text, text_match=None)


def test_candidate_text_match_must_come_from_the_text() -> None:
    [from_list] = PROFILE.get_marker("gach").match_list_marker("-")
    with pytest.raises(ValueError, match="text_match.source must be 'text'"):
        Candidate(level="gach", list_match=None, text_match=from_list)


def test_candidate_is_frozen() -> None:
    [found] = find_candidates(PROFILE, paragraph("Điều 1. X"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        found.level = "khoan"  # type: ignore[misc]


# ============================================================ blocks that are never candidates


def test_table_block() -> None:
    assert find_candidates(PROFILE, make_table_block()) == ()


def test_image_block() -> None:
    image = make_block(block_type=BlockType.IMAGE, text="", image_info=make_image_info())
    assert find_candidates(PROFILE, image) == ()


@pytest.mark.parametrize("region", [Region.HEADER, Region.FOOTER, Region.FOOTNOTE])
def test_blocks_outside_the_body(region: Region) -> None:
    assert find_candidates(PROFILE, paragraph("Điều 1. X", region=region)) == ()


def test_profile_without_levels() -> None:
    no_levels = Profile(name="flat", version="1", levels=None, end_matter=None)
    assert find_candidates(no_levels, paragraph("Điều 1. X")) == ()


@pytest.mark.parametrize("text", ["Nghị định này quy định về:", "Điều kiện cấp phép", "5.000 đồng", "-5%"])
def test_plain_paragraph(text: str) -> None:
    assert find_candidates(PROFILE, paragraph(text)) == ()


# ============================================================ one source


def test_typed_marker_only() -> None:
    [found] = find_candidates(PROFILE, paragraph("Điều 9. Phạm vi điều chỉnh"))
    assert found.level == "dieu"
    assert found.list_match is None
    assert isinstance(found.text_match, TemplateMatch)
    assert found.text_match.end == len("Điều 9.")


def test_generated_marker_only() -> None:
    [found] = find_candidates(PROFILE, numbered("Nội dung khoản", "1."))
    assert found.level == "khoan"
    assert found.text_match is None
    assert isinstance(found.list_match, TemplateMatch)
    assert found.list_match.source == "list_marker"
    assert found.list_match.end == 2


@pytest.mark.parametrize("displayed", [None, ""])
def test_list_item_that_displays_nothing(displayed: str | None) -> None:
    assert find_candidates(PROFILE, numbered("Nội dung", displayed)) == ()


def test_list_item_whose_marker_no_level_reads() -> None:
    assert find_candidates(PROFILE, numbered("Nội dung", "A.")) == ()


def test_list_marker_must_match_whole() -> None:
    # "1.1." is multi-level numbering, not "{arabic}." followed by more.
    assert find_candidates(PROFILE, numbered("Nội dung", "1.1.")) == ()


def test_typed_marker_inside_a_list_item_with_no_displayed_marker() -> None:
    assert summary(find_candidates(PROFILE, numbered("Điều 3. X", None))) == [("dieu", None, (("arabic", 3),))]


# ============================================================ both sources


def test_same_level_and_number_is_one_candidate() -> None:
    # Word shows "1." and the author also typed "1.": one heading, both matches kept.
    [found] = find_candidates(PROFILE, numbered("1. Nội dung", "1."))
    assert found.level == "khoan"
    assert number_of(found.list_match) == ARABIC_1
    assert number_of(found.text_match) == ARABIC_1


def test_merged_text_match_says_where_the_typed_marker_ends() -> None:
    # The title starts after the typed duplicate, not at offset 0.
    text = "1. Nội dung"
    [found] = find_candidates(PROFILE, numbered(text, "1."))
    assert found.text_match is not None
    assert text[found.text_match.end :].strip() == "Nội dung"


def test_merge_ignores_letter_case_in_words() -> None:
    [found] = find_candidates(PROFILE, numbered("ĐIỀU 1. Phạm vi", "Điều 1."))
    assert (found.level, found.list_match is not None, found.text_match is not None) == ("dieu", True, True)


def test_same_level_different_numbers_are_two_candidates() -> None:
    assert summary(find_candidates(PROFILE, numbered("1. Nội dung", "2."))) == [
        ("khoan", ARABIC_2, None),
        ("khoan", None, ARABIC_1),
    ]


def test_different_levels_are_separate_candidates_in_rank_order() -> None:
    # Word numbering says khoản, the typed text says Điều; the builder decides.
    assert summary(find_candidates(PROFILE, numbered("Điều 3. Phạm vi", "1."))) == [
        ("dieu", None, (("arabic", 3),)),
        ("khoan", ARABIC_1, None),
    ]


def test_glyphs_of_the_same_style_merge() -> None:
    [found] = find_candidates(PROFILE, bulleted("- Nội dung", "–"))
    assert found.level == "gach"
    assert number_of(found.list_match) == "dash"
    assert number_of(found.text_match) == "dash"
    assert isinstance(found.list_match, GlyphMatch) and found.list_match.glyph == "–"
    assert isinstance(found.text_match, GlyphMatch) and found.text_match.glyph == "-"


# ============================================================ several readings


ROMAN_1 = (("lower_roman", 1),)
ROMAN_2 = (("lower_roman", 2),)
LETTER_I = (("lower_vn", 9),)  # a b c d đ e g h i: the Vietnamese alphabet has no f


def test_each_reading_is_its_own_candidate() -> None:
    found = summary(find_candidates(PROFILE, paragraph("i) Nội dung")))
    assert found == [("item", None, ROMAN_1), ("item", None, LETTER_I)]


def test_readings_pair_by_style_and_value() -> None:
    # Roman pairs with roman and letter with letter, never across.
    found = summary(find_candidates(PROFILE, numbered("i) Nội dung", "i)")))
    assert found == [("item", ROMAN_1, ROMAN_1), ("item", LETTER_I, LETTER_I)]


def test_readings_that_disagree_do_not_pair() -> None:
    # Word shows "ii)" (roman only: letter styles read one letter), the text says "i)".
    found = summary(find_candidates(PROFILE, numbered("i) Nội dung", "ii)")))
    assert found == [("item", ROMAN_2, None), ("item", None, ROMAN_1), ("item", None, LETTER_I)]


def test_both_readings_of_i() -> None:
    readings = PROFILE.get_marker("item").match_text("i) x")
    assert {number_of(r) for r in readings} == {ROMAN_1, LETTER_I}


# ============================================================ word_bullet fallback


BULLET = LevelSpec(name="bullet", marker=glyphs(WORD_BULLET))
TICK = LevelSpec(name="tick", marker=glyphs("check", WORD_BULLET))


def test_unclaimed_word_bullet_falls_back() -> None:
    [found] = find_candidates(profile(DIEU, GACH, BULLET), bulleted("Nội dung", "•"))
    assert found == Candidate(
        level="bullet",
        list_match=GlyphMatch(source="list_marker", style=WORD_BULLET, glyph="•", end=1),
        text_match=None,
    )


def test_fallback_keeps_an_unmapped_symbol_as_it_is() -> None:
    [found] = find_candidates(profile(DIEU, BULLET), bulleted("Nội dung", ""))
    assert isinstance(found.list_match, GlyphMatch)
    assert (found.list_match.glyph, found.list_match.end) == ("", 1)


@pytest.mark.parametrize("displayed", [None, ""])
def test_fallback_for_a_bullet_that_displays_nothing(displayed: str | None) -> None:
    [found] = find_candidates(profile(DIEU, BULLET), bulleted("Nội dung", displayed))
    assert found.list_match == GlyphMatch(source="list_marker", style=WORD_BULLET, glyph="", end=0)


def test_a_bullet_read_by_a_named_style_does_not_fall_back() -> None:
    [found] = find_candidates(profile(DIEU, TICK), bulleted("Đạt", "✓"))
    assert found.level == "tick"
    assert isinstance(found.list_match, GlyphMatch) and found.list_match.style == "check"


def test_the_fallback_level_takes_other_bullets() -> None:
    [found] = find_candidates(profile(DIEU, TICK), bulleted("Nội dung", "•"))
    assert found.level == "tick"
    assert isinstance(found.list_match, GlyphMatch) and found.list_match.style == WORD_BULLET


def test_a_bullet_read_by_another_level_does_not_fall_back() -> None:
    assert summary(find_candidates(profile(DIEU, GACH, BULLET), bulleted("Nội dung", "-"))) == [("gach", "dash", None)]


def test_a_text_reading_does_not_claim_the_bullet() -> None:
    # A typed "- x" inside a "•" list: both are reported; the builder prefers the text.
    found = summary(find_candidates(profile(DIEU, GACH, BULLET), bulleted("- Nội dung", "•")))
    assert found == [("gach", None, "dash"), ("bullet", WORD_BULLET, None)]


def test_numbered_lists_never_fall_back() -> None:
    assert find_candidates(profile(DIEU, BULLET), numbered("Nội dung", "A.")) == ()


def test_typed_bullets_never_fall_back() -> None:
    # word_bullet is about Word lists; a typed "•" in a plain paragraph is not one.
    assert find_candidates(profile(DIEU, BULLET), paragraph("• Nội dung")) == ()


def none_list(text: str) -> Any:
    """A heading in a Word list that numbers nothing (num_fmt "none"), as in the real document."""
    info = make_list_info(
        displayed_marker=None,
        is_ordered=False,
        declared_numbering=make_declared_numbering(num_fmt="none", segment_formats=()),
    )
    return make_block(block_type=BlockType.LIST_ITEM, text=text, list_info=info)


def test_a_list_that_numbers_nothing_is_not_a_bullet_list() -> None:
    # is_ordered is false here too, but nothing is displayed and the format is "none".
    assert summary(find_candidates(profile(DIEU, BULLET), none_list("Điều 1. Phạm vi"))) == [("dieu", None, (("arabic", 1),))]
    assert find_candidates(profile(DIEU, BULLET), none_list("QUY ĐỊNH CHUNG")) == ()


def test_without_a_declared_format_an_unordered_list_counts_as_bullets() -> None:
    # Sources that declare no numbering (not docx) only say whether the list is ordered.
    info = make_list_info(displayed_marker="•", is_ordered=False, declared_numbering=None)
    block = make_block(block_type=BlockType.LIST_ITEM, text="Nội dung", list_info=info)
    assert summary(find_candidates(profile(DIEU, BULLET), block)) == [("bullet", WORD_BULLET, None)]


def test_no_fallback_without_a_word_bullet_level() -> None:
    assert find_candidates(profile(DIEU, GACH), bulleted("Nội dung", "•")) == ()


def test_fallback_candidate_is_placed_at_its_rank() -> None:
    found = summary(find_candidates(profile(DIEU, BULLET, GACH), bulleted("Điều 2. X", "•")))
    assert [level for level, _, _ in found] == ["dieu", "bullet"]
    found = summary(find_candidates(profile(BULLET, DIEU), bulleted("Điều 2. X", "•")))
    assert [level for level, _, _ in found] == ["bullet", "dieu"]


# ============================================================ order


def test_candidates_are_in_rank_order() -> None:
    # Reverse the profile: the same block gives the same candidates, reversed.
    khoan = LevelSpec(name="khoan", marker=templates("{arabic}."))  # no inside, so it can come first
    block = numbered("Điều 3. Phạm vi", "1.")
    forward = [c.level for c in find_candidates(profile(DIEU, khoan), block)]
    backward = [c.level for c in find_candidates(profile(khoan, DIEU), block)]
    assert forward == ["dieu", "khoan"]
    assert backward == ["khoan", "dieu"]


def test_returns_a_tuple() -> None:
    assert isinstance(find_candidates(PROFILE, paragraph("x")), tuple)


# ============================================================ VN_REGULATION


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        (paragraph("Chương I"), [("chuong", None, (("upper_roman", 1),))]),
        (paragraph("Điều 9. Phạm vi điều chỉnh"), [("dieu", None, (("arabic", 9),))]),
        (numbered("Nội dung khoản", "1."), [("khoan", (("arabic", 1),), None)]),
        (
            numbered("a) Nội dung", "a)"),
            [("diem", (("lower_latin", 1),), (("lower_latin", 1),)), ("diem", (("lower_vn", 1),), (("lower_vn", 1),))],
        ),
        (numbered("Được hỗ trợ", "f)"), [("diem", (("lower_latin", 6),), None)]),
        (paragraph("- Nội dung"), [("gach", None, "dash")]),
        (paragraph("Chương trình đào tạo"), []),
        # A typed table-of-contents entry under a Word bullet: read from the text;
        # VN_REGULATION has no word_bullet level, so the bullet itself gives nothing.
        (bulleted("1.1. Mục tiêu", "◦"), [("khoan_cap_2", None, (("arabic", 1), ("arabic", 1)))]),
        (bulleted("Nội dung", "✓"), []),
    ],
)
def test_vn_regulation(block: Any, expected: list[tuple[str, object, object]]) -> None:
    assert summary(find_candidates(VN_REGULATION, block)) == expected