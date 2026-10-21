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
from document_processor.logical.candidates import (
    BlockCandidates,
    Candidate,
    ClassifiedBlock,
    choose_headings,
    classify_blocks,
    count_kept_open_by_heading,
    count_kept_open_by_plain_paragraph,
    find_candidates,
    narrow_candidates,
)
from document_processor.logical.glyph_marker import GlyphMatch
from document_processor.logical.levels import GlyphMarkerSpec, LevelSpec, MarkerMatch, TemplateMarkerSpec
from document_processor.logical.markers import WORD_BULLET
from document_processor.logical.profile import Profile
from document_processor.logical.profiles.vn_regulation import VN_REGULATION
from document_processor.logical.template_marker import TemplateMatch
from tests.factories import (
    make_block,
    make_declared_numbering,
    make_image_info,
    make_list_info,
    make_table_block,
    make_table_cell,
    make_table_info,
)


def templates(*items: str) -> TemplateMarkerSpec:
    return TemplateMarkerSpec(templates=items)


def glyphs(*names: str) -> GlyphMarkerSpec:
    return GlyphMarkerSpec(glyphs=names)


DIEU = LevelSpec(name="dieu", marker=templates("Điều {arabic}."))
KHOAN = LevelSpec(name="khoan", marker=templates("{arabic}."), inside=("dieu",))
ITEM = LevelSpec(name="item", marker=templates("{lower_roman|lower_vn})"))  # "i)" has two readings
GACH = LevelSpec(name="gach", marker=glyphs("dash"))
KHOAN_ANYWHERE = LevelSpec(name="khoan", marker=templates("{arabic}."))


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


# ============================================================ narrow_candidates (pass 2)


LETTER_POINT = LevelSpec(name="diem", marker=templates("{lower_latin|lower_vn})"))  # Latin preferred
ROMAN_POINT = LevelSpec(name="point", marker=templates("({lower_roman|lower_latin})"))  # roman preferred


def listed(text: str, displayed: str, num_fmt: str) -> Any:
    """A list item whose Word list declares `num_fmt` ("decimal", "lowerLetter", "none", ...)."""
    info = make_list_info(
        displayed_marker=displayed,
        is_ordered=num_fmt not in ("bullet", "none"),
        declared_numbering=make_declared_numbering(num_fmt=num_fmt, segment_formats=(num_fmt,)),
    )
    return make_block(block_type=BlockType.LIST_ITEM, text=text, list_info=info)


def narrowed(test_profile: Profile, *blocks: Any) -> list[BlockCandidates]:
    items = [BlockCandidates(block=b, candidates=find_candidates(test_profile, b)) for b in blocks]
    return list(narrow_candidates(items))


def chosen(items: list[BlockCandidates]) -> list[list[tuple[str, str, int] | str]]:
    """Per block: (level, style, value) of each kept candidate's counting numeral, or the level alone."""
    out: list[list[tuple[str, str, int] | str]] = []
    for item in items:
        row: list[tuple[str, str, int] | str] = []
        for c in item.candidates:
            match = c.list_match or c.text_match
            if isinstance(match, TemplateMatch) and match.numerals:
                last = match.numerals[-1]
                row.append((c.level, last.style, last.value))
            else:
                row.append(c.level)
        out.append(row)
    return out


def styles(items: list[BlockCandidates], level: str) -> list[str]:
    """The style chosen at `level`, block by block (blocks without that level are skipped)."""
    return [entry[1] for row in chosen(items) for entry in row if isinstance(entry, tuple) and entry[0] == level]


# ---------- shape


def test_narrow_keeps_every_block_in_order() -> None:
    blocks = [paragraph("Nghị định này quy định"), make_table_block(), paragraph("Điều 1. X"), paragraph("a) Y")]
    items = narrowed(profile(DIEU, LETTER_POINT), *blocks)
    assert [item.block for item in items] == blocks
    assert all(item.block is block for item, block in zip(items, blocks))


def test_narrow_keeps_blocks_without_candidates_empty() -> None:
    items = narrowed(PROFILE, paragraph("Nội dung"), make_table_block())
    assert [item.candidates for item in items] == [(), ()]


def test_narrow_returns_a_tuple() -> None:
    assert narrow_candidates([]) == ()
    assert isinstance(narrow_candidates([BlockCandidates(block=paragraph("x"), candidates=())]), tuple)


def test_single_candidates_are_kept_as_they_are() -> None:
    block = numbered("1. Nội dung", "1.")
    [before] = find_candidates(PROFILE, block)
    [item] = narrowed(PROFILE, block)
    assert item.candidates == (before,)


def test_candidates_of_different_levels_are_all_kept() -> None:
    # Choosing between levels is the builder's job.
    [item] = narrowed(PROFILE, numbered("Điều 3. Phạm vi", "1."))
    assert [c.level for c in item.candidates] == ["dieu", "khoan"]


def test_at_most_one_candidate_per_level() -> None:
    items = narrowed(profile(LETTER_POINT, ROMAN_POINT), paragraph("a) x"), paragraph("(i) y"), paragraph("(c) z"))
    for item in items:
        levels = [c.level for c in item.candidates]
        assert len(levels) == len(set(levels))


def test_kept_candidates_are_the_ones_from_pass_1() -> None:
    block = paragraph("a) x")
    before = find_candidates(profile(LETTER_POINT), block)
    [item] = narrowed(profile(LETTER_POINT), block)
    assert item.candidates[0] in before


# ---------- letters: Latin or Vietnamese


def test_letters_default_to_the_first_style_in_the_slot() -> None:
    items = narrowed(profile(LETTER_POINT), *[paragraph(f"{x}) Nội dung") for x in "abcd"])
    assert styles(items, "diem") == ["lower_latin"] * 4


def test_dj_switches_the_count_to_vietnamese() -> None:
    # a b c d đ e: after đ (vn 5), only the Vietnamese e (6) continues.
    items = narrowed(profile(LETTER_POINT), *[paragraph(f"{x}) Nội dung") for x in "abcdđe"])
    assert chosen(items)[4:] == [[("diem", "lower_vn", 5)], [("diem", "lower_vn", 6)]]


def test_latin_e_continues_after_d() -> None:
    # a b c d e f: e is Latin 5 (continues d = 4), f is Latin only.
    items = narrowed(profile(LETTER_POINT), *[paragraph(f"{x}) Nội dung") for x in "abcdef"])
    assert chosen(items)[4:] == [[("diem", "lower_latin", 5)], [("diem", "lower_latin", 6)]]


def test_a_new_list_starts_again_at_a() -> None:
    blocks = [paragraph(f"{x}) X") for x in "abcdđ"] + [paragraph("1. Khoản mới")] + [paragraph(f"{x}) Y") for x in "ab"]
    items = narrowed(profile(KHOAN_ANYWHERE, LETTER_POINT), *blocks)
    # The second list has no đ: it starts again with the preferred alphabet.
    assert styles(items, "diem")[-2:] == ["lower_latin", "lower_latin"]


def test_items_of_other_levels_in_between_do_not_break_the_count() -> None:
    blocks = [paragraph(f"{x}) X") for x in "abcdđ"] + [paragraph("- gạch"), paragraph("- gạch")] + [paragraph("e) Y")]
    items = narrowed(profile(LETTER_POINT, GACH), *blocks)
    assert chosen(items)[-1] == [("diem", "lower_vn", 6)]


# ---------- roman or letter


def test_i_at_the_start_is_roman() -> None:
    items = narrowed(profile(ROMAN_POINT), paragraph("(i) x"), paragraph("(ii) y"), paragraph("(iii) z"))
    assert chosen(items) == [[("point", "lower_roman", 1)], [("point", "lower_roman", 2)], [("point", "lower_roman", 3)]]


def test_i_after_h_is_a_letter() -> None:
    items = narrowed(profile(ROMAN_POINT), *[paragraph(f"({x}) x") for x in "abcdefghi"])
    assert chosen(items)[-1] == [("point", "lower_latin", 9)]
    # c and d are also roman (100, 500), but letters continue from b.
    assert styles(items, "point") == ["lower_latin"] * 9


def test_c_at_the_start_falls_back_to_the_slot_order() -> None:
    # Neither continues nor starts at 1: the first style in the slot (roman) wins.
    [item] = narrowed(profile(ROMAN_POINT), paragraph("(c) x"))
    assert chosen([item]) == [[("point", "lower_roman", 100)]]


# ---------- the format Word declares


def test_declared_lower_letter_keeps_the_letter_reading() -> None:
    # "i)" at the start would be roman 1, but Word says the list counts in letters.
    level = LevelSpec(name="point", marker=templates("{lower_roman|lower_latin})"))
    [item] = narrowed(profile(level), listed("Nội dung", "i)", "lowerLetter"))
    assert chosen([item]) == [[("point", "lower_latin", 9)]]


def test_declared_lower_roman_keeps_the_roman_reading() -> None:
    level = LevelSpec(name="point", marker=templates("{lower_latin|lower_roman})"))  # letter preferred
    [item] = narrowed(profile(level), listed("Nội dung", "c)", "lowerRoman"))
    assert chosen([item]) == [[("point", "lower_roman", 100)]]


def test_declared_format_does_not_touch_text_readings() -> None:
    # The block declares decimal, but the typed "i)" is text: both its readings stay in play.
    level = LevelSpec(name="point", marker=templates("{lower_roman|lower_latin})"))
    [item] = narrowed(profile(level), listed("i) Nội dung", "1.", "decimal"))
    assert chosen([item]) == [[("point", "lower_roman", 1)]]


# ---------- list and text disagree within one level


def test_disagreement_continuing_reading_wins() -> None:
    blocks = [listed("Nội dung", "1.", "decimal"), listed("2. Nội dung", "3.", "decimal")]
    items = narrowed(profile(KHOAN_ANYWHERE), *blocks)
    # Word shows 3, the author typed 2; 2 continues 1.
    assert chosen(items)[1] == [("khoan", "arabic", 2)]
    [kept] = items[1].candidates
    assert kept.text_match is not None and kept.list_match is None


def test_disagreement_counted_list_wins_when_nothing_decides() -> None:
    blocks = [listed("Nội dung", "1.", "decimal"), listed("4. Nội dung", "5.", "decimal")]
    items = narrowed(profile(KHOAN_ANYWHERE), *blocks)
    [kept] = items[1].candidates
    assert kept.list_match is not None and kept.text_match is None


def test_disagreement_text_wins_when_the_list_does_not_count() -> None:
    # A "none" format shows fixed text, not a counter: the typed number is trusted.
    blocks = [listed("Nội dung", "1.", "decimal"), listed("4. Nội dung", "5.", "none")]
    items = narrowed(profile(KHOAN_ANYWHERE), *blocks)
    [kept] = items[1].candidates
    assert kept.text_match is not None and kept.list_match is None


def test_glyphs_prefer_the_typed_one_in_a_bullet_list() -> None:
    # One level with two glyph styles: Word shows "–", the author typed "+".
    item_level = LevelSpec(name="item", marker=glyphs("dash", "plus"))
    [item] = narrowed(profile(item_level), bulleted("+ Nội dung", "–"))
    [kept] = item.candidates
    assert isinstance(kept.text_match, GlyphMatch) and kept.text_match.style == "plus"


def test_typed_glyph_wins_over_the_word_bullet_fallback_of_the_same_level() -> None:
    item_level = LevelSpec(name="item", marker=glyphs("dash", WORD_BULLET))
    [item] = narrowed(profile(item_level), bulleted("- Nội dung", "•"))
    [kept] = item.candidates
    assert kept.text_match is not None and kept.list_match is None


# ---------- VN_REGULATION, a real-looking run


def test_vn_regulation_internal_regulation_letters_stay_latin() -> None:
    blocks = [paragraph("Điều 6. Yêu cầu"), numbered("Nội dung", "4.")] + [paragraph(f"{x}) Nội dung") for x in "abcdefgh"]
    items = narrowed(VN_REGULATION, *blocks)
    assert set(styles(items, "diem")) == {"lower_latin"}


def test_vn_regulation_legal_text_letters_become_vietnamese_from_dj() -> None:
    blocks = [paragraph("Điều 6. Yêu cầu"), numbered("Nội dung", "1.")] + [paragraph(f"{x}) Nội dung") for x in "abcdđeg"]
    items = narrowed(VN_REGULATION, *blocks)
    assert styles(items, "diem") == ["lower_latin"] * 4 + ["lower_vn"] * 3


# ============================================================ choose_headings (pass 3)


CHUONG = LevelSpec(name="chuong", marker=templates("Chương {upper_roman}"), title="next_block")
GACH_CLOSING = LevelSpec(name="gach", marker=glyphs("dash"), closes_on_plain_paragraph=True)
CONG_UNDER_GACH = LevelSpec(name="cong", marker=glyphs("plus"), inside=("gach",), closes_on_plain_paragraph=True)
DIEM_UNDER_KHOAN = LevelSpec(name="diem", marker=templates("{lower_latin})"), inside=("khoan",))
SUB_UNDER_KHOAN = LevelSpec(name="khoan_cap_2", marker=templates("{arabic}.{arabic}"), inside=("khoan",))

STRUCTURE = Profile(
    name="structure",
    version="1",
    levels=(CHUONG, DIEU, KHOAN, SUB_UNDER_KHOAN, DIEM_UNDER_KHOAN, GACH_CLOSING, CONG_UNDER_GACH),
    end_matter=TemplateMarkerSpec(templates=("Nơi nhận:",)),
)


def classify(test_profile: Profile, *blocks: Any) -> list[ClassifiedBlock]:
    return list(classify_blocks(test_profile, blocks))


def headings(classified: list[ClassifiedBlock]) -> list[str | None]:
    return [c.heading.level if c.heading is not None else None for c in classified]


def parts(classified: list[ClassifiedBlock]) -> list[str]:
    return [c.part for c in classified]


def signature_table(text: str) -> Any:
    info = make_table_info(n_rows=1, n_cols=1, n_header_rows=0, cells=(make_table_cell(paragraphs=(text,)),))
    return make_table_block(table_info=info)


# ---------- the open/close helpers


@pytest.mark.parametrize(
    ("open_levels", "heading_level", "kept"),
    [
        ([], "dieu", 0),
        (["chuong", "dieu", "khoan"], "khoan", 2),  # a sibling closes the current khoản
        (["chuong", "dieu", "khoan"], "dieu", 1),  # a new Điều closes Điều and khoản
        (["chuong", "dieu", "khoan"], "chuong", 0),  # a new Chương closes everything
        (["chuong", "dieu", "khoan"], "diem", 3),  # deeper: everything stays open
        (["chuong", "khoan"], "dieu", 1),  # levels may be skipped
    ],
)
def test_count_kept_open_by_heading(open_levels: list[str], heading_level: str, kept: int) -> None:
    assert count_kept_open_by_heading(STRUCTURE, open_levels, heading_level) == kept


@pytest.mark.parametrize(
    ("open_levels", "kept"),
    [
        ([], 0),
        (["dieu", "khoan"], 2),  # nothing closes on a plain paragraph
        (["dieu", "khoan", "gach"], 2),
        (["dieu", "khoan", "gach", "cong"], 2),  # the run of closing levels at the inner end
        (["dieu", "gach", "cong"], 1),
    ],
)
def test_count_kept_open_by_plain_paragraph(open_levels: list[str], kept: int) -> None:
    assert count_kept_open_by_plain_paragraph(STRUCTURE, open_levels) == kept


def test_a_level_that_stays_open_keeps_its_parents_open() -> None:
    # "note" does not close and sits inside "gach", which does: gach cannot close while note is open.
    note = LevelSpec(name="note", marker=templates("Ghi chú {arabic}:"))
    shielded = Profile(name="shielded", version="1", levels=(DIEU, GACH_CLOSING, note), end_matter=None)
    assert count_kept_open_by_plain_paragraph(shielded, ["dieu", "gach", "note"]) == 3


def test_helpers_do_not_change_the_list() -> None:
    open_levels = ["chuong", "dieu", "khoan", "gach"]
    count_kept_open_by_heading(STRUCTURE, open_levels, "dieu")
    count_kept_open_by_plain_paragraph(STRUCTURE, open_levels)
    assert open_levels == ["chuong", "dieu", "khoan", "gach"]


# ---------- shape


def test_choose_headings_keeps_every_block_in_order() -> None:
    blocks = [paragraph("Điều 1. X"), make_table_block(), paragraph("Nội dung")]
    classified = classify(STRUCTURE, *blocks)
    assert [c.block for c in classified] == blocks
    assert all(c.block is b for c, b in zip(classified, blocks))


def test_choose_headings_of_nothing() -> None:
    assert choose_headings(STRUCTURE, []) == ()


def test_the_heading_is_one_of_the_blocks_candidates() -> None:
    block = numbered("Điều 3. Phạm vi", "1.")
    [classified] = classify(STRUCTURE, block)
    assert classified.heading in find_candidates(STRUCTURE, block)


# ---------- inside, then the outermost level


def test_a_simple_run() -> None:
    blocks = [
        paragraph("Chương I"),
        paragraph("QUY ĐỊNH CHUNG"),
        paragraph("Điều 1. Phạm vi"),
        numbered("Nội dung", "1."),
        paragraph("a) Điểm"),
        paragraph("2. Khoản hai"),
    ]
    assert headings(classify(STRUCTURE, *blocks)) == ["chuong", None, "dieu", "khoan", "diem", "khoan"]


def test_inside_rejects_a_khoan_with_no_dieu_open() -> None:
    assert headings(classify(STRUCTURE, paragraph("1. Nội dung"))) == [None]


def test_inside_any_open_ancestor_not_only_the_parent() -> None:
    # điểm needs a khoản; a khoản_cap_2 sits between them and khoản is still open.
    blocks = [paragraph("Điều 1. X"), paragraph("2. Y"), paragraph("2.1 Z"), paragraph("a) W")]
    assert headings(classify(STRUCTURE, *blocks)) == ["dieu", "khoan", "khoan_cap_2", "diem"]


def test_the_outermost_level_wins() -> None:
    # Word numbering says khoản, the typed text says Điều.
    assert headings(classify(STRUCTURE, paragraph("Điều 1. X"), numbered("Điều 3. Phạm vi", "1."))) == ["dieu", "dieu"]


def test_an_outer_candidate_that_fails_inside_gives_way() -> None:
    # Word "1." with text "2.5 triệu…" under a Điều: khoản passes, the sub-clause would need a khoản.
    blocks = [paragraph("Điều 1. X"), numbered("2.5 triệu đồng là mức tối đa", "1.")]
    assert headings(classify(STRUCTURE, *blocks)) == ["dieu", "khoan"]


def test_a_rejected_marker_is_not_a_heading() -> None:
    blocks = [paragraph("Điều 1. X"), paragraph("2.5 triệu đồng là mức tối đa")]
    assert headings(classify(STRUCTURE, *blocks)) == ["dieu", None]


def test_a_new_dieu_closes_the_khoan() -> None:
    blocks = [paragraph("Điều 1. X"), paragraph("1. Y"), paragraph("Điều 2. Z"), paragraph("a) W")]
    # After Điều 2, no khoản is open, so "a)" fails inside.
    assert headings(classify(STRUCTURE, *blocks)) == ["dieu", "khoan", "dieu", None]


# ---------- plain paragraphs close


def test_a_plain_paragraph_closes_dash_items() -> None:
    blocks = [paragraph("Điều 1. X"), paragraph("- gạch"), paragraph("+ cộng"), paragraph("Đoạn kết."), paragraph("+ cộng")]
    # cộng needs an open gạch: allowed before the plain paragraph, rejected after it.
    assert headings(classify(STRUCTURE, *blocks)) == ["dieu", "gach", "cong", None, None]


def test_a_plain_paragraph_does_not_close_a_khoan() -> None:
    blocks = [paragraph("Điều 1. X"), paragraph("1. Y"), paragraph("Đoạn giữa."), paragraph("a) Z")]
    assert headings(classify(STRUCTURE, *blocks)) == ["dieu", "khoan", None, "diem"]


def test_tables_and_images_do_not_close() -> None:
    image = make_block(block_type=BlockType.IMAGE, text="", image_info=make_image_info())
    blocks = [paragraph("Điều 1. X"), paragraph("- gạch"), make_table_block(), image, paragraph("+ cộng")]
    assert headings(classify(STRUCTURE, *blocks)) == ["dieu", "gach", None, None, "cong"]


def test_a_rejected_marker_counts_as_a_plain_paragraph() -> None:
    # "2.5 triệu…" is not a heading (no khoản), so it closes the dash item like any plain text.
    blocks = [paragraph("Điều 1. X"), paragraph("- gạch"), paragraph("2.5 triệu đồng"), paragraph("+ cộng")]
    assert headings(classify(STRUCTURE, *blocks)) == ["dieu", "gach", None, None]


def test_an_unclaimed_word_bullet_counts_as_a_plain_paragraph() -> None:
    blocks = [paragraph("Điều 1. X"), paragraph("- gạch"), bulleted("Ghi chú", "✓"), paragraph("+ cộng")]
    assert headings(classify(STRUCTURE, *blocks)) == ["dieu", "gach", None, None]


# ---------- end matter


def test_end_matter_starts_at_the_first_match() -> None:
    blocks = [paragraph("Điều 1. X"), paragraph("Nơi nhận:"), paragraph("- Như Điều 3;"), paragraph("Điều 9. Y")]
    classified = classify(STRUCTURE, *blocks)
    assert parts(classified) == ["body", "end_matter", "end_matter", "end_matter"]
    assert headings(classified) == ["dieu", None, None, None]  # nothing after it is a heading


def test_end_matter_in_a_signature_table() -> None:
    # In the real internal regulation, "Nơi nhận:" is the first cell of the last table.
    blocks = [paragraph("Điều 27. X"), signature_table("Nơi nhận: - Như Điều 26; - Lưu: VT")]
    assert parts(classify(STRUCTURE, *blocks)) == ["body", "end_matter"]


def test_nothing_is_end_matter_without_a_match() -> None:
    blocks = [paragraph("Điều 1. X"), paragraph("Nơi nhận hồ sơ: Bộ phận một cửa")]
    assert parts(classify(STRUCTURE, *blocks)) == ["body", "body"]


def test_profile_without_end_matter() -> None:
    no_end = Profile(name="no_end", version="1", levels=(DIEU,), end_matter=None)
    assert parts(classify(no_end, paragraph("Điều 1. X"), paragraph("Nơi nhận:"))) == ["body", "body"]


def test_profile_without_levels_still_finds_end_matter() -> None:
    flat = Profile(name="flat", version="1", levels=None, end_matter=TemplateMarkerSpec(templates=("Nơi nhận:",)))
    classified = classify(flat, paragraph("Nội dung"), paragraph("Nơi nhận:"))
    assert headings(classified) == [None, None]
    assert parts(classified) == ["body", "end_matter"]  # no levels: the body starts at once


# ---------- VN_REGULATION, a stretch of the real document


def test_vn_regulation_real_stretch() -> None:
    none_heading = make_declared_numbering(num_fmt="none", segment_formats=())

    def heading_list(text: str, level: int) -> Any:
        info = make_list_info(displayed_marker=None, is_ordered=False, nested_level=level, declared_numbering=none_heading)
        return make_block(block_type=BlockType.LIST_ITEM, text=text, list_info=info)

    blocks = [
        paragraph("Theo đề nghị của Giám đốc Ban Công nghệ,"),  # front matter
        heading_list("Chương III", 6),
        heading_list("KIẾN TRÚC CNTT THÀNH PHẦN", 6),
        heading_list("Mục III", 6),
        heading_list("KIẾN TRÚC DỮ LIỆU", 6),
        heading_list("Điều 9. Mô hình kiến trúc dữ liệu", 0),
        listed("Mô hình", "1.", "decimal"),
        listed("Mô tả các thành phần", "2.", "decimal"),
        paragraph("2.1 Nguồn dữ liệu (Source): Là tập hợp các nguồn dữ liệu"),
        paragraph("2.3.1 Tầng xử lý dữ liệu (Data Processing): Là Khung công nghệ"),
        paragraph("a. Chuẩn bị dữ liệu (Data Preparation)."),
        paragraph("Là một tập hợp các hoạt động cần thiết"),
        paragraph("- Vùng Vector Data: Được sử dụng để lưu trữ"),
        paragraph("2.4.Tầng truy cập dữ liệu (Data Access)"),  # typo: not read
        heading_list("Điều 10. Các yêu cầu đối với kiến trúc dữ liệu", 0),
        signature_table("Nơi nhận: - Như Điều 26; - Lưu: VT"),
    ]
    classified = classify(VN_REGULATION, *blocks)
    assert headings(classified) == [
        None, "chuong", None, "muc", None, "dieu", "khoan", "khoan",
        "khoan_cap_2", "khoan_cap_3", "diem", None, "gach", None, "dieu", None,
    ]
    assert parts(classified) == ["front_matter"] + ["body"] * 14 + ["end_matter"]


def test_end_matter_is_not_looked_for_before_the_body() -> None:
    # A stray "Nơi nhận:" in the front matter must not turn the whole document into end matter.
    blocks = [paragraph("Căn cứ Luật Tổ chức Chính phủ;"), paragraph("Nơi nhận:"), paragraph("Điều 1. X"), paragraph("Nơi nhận:")]
    classified = classify(STRUCTURE, *blocks)
    assert headings(classified) == [None, None, "dieu", None]
    assert parts(classified) == ["front_matter", "front_matter", "body", "end_matter"]


def test_end_matter_ignores_headers_and_footers() -> None:
    blocks = [paragraph("Điều 1. X"), paragraph("Nơi nhận:", region=Region.FOOTER), paragraph("1. Y")]
    classified = classify(STRUCTURE, *blocks)
    assert parts(classified) == ["body", "body", "body"]
    assert headings(classified) == ["dieu", None, "khoan"]



# ---------- front matter, body, end matter


def test_everything_before_the_first_heading_is_front_matter() -> None:
    blocks = [paragraph("NGÂN HÀNG TMCP"), paragraph("QUY ĐỊNH"), paragraph("Điều 1. X"), paragraph("Nội dung")]
    assert parts(classify(STRUCTURE, *blocks)) == ["front_matter", "front_matter", "body", "body"]


def test_the_first_heading_itself_is_body() -> None:
    [classified] = classify(STRUCTURE, paragraph("Điều 1. X"))
    assert classified.part == "body"


def test_a_rejected_marker_before_the_body_stays_front_matter() -> None:
    # "1." with no Điều open is not a heading, so the body has not started.
    assert parts(classify(STRUCTURE, paragraph("1. Căn cứ"), paragraph("Điều 1. X"))) == ["front_matter", "body"]


def test_the_body_stays_started_when_every_level_has_closed() -> None:
    # A dash item closes on a plain paragraph; the open levels are empty again, but this is still the body.
    gach_first = Profile(name="gach_first", version="1", levels=(GACH_CLOSING,), end_matter=None)
    assert parts(classify(gach_first, paragraph("- x"), paragraph("Đoạn."))) == ["body", "body"]


def test_no_heading_at_all_leaves_everything_front_matter() -> None:
    # The builder then starts the body at segment 0 (plan rule); pass 3 just reports what it saw.
    assert parts(classify(STRUCTURE, paragraph("Nội dung"), paragraph("Nơi nhận:"))) == ["front_matter", "front_matter"]


def test_classify_blocks_runs_all_three_passes() -> None:
    # Pass 2 picks Latin for "a)"; pass 3 needs a khoản open for điểm.
    blocks = [paragraph("Điều 1. X"), paragraph("1. Y"), paragraph("a) Z")]
    classified = classify_blocks(STRUCTURE, blocks)
    assert isinstance(classified, tuple)
    assert headings(list(classified)) == ["dieu", "khoan", "diem"]
    diem = classified[2].heading
    assert diem is not None and isinstance(diem.text_match, TemplateMatch)
    assert diem.text_match.numerals[0].style == "lower_latin"