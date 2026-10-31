"""Tests for logical/profile.py: Profile.

A Profile holds the levels in order (the position is the rank), compiles every
level's marker once when it is built, and checks what needs more than one level:
unique names, `inside` naming earlier levels, sections ranking above items, a
template or glyph style used by one level only, and `word_bullet` resolved to at
most one level. Lookups by
level name raise KeyError for unknown names.
"""

import dataclasses
import unicodedata
from types import MappingProxyType

import pytest

from document_processor.logical.counter_styles import (
    BUILTIN_COUNTER_STYLES,
    alphabet_style,
)
from document_processor.logical.glyph_marker import GlyphMarker
from document_processor.logical.glyph_styles import BUILTIN_GLYPH_STYLES, GlyphStyle
from document_processor.logical.levels import (
    GlyphMarkerSpec,
    LevelSpec,
    TemplateMarkerSpec,
)
from document_processor.logical.profile import Profile
from document_processor.logical.template_marker import TemplateMarker


def templates(*items: str, case_sensitive: bool = False) -> TemplateMarkerSpec:
    return TemplateMarkerSpec(templates=items, case_sensitive=case_sensitive)


def glyphs(*names: str) -> GlyphMarkerSpec:
    return GlyphMarkerSpec(glyphs=names)


CHUONG = LevelSpec(
    name="chuong",
    is_section=True,
    marker=templates("Chương {upper_roman}"),
    title="next_block",
)
DIEU = LevelSpec(
    name="dieu",
    is_section=True,
    marker=templates("Điều {arabic}."),
    title="rest_of_paragraph",
)
KHOAN = LevelSpec(
    name="khoan", is_section=False, marker=templates("{arabic}."), inside=("dieu",)
)
GACH = LevelSpec(
    name="gach", is_section=False, marker=glyphs("dash"), closes_on_plain_paragraph=True
)
KHOAN_ANYWHERE = LevelSpec(
    name="khoan", is_section=False, marker=templates("{arabic}.")
)  # no inside


def make(
    levels: tuple[LevelSpec, ...] | None = (CHUONG, DIEU, KHOAN, GACH),
    **overrides: object,
) -> Profile:
    fields: dict[str, object] = {
        "name": "test",
        "version": "1",
        "levels": levels,
        "end_matter": None,
    }
    fields.update(overrides)
    return Profile(**fields)  # type: ignore[arg-type]


def first_text_match(profile: Profile, level: str, text: str) -> object:
    found = profile.get_marker(level).match_text(text)
    return found[0] if found else None


# ============================================================ fields


def test_keeps_its_fields() -> None:
    end = templates("Nơi nhận:")
    profile = make(end_matter=end)
    assert profile.name == "test"
    assert profile.version == "1"
    assert profile.levels == (CHUONG, DIEU, KHOAN, GACH)
    assert profile.end_matter == end


def test_registries_default_to_the_builtins() -> None:
    profile = make()
    assert profile.counter_styles is BUILTIN_COUNTER_STYLES
    assert profile.glyph_styles is BUILTIN_GLYPH_STYLES


def test_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make().name = "other"  # type: ignore[misc]


def test_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        Profile("test", "1", (DIEU,), None)  # type: ignore[call-arg]


def test_equal_profiles_compare_equal_and_hash_alike() -> None:
    assert make() == make()
    assert hash(make()) == hash(make())


def test_registries_and_compiled_markers_are_left_out_of_equality() -> None:
    extended = BUILTIN_COUNTER_STYLES.extend({})
    assert make(counter_styles=extended) == make()


def test_profiles_with_different_levels_differ() -> None:
    assert make(levels=(DIEU,)) != make(levels=(DIEU, KHOAN))


def test_repr_hides_the_compiled_fields() -> None:
    text = repr(make())
    assert "_marker_by_name" not in text
    assert "_rank_by_name" not in text


# ---------- name and version


@pytest.mark.parametrize("name", ["test", "vn_regulation", "x", "profile2"])
def test_valid_names(name: str) -> None:
    assert make(name=name).name == name


@pytest.mark.parametrize(
    "name", ["", "VN", "vn-regulation", "vn regulation", "2vn", "_vn", "quy_định"]
)
def test_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="invalid profile name"):
        make(name=name)


def test_version_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="version must not be empty"):
        make(version="")


# ============================================================ levels


def test_levels_none_means_no_structure() -> None:
    profile = make(levels=None)
    assert profile.levels is None
    assert profile.word_bullet_level is None
    with pytest.raises(KeyError):
        profile.get_level("dieu")


def test_levels_none_may_still_have_end_matter() -> None:
    profile = make(levels=None, end_matter=templates("Nơi nhận:"))
    assert profile.end_matter_marker is not None


def test_empty_levels_are_refused() -> None:
    # One way to say "no structure": levels=None.
    with pytest.raises(ValueError, match="levels"):
        make(levels=())


def test_rank_is_the_position_in_levels() -> None:
    profile = make()
    assert [profile.get_rank(n) for n in ("chuong", "dieu", "khoan", "gach")] == [
        0,
        1,
        2,
        3,
    ]


def test_reordering_levels_changes_the_ranks() -> None:
    profile = make(levels=(DIEU, CHUONG))
    assert profile.get_rank("dieu") == 0
    assert profile.get_rank("chuong") == 1


def test_get_level_returns_the_spec() -> None:
    assert make().get_level("khoan") is KHOAN


def test_duplicate_level_name() -> None:
    other_dieu = LevelSpec(
        name="dieu", is_section=True, marker=templates("Article {arabic}.")
    )
    with pytest.raises(
        ValueError, match="profile 'test', level 'dieu': duplicate level name"
    ):
        make(levels=(DIEU, other_dieu))


# ---------- inside


def test_inside_may_name_an_earlier_level() -> None:
    assert make(levels=(DIEU, KHOAN)).get_level("khoan").inside == ("dieu",)


def test_inside_may_name_any_earlier_level_not_only_the_one_just_above() -> None:
    diem = LevelSpec(
        name="diem", is_section=False, marker=templates("{lower_vn})"), inside=("dieu",)
    )
    make(levels=(CHUONG, DIEU, KHOAN, diem))


def test_inside_may_name_several_levels() -> None:
    khoan = LevelSpec(
        name="khoan",
        is_section=False,
        marker=templates("{arabic}."),
        inside=("chuong", "dieu"),
    )
    make(levels=(CHUONG, DIEU, khoan))


def test_inside_unknown_level() -> None:
    khoan = LevelSpec(
        name="khoan", is_section=False, marker=templates("{arabic}."), inside=("dieuu",)
    )
    with pytest.raises(
        ValueError, match="level 'khoan': inside names unknown level 'dieuu'"
    ):
        make(levels=(DIEU, khoan))


def test_inside_a_later_level() -> None:
    dieu = LevelSpec(
        name="dieu",
        is_section=True,
        marker=templates("Điều {arabic}."),
        inside=("khoan",),
    )
    with pytest.raises(
        ValueError,
        match="level 'dieu': inside may only name earlier levels, got 'khoan'",
    ):
        make(levels=(dieu, KHOAN))


def test_inside_itself() -> None:
    dieu = LevelSpec(
        name="dieu",
        is_section=True,
        marker=templates("Điều {arabic}."),
        inside=("dieu",),
    )
    with pytest.raises(
        ValueError, match="inside may only name earlier levels, got 'dieu'"
    ):
        make(levels=(CHUONG, dieu))


def test_first_level_cannot_be_inside_anything() -> None:
    chuong = LevelSpec(
        name="chuong",
        is_section=True,
        marker=templates("Chương {upper_roman}"),
        inside=("dieu",),
    )
    with pytest.raises(ValueError, match="earlier levels"):
        make(levels=(chuong, DIEU))


# ============================================================ markers


def test_template_levels_get_a_template_marker() -> None:
    assert isinstance(make().get_marker("dieu"), TemplateMarker)


def test_glyph_levels_get_a_glyph_marker() -> None:
    assert isinstance(make().get_marker("gach"), GlyphMarker)


def test_compiled_markers_match() -> None:
    profile = make()
    assert first_text_match(profile, "chuong", "Chương IV") is not None
    assert first_text_match(profile, "dieu", "Điều 9. Phạm vi") is not None
    assert first_text_match(profile, "khoan", "1. Nội dung") is not None
    assert first_text_match(profile, "gach", "- Nội dung") is not None
    assert first_text_match(profile, "dieu", "Điều kiện") is None


def test_case_sensitive_is_passed_to_the_compiler() -> None:
    insensitive = make(
        levels=(
            LevelSpec(name="dieu", is_section=True, marker=templates("Điều {arabic}.")),
        )
    )
    sensitive = make(
        levels=(
            LevelSpec(
                name="dieu",
                is_section=True,
                marker=templates("Điều {arabic}.", case_sensitive=True),
            ),
        )
    )
    assert first_text_match(insensitive, "dieu", "ĐIỀU 9. X") is not None
    assert first_text_match(sensitive, "dieu", "ĐIỀU 9. X") is None


def test_every_template_of_a_level_is_compiled() -> None:
    profile = make(
        levels=(
            LevelSpec(
                name="diem",
                is_section=False,
                marker=templates("{lower_vn})", "{lower_vn}."),
            ),
        )
    )
    assert first_text_match(profile, "diem", "a) x") is not None
    assert first_text_match(profile, "diem", "b. y") is not None


# ---------- one template, one level


def test_same_template_in_two_levels() -> None:
    other = LevelSpec(name="muc", is_section=True, marker=templates("Điều {arabic}."))
    with pytest.raises(
        ValueError,
        match="level 'muc': 'Điều \\{arabic\\}.' already used by level 'dieu'",
    ):
        make(levels=(DIEU, other))


def test_same_template_among_others_in_two_levels() -> None:
    a = LevelSpec(
        name="a", is_section=False, marker=templates("{arabic}.", "{arabic})")
    )
    b = LevelSpec(
        name="b", is_section=False, marker=templates("({arabic})", "{arabic})")
    )
    with pytest.raises(ValueError, match="already used by level 'a'"):
        make(levels=(a, b))


def test_same_template_in_nfc_and_nfd_is_still_the_same() -> None:
    nfd = unicodedata.normalize("NFD", "Điều {arabic}.")
    assert nfd != "Điều {arabic}."
    other = LevelSpec(name="muc", is_section=True, marker=templates(nfd))
    with pytest.raises(ValueError, match="already used by level 'dieu'"):
        make(levels=(DIEU, other))


def test_different_templates_that_overlap_are_allowed() -> None:
    # Only identical templates are refused; partial overlaps are left to pass 2
    # and the test matrix.
    a = LevelSpec(name="a", is_section=False, marker=templates("{arabic}."))
    b = LevelSpec(name="b", is_section=False, marker=templates("{lower_roman|arabic}."))
    make(levels=(a, b))


def test_a_template_and_a_glyph_style_with_the_same_text_do_not_clash() -> None:
    a = LevelSpec(name="a", is_section=False, marker=templates("dash {arabic}"))
    b = LevelSpec(name="b", is_section=False, marker=glyphs("dash"))
    c = LevelSpec(name="c", is_section=False, marker=templates("dash"))
    make(levels=(a, b, c))


# ---------- one glyph style, one level


def test_same_glyph_style_in_two_levels() -> None:
    other = LevelSpec(name="item", is_section=False, marker=glyphs("plus", "dash"))
    with pytest.raises(
        ValueError, match="level 'item': 'dash' already used by level 'gach'"
    ):
        make(levels=(GACH, other))


def test_different_glyph_styles_in_two_levels() -> None:
    cong = LevelSpec(name="cong", is_section=False, marker=glyphs("plus"))
    profile = make(levels=(GACH, cong))
    assert first_text_match(profile, "gach", "+ x") is None
    assert first_text_match(profile, "cong", "+ x") is not None


# ---------- word_bullet


def test_no_word_bullet_level_by_default() -> None:
    assert make().word_bullet_level is None


def test_word_bullet_alone() -> None:
    bullet = LevelSpec(name="bullet", is_section=False, marker=glyphs("word_bullet"))
    profile = make(levels=(DIEU, bullet))
    assert profile.word_bullet_level == "bullet"
    marker = profile.get_marker("bullet")
    assert isinstance(marker, GlyphMarker)
    assert dict(marker.by_glyph) == {}


def test_word_bullet_alone_matches_nothing_itself() -> None:
    # Unclaimed Word bullets are given to this level by the classifier, not by the marker.
    profile = make(
        levels=(
            LevelSpec(name="bullet", is_section=False, marker=glyphs("word_bullet")),
        )
    )
    marker = profile.get_marker("bullet")
    assert isinstance(marker, GlyphMarker)
    for text in ["• x", "- x", "✓ x"]:
        assert marker.match_text(text) == []
    for displayed in ["•", "◦", "▪", "✓", ""]:
        assert marker.match_list_marker(displayed) == []


def test_word_bullet_with_other_styles() -> None:
    item = LevelSpec(
        name="item", is_section=False, marker=glyphs("check", "word_bullet")
    )
    profile = make(levels=(item,))
    assert profile.word_bullet_level == "item"
    marker = profile.get_marker("item")
    assert isinstance(marker, GlyphMarker)
    assert set(marker.by_glyph) == set(BUILTIN_GLYPH_STYLES.get("check").glyphs)


def test_word_bullet_in_two_levels() -> None:
    a = LevelSpec(name="a", is_section=False, marker=glyphs("dash", "word_bullet"))
    b = LevelSpec(name="b", is_section=False, marker=glyphs("word_bullet"))
    with pytest.raises(
        ValueError, match="level 'b': 'word_bullet' already used by level 'a'"
    ):
        make(levels=(a, b))


# ---------- compile errors name the profile and the level


@pytest.mark.parametrize(
    ("marker", "detail"),
    [
        (templates("Điều {arabik}."), "unknown counter style 'arabik'"),
        (templates("Điều {arabic"), "Điều \\{arabic"),
        (templates("{arabic}"), "\\{arabic\\}"),
        (glyphs("dahs"), "unknown glyph style 'dahs'"),
    ],
)
def test_compile_errors_name_the_profile_and_the_level(
    marker: TemplateMarkerSpec | GlyphMarkerSpec, detail: str
) -> None:
    with pytest.raises(
        ValueError, match=f"^profile 'test', level 'bad': .*{detail}"
    ) as excinfo:
        make(levels=(DIEU, LevelSpec(name="bad", is_section=False, marker=marker)))
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_word_bullet_is_not_passed_to_the_glyph_compiler() -> None:
    # compile_glyph_marker refuses "word_bullet"; the profile takes it out first.
    make(
        levels=(
            LevelSpec(
                name="item", is_section=False, marker=glyphs("dash", "word_bullet")
            ),
        )
    )


# ============================================================ end matter


def test_no_end_matter() -> None:
    assert make().end_matter_marker is None


def test_end_matter_is_compiled() -> None:
    profile = make(end_matter=templates("Nơi nhận:"))
    assert isinstance(profile.end_matter_marker, TemplateMarker)
    assert profile.end_matter_marker.match_text("Nơi nhận:")
    assert profile.end_matter_marker.match_text("NƠI NHẬN:")
    assert not profile.end_matter_marker.match_text("Nơi nhận hồ sơ")


def test_end_matter_respects_case_sensitive() -> None:
    profile = make(end_matter=templates("Nơi nhận:", case_sensitive=True))
    assert profile.end_matter_marker is not None
    assert not profile.end_matter_marker.match_text("NƠI NHẬN:")


def test_end_matter_may_repeat_a_level_template() -> None:
    # End matter is not a level; it is matched separately.
    make(end_matter=templates("Điều {arabic}."))


def test_end_matter_compile_error_names_the_profile() -> None:
    with pytest.raises(
        ValueError, match="^profile 'test', end_matter: .*unknown counter style 'nope'"
    ):
        make(end_matter=templates("Phụ lục {nope}"))


# ============================================================ registries


def test_profile_counter_styles_are_used() -> None:
    upper_greek = alphabet_style("upper_greek", "ΑΒΓΔΕ")
    styles = BUILTIN_COUNTER_STYLES.extend({"upper_greek": upper_greek})
    level = LevelSpec(
        name="part", is_section=False, marker=templates("Part {upper_greek}")
    )
    profile = make(levels=(level,), counter_styles=styles)
    assert first_text_match(profile, "part", "Part Γ") is not None


def test_builtin_counter_styles_do_not_know_profile_styles() -> None:
    level = LevelSpec(
        name="part", is_section=False, marker=templates("Part {upper_greek}")
    )
    with pytest.raises(ValueError, match="unknown counter style 'upper_greek'"):
        make(levels=(level,))


def test_profile_glyph_styles_are_used() -> None:
    pointing = GlyphStyle(name="pointing_hand", glyphs=frozenset({"☞"}))
    styles = BUILTIN_GLYPH_STYLES.extend({"pointing_hand": pointing})
    level = LevelSpec(name="note", is_section=False, marker=glyphs("pointing_hand"))
    profile = make(levels=(level,), glyph_styles=styles)
    assert first_text_match(profile, "note", "☞ Xem thêm") is not None


def test_end_matter_uses_the_profile_counter_styles() -> None:
    upper_greek = alphabet_style("upper_greek", "ΑΒΓΔΕ")
    styles = BUILTIN_COUNTER_STYLES.extend({"upper_greek": upper_greek})
    profile = make(
        end_matter=templates("Appendix {upper_greek}"), counter_styles=styles
    )
    assert profile.end_matter_marker is not None
    assert profile.end_matter_marker.match_text("Appendix Β")


# ============================================================ lookups


@pytest.mark.parametrize("lookup", ["get_level", "get_rank", "get_marker"])
def test_unknown_level_raises_key_error(lookup: str) -> None:
    with pytest.raises(KeyError) as excinfo:
        getattr(make(), lookup)("dieuu")
    message = str(excinfo.value)
    assert "'dieuu'" in message
    assert "'test'" in message
    for name in ("chuong", "dieu", "khoan", "gach"):
        assert name in message


@pytest.mark.parametrize("lookup", ["get_level", "get_rank", "get_marker"])
def test_lookups_are_case_sensitive(lookup: str) -> None:
    with pytest.raises(KeyError):
        getattr(make(), lookup)("Dieu")


def test_compiled_tables_are_read_only() -> None:
    profile = make()
    assert isinstance(profile._marker_by_name, MappingProxyType)
    assert isinstance(profile._rank_by_name, MappingProxyType)
    assert isinstance(profile._level_by_name, MappingProxyType)


# ============================================================ sections rank above items
#
# A node's children are always deeper levels, so this keeps an item (a numbered
# paragraph) from ever holding a section (a cited unit like Điều).


def test_sections_then_items_is_accepted() -> None:
    assert [level.is_section for level in make().levels or ()] == [
        True,
        True,
        False,
        False,
    ]


def test_all_sections_is_accepted() -> None:
    assert make(levels=(CHUONG, DIEU)).levels == (CHUONG, DIEU)


def test_all_items_is_accepted() -> None:
    assert make(levels=(KHOAN_ANYWHERE, GACH)).levels == (KHOAN_ANYWHERE, GACH)


def test_a_section_below_an_item_is_rejected() -> None:
    phan = LevelSpec(
        name="phan", is_section=False, marker=templates("Phần {upper_roman}")
    )
    with pytest.raises(
        ValueError,
        match=r"^profile 'test', level 'chuong': a section cannot rank below an item level \('phan'\)$",
    ):
        make(levels=(phan, CHUONG, DIEU))


def test_the_error_names_the_first_item_above() -> None:
    # Two items above Điều: the message points at the first one, where the order went wrong.
    with pytest.raises(ValueError, match=r"level 'dieu': .*\('khoan'\)"):
        make(levels=(KHOAN_ANYWHERE, GACH, DIEU))


def test_a_section_after_items_and_sections_is_rejected() -> None:
    late = LevelSpec(name="muc", is_section=True, marker=templates("Mục {arabic}"))
    with pytest.raises(
        ValueError, match="level 'muc': a section cannot rank below an item level"
    ):
        make(levels=(CHUONG, DIEU, KHOAN, late))
