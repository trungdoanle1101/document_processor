"""Tests for logical/levels.py: the marker specs and LevelSpec.

A spec is what the profile author writes: names and strings only, never compiled
markers. Types are checked by mypy; these classes only check meaning (not empty,
no repeats, a valid name, a known title rule). Checks that need the other levels
(what `inside` names, a glyph style used twice) belong to Profile.
"""

import dataclasses

import pytest

from document_processor.logical.counter_styles import BUILTIN_COUNTER_STYLES
from document_processor.logical.glyph_marker import (
    GlyphMarker,
    GlyphMatch,
    compile_glyph_marker,
)
from document_processor.logical.glyph_styles import BUILTIN_GLYPH_STYLES
from document_processor.logical.levels import (
    TITLE_RULES,
    GlyphMarkerSpec,
    LevelSpec,
    Marker,
    MarkerMatch,
    MarkerSpec,
    TemplateMarkerSpec,
    TitleRule,
)
from document_processor.logical.template_marker import (
    TemplateMarker,
    TemplateMatch,
    compile_template_marker,
)

DIEU = TemplateMarkerSpec(templates=("Điều {arabic}.",))
DASH = GlyphMarkerSpec(glyphs=("dash",))


def level(**overrides: object) -> LevelSpec:
    fields: dict[str, object] = {"name": "dieu", "marker": DIEU, "is_section": True}
    fields.update(overrides)
    return LevelSpec(**fields)  # type: ignore[arg-type]


# ============================================================ type aliases


def test_marker_spec_is_either_kind() -> None:
    assert MarkerSpec.__value__ == TemplateMarkerSpec | GlyphMarkerSpec


def test_marker_is_either_compiled_kind() -> None:
    assert Marker.__value__ == TemplateMarker | GlyphMarker


def test_marker_match_is_either_kind_of_match() -> None:
    assert MarkerMatch.__value__ == TemplateMatch | GlyphMatch


def test_compiled_markers_are_markers() -> None:
    template = compile_template_marker(list(DIEU.templates), BUILTIN_COUNTER_STYLES)
    glyph = compile_glyph_marker(list(DASH.glyphs), BUILTIN_GLYPH_STYLES)
    assert isinstance(template, Marker.__value__)
    assert isinstance(glyph, Marker.__value__)


def test_title_rules() -> None:
    assert TITLE_RULES == ("none", "rest_of_paragraph", "next_block")


def test_title_rules_come_from_the_literal() -> None:
    assert TITLE_RULES == TitleRule.__value__.__args__


# ============================================================ TemplateMarkerSpec


def test_template_spec_keeps_its_fields() -> None:
    spec = TemplateMarkerSpec(
        templates=("Điều {arabic}.", "Điều {arabic}:"), case_sensitive=True
    )
    assert spec.templates == ("Điều {arabic}.", "Điều {arabic}:")
    assert spec.case_sensitive is True


def test_template_spec_is_case_insensitive_by_default() -> None:
    assert DIEU.case_sensitive is False


def test_template_spec_does_not_parse_its_templates() -> None:
    # Syntax and style names are checked when the profile compiles the marker,
    # where the registry is known; the spec only holds the strings.
    spec = TemplateMarkerSpec(templates=("Điều {no_such_style}.", "{"))
    assert spec.templates == ("Điều {no_such_style}.", "{")


def test_template_spec_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TemplateMarkerSpec(templates=())


def test_template_spec_must_not_repeat() -> None:
    with pytest.raises(ValueError, match="repeat"):
        TemplateMarkerSpec(
            templates=("Điều {arabic}.", "Điều {arabic}:", "Điều {arabic}.")
        )


def test_template_repeat_error_shows_the_templates() -> None:
    with pytest.raises(ValueError, match=r"'\{lower_vn\}\)'"):
        TemplateMarkerSpec(templates=("{lower_vn})", "{lower_vn})"))


def test_templates_differing_only_in_case_are_different() -> None:
    spec = TemplateMarkerSpec(templates=("MỞ ĐẦU", "Mở đầu"), case_sensitive=True)
    assert len(spec.templates) == 2


def test_template_spec_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DIEU.case_sensitive = True  # type: ignore[misc]


def test_template_spec_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        TemplateMarkerSpec(("Điều {arabic}.",))  # type: ignore[call-arg]


def test_template_specs_compare_by_value_and_hash() -> None:
    same = TemplateMarkerSpec(templates=("Điều {arabic}.",))
    assert same == DIEU
    assert hash(same) == hash(DIEU)
    assert (
        TemplateMarkerSpec(templates=("Điều {arabic}.",), case_sensitive=True) != DIEU
    )


# ============================================================ GlyphMarkerSpec


def test_glyph_spec_keeps_its_fields() -> None:
    assert GlyphMarkerSpec(glyphs=("dash", "plus")).glyphs == ("dash", "plus")


def test_glyph_spec_does_not_look_up_its_styles() -> None:
    # Unknown names and the reserved word_bullet are resolved by the profile.
    assert GlyphMarkerSpec(glyphs=("word_bullet", "no_such_style")).glyphs == (
        "word_bullet",
        "no_such_style",
    )


def test_glyph_spec_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        GlyphMarkerSpec(glyphs=())


def test_glyph_spec_must_not_repeat() -> None:
    with pytest.raises(ValueError, match="repeat"):
        GlyphMarkerSpec(glyphs=("dash", "plus", "dash"))


def test_glyph_spec_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DASH.glyphs = ("plus",)  # type: ignore[misc]


def test_glyph_spec_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        GlyphMarkerSpec(("dash",))  # type: ignore[call-arg]


def test_glyph_specs_compare_by_value_and_hash() -> None:
    assert GlyphMarkerSpec(glyphs=("dash",)) == DASH
    assert hash(GlyphMarkerSpec(glyphs=("dash",))) == hash(DASH)


def test_the_two_kinds_are_never_equal() -> None:
    assert TemplateMarkerSpec(templates=("dash",)) != GlyphMarkerSpec(glyphs=("dash",))  # type: ignore[comparison-overlap]


# ============================================================ LevelSpec


def test_level_defaults() -> None:
    spec = LevelSpec(name="dieu", is_section=True, marker=DIEU)
    assert spec.title == "none"
    assert spec.closes_on_plain_paragraph is False
    assert spec.inside == ()


def test_level_keeps_its_fields() -> None:
    spec = LevelSpec(
        name="khoan",
        is_section=False,
        marker=TemplateMarkerSpec(templates=("{arabic}.",)),
        title="rest_of_paragraph",
        closes_on_plain_paragraph=True,
        inside=("dieu",),
    )
    assert spec.name == "khoan"
    assert spec.marker == TemplateMarkerSpec(templates=("{arabic}.",))
    assert spec.title == "rest_of_paragraph"
    assert spec.closes_on_plain_paragraph is True
    assert spec.inside == ("dieu",)


def test_level_with_a_glyph_marker() -> None:
    assert LevelSpec(name="gach", is_section=False, marker=DASH).marker == DASH


def test_level_has_no_rank() -> None:
    # The position in Profile.levels is the rank.
    field_names = {f.name for f in dataclasses.fields(LevelSpec)}
    assert field_names == {
        "name",
        "marker",
        "is_section",
        "title",
        "closes_on_plain_paragraph",
        "inside",
    }


def test_is_section_is_required() -> None:
    # Every level states whether it is a section (Chương, Điều) or an item (khoản, gạch); no default.
    with pytest.raises(TypeError, match="is_section"):
        LevelSpec(name="dieu", marker=DIEU)  # type: ignore[call-arg]


@pytest.mark.parametrize("is_section", [True, False])
def test_is_section_is_kept(is_section: bool) -> None:
    assert level(is_section=is_section).is_section is is_section


def test_level_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        level().name = "khoan"  # type: ignore[misc]


def test_level_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        LevelSpec("dieu", DIEU)  # type: ignore[call-arg]


def test_levels_compare_by_value_and_hash() -> None:
    a = LevelSpec(name="khoan", is_section=False, marker=DIEU, inside=("dieu",))
    b = LevelSpec(
        name="khoan",
        is_section=False,
        marker=TemplateMarkerSpec(templates=("Điều {arabic}.",)),
        inside=("dieu",),
    )
    assert a == b
    assert hash(a) == hash(b)


# ---------- name


@pytest.mark.parametrize("name", ["dieu", "khoan", "x", "level2", "diem_con"])
def test_valid_level_names(name: str) -> None:
    assert level(name=name).name == name


@pytest.mark.parametrize(
    "name", ["", "Dieu", "điều", "2dieu", "_dieu", "diem-con", "diem con", "dieu "]
)
def test_invalid_level_names(name: str) -> None:
    with pytest.raises(ValueError, match="invalid level name"):
        level(name=name)


def test_invalid_name_error_shows_the_name_and_the_rule() -> None:
    with pytest.raises(ValueError) as excinfo:
        level(name="Điều")
    message = str(excinfo.value)
    assert "'Điều'" in message
    assert "[a-z][a-z0-9_]*" in message


# ---------- title


@pytest.mark.parametrize("title", ["none", "rest_of_paragraph", "next_block"])
def test_every_title_rule_is_accepted(title: str) -> None:
    assert level(title=title).title == title


@pytest.mark.parametrize(
    "title", ["", "None", "next_paragraph", "nxt_block", "rest-of-paragraph", " none"]
)
def test_unknown_title_rule(title: str) -> None:
    # mypy rejects these in Python profiles (title is a Literal); YAML gets this message.
    with pytest.raises(ValueError, match="title"):
        level(title=title)


def test_unknown_title_error_names_the_level_the_value_and_the_choices() -> None:
    with pytest.raises(ValueError) as excinfo:
        level(name="chuong", title="next_paragraph")
    message = str(excinfo.value)
    assert "'chuong'" in message
    assert "'next_paragraph'" in message
    for rule in TITLE_RULES:
        assert rule in message


def test_name_is_checked_before_title() -> None:
    # The title error quotes the level name, so a bad name is reported first.
    with pytest.raises(ValueError, match="invalid level name"):
        level(name="Bad", title="bad")


# ---------- inside


def test_inside_names_are_not_checked_here() -> None:
    # Whether `inside` names real, earlier levels is checked by Profile.
    assert level(name="khoan", inside=("no_such_level",)).inside == ("no_such_level",)


def test_inside_may_name_several_levels() -> None:
    # Any of them may be an open ancestor.
    assert level(name="khoan", inside=("dieu", "muc")).inside == ("dieu", "muc")
