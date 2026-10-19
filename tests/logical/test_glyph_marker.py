"""Tests for logical/glyph_marker.py: compile_glyph_marker and GlyphMarker.

A glyph marker is one level's marker when the level is declared with glyph
styles (`glyphs: [dash]`). It reads one glyph:

* match_text: at the start of a paragraph, followed by at least one blank
  ("- Nội dung"); the rest of the text is the item's content.
* match_list_marker: the whole Word list marker ("•" from a bullet list).

Styles never share a glyph, so there is at most one reading (None when there is
none). One optional variation selector (U+FE0E / U+FE0F) after the glyph belongs
to it: "✔️" is one glyph two characters long.
"""

import dataclasses

import pytest

from document_processor.logical.glyph_marker import GlyphMarker, GlyphMatch, compile_glyph_marker
from document_processor.logical.glyph_styles import BUILTIN_GLYPH_STYLES, GlyphStyle
from document_processor.logical.registry import Registry

VS15 = "︎"  # text presentation
VS16 = "️"  # emoji presentation


def marker(*names: str) -> GlyphMarker:
    return compile_glyph_marker(list(names), BUILTIN_GLYPH_STYLES)


def style_of_text(names: tuple[str, ...], text: str) -> str | None:
    found = marker(*names).match_text(text)
    return None if found is None else found.style


# ============================================================ compile_glyph_marker


def test_returns_a_glyph_marker() -> None:
    assert isinstance(marker("dash"), GlyphMarker)


def test_maps_every_glyph_of_every_chosen_style_to_its_style() -> None:
    m = marker("dash", "check")
    dash = BUILTIN_GLYPH_STYLES.get("dash")
    check = BUILTIN_GLYPH_STYLES.get("check")
    assert dict(m.by_glyph) == {**{g: dash for g in dash.glyphs}, **{g: check for g in check.glyphs}}


def test_only_the_chosen_styles_are_mapped() -> None:
    assert set(marker("dash").by_glyph) == set(BUILTIN_GLYPH_STYLES.get("dash").glyphs)


def test_by_glyph_is_read_only() -> None:
    m = marker("dash")
    with pytest.raises(TypeError):
        m.by_glyph["+"] = BUILTIN_GLYPH_STYLES.get("plus")  # type: ignore[index]


def test_glyph_marker_is_frozen() -> None:
    m = marker("dash")
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.by_glyph = {}  # type: ignore[misc]


def test_listing_a_style_twice_is_harmless() -> None:
    assert dict(marker("dash", "dash").by_glyph) == dict(marker("dash").by_glyph)


def test_uses_the_registry_it_is_given() -> None:
    pointing = GlyphStyle(name="pointing_hand", glyphs=frozenset({"☞"}))
    styles = BUILTIN_GLYPH_STYLES.extend({"pointing_hand": pointing})
    m = compile_glyph_marker(["pointing_hand"], styles)
    found = m.match_text("☞ Xem thêm")
    assert found is not None and found.style == "pointing_hand"


def test_unknown_name_in_the_given_registry() -> None:
    empty: Registry[GlyphStyle] = Registry("glyph style", {})
    with pytest.raises(ValueError, match="unknown glyph style 'dash'"):
        compile_glyph_marker(["dash"], empty)


def test_empty_list() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compile_glyph_marker([], BUILTIN_GLYPH_STYLES)


def test_a_single_string_is_refused() -> None:
    # A str is a sequence of characters; each would be looked up as a style name.
    with pytest.raises(ValueError, match="not one string"):
        compile_glyph_marker("dash", BUILTIN_GLYPH_STYLES)  # type: ignore[arg-type]


def test_word_bullet_is_reserved_for_the_profile() -> None:
    with pytest.raises(ValueError, match="reserved for the profile"):
        compile_glyph_marker(["dash", "word_bullet"], BUILTIN_GLYPH_STYLES)


def test_unknown_style_lists_the_known_ones() -> None:
    with pytest.raises(ValueError) as excinfo:
        compile_glyph_marker(["dash", "dahs"], BUILTIN_GLYPH_STYLES)
    message = str(excinfo.value)
    assert "dahs" in message
    assert "known: arrow, check, dash" in message


def test_unknown_style_error_is_a_value_error_not_a_key_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        compile_glyph_marker(["dahs"], BUILTIN_GLYPH_STYLES)
    assert not isinstance(excinfo.value, KeyError)


# ============================================================ match_text


def test_full_text_result() -> None:
    assert marker("dash").match_text("- Nội dung") == GlyphMatch(source="text", style="dash", glyph="-", end=1)


@pytest.mark.parametrize(
    ("name", "glyph"),
    [(name, glyph) for name in BUILTIN_GLYPH_STYLES.names() for glyph in sorted(BUILTIN_GLYPH_STYLES.get(name).glyphs)],
)
def test_every_builtin_glyph_is_read_in_text(name: str, glyph: str) -> None:
    found = marker(name).match_text(f"{glyph} Nội dung")
    assert found == GlyphMatch(source="text", style=name, glyph=glyph, end=1)


def test_end_splits_the_marker_from_the_content() -> None:
    text = "– Kiến trúc tổng thể"
    found = marker("dash").match_text(text)
    assert found is not None
    assert text[: found.end] == "–"
    assert text[found.end :].lstrip() == "Kiến trúc tổng thể"


def test_look_alikes_are_the_same_style() -> None:
    for glyph in ["-", "‐", "‑", "–", "—", "−"]:
        assert style_of_text(("dash",), f"{glyph} Nội dung") == "dash", repr(glyph)


def test_a_level_with_several_styles_reports_which_one() -> None:
    assert style_of_text(("dash", "plus"), "- Nội dung") == "dash"
    assert style_of_text(("dash", "plus"), "+ Nội dung") == "plus"


@pytest.mark.parametrize("blank", [" ", "  ", "\t", " ", " ", "\n"])
def test_any_whitespace_after_the_glyph(blank: str) -> None:
    assert marker("dash").match_text(f"-{blank}Nội dung") is not None


@pytest.mark.parametrize(
    "text",
    [
        "-5%",  # a negative number
        "-Nội dung",  # no blank
        "--",  # two dashes
        "-- Nội dung",
        "->",  # an arrow typed with two characters
        "-",  # the glyph alone: nothing follows
        "",
        " - Nội dung",  # leading blank: the glyph must be first
        "Nội dung - tiếp",  # glyph in the middle
        "+ Nội dung",  # a glyph of a style this level did not choose
        "• Nội dung",
        "* Nội dung",  # punctuation that is in no style
        "o Nội dung",  # the letter o is never a glyph in text
    ],
)
def test_text_without_a_marker(text: str) -> None:
    assert marker("dash").match_text(text) is None


def test_only_the_first_character_counts() -> None:
    found = marker("dash", "plus").match_text("- + Nội dung")
    assert found is not None and found.style == "dash"
    assert found.end == 1


# ---------- variation selectors


@pytest.mark.parametrize("selector", [VS15, VS16])
def test_variation_selector_belongs_to_the_glyph(selector: str) -> None:
    text = f"✔{selector} Đạt"
    found = marker("check").match_text(text)
    assert found == GlyphMatch(source="text", style="check", glyph="✔", end=2)
    assert text[found.end :].lstrip() == "Đạt"


def test_glyph_is_the_base_character_without_the_selector() -> None:
    found = marker("check").match_text(f"☑{VS16} Đạt")
    assert found is not None and found.glyph == "☑"


def test_variation_selector_still_needs_a_blank_after_it() -> None:
    assert marker("check").match_text(f"✔{VS16}Đạt") is None
    assert marker("check").match_text(f"✔{VS16}") is None


def test_only_one_variation_selector_is_taken() -> None:
    assert marker("check").match_text(f"✔{VS16}{VS16} Đạt") is None


def test_variation_selector_alone_is_not_a_glyph() -> None:
    assert marker("check").match_text(f"{VS16} Đạt") is None


# ============================================================ match_list_marker


def test_full_list_marker_result() -> None:
    assert marker("round_bullet").match_list_marker("•") == GlyphMatch(
        source="list_marker", style="round_bullet", glyph="•", end=1
    )


@pytest.mark.parametrize(
    ("names", "displayed", "style"),
    [
        # What the docx adapter produces for Word's default bullet list:
        (("round_bullet",), "•", "round_bullet"),  # Symbol 0xB7
        (("white_bullet",), "◦", "white_bullet"),  # Courier New "o"
        (("square",), "▪", "square"),  # Wingdings 0xA7
        # The check-mark bullet in the real document (Wingdings 0xFC):
        (("check",), "✓", "check"),
        # Google Docs' defaults:
        (("round_bullet", "white_bullet", "square"), "●", "round_bullet"),
        (("round_bullet", "white_bullet", "square"), "○", "white_bullet"),
        (("round_bullet", "white_bullet", "square"), "■", "square"),
        # A Word list whose bullet is a dash:
        (("dash",), "-", "dash"),
        (("dash",), "–", "dash"),
    ],
)
def test_list_markers_from_real_sources(names: tuple[str, ...], displayed: str, style: str) -> None:
    found = marker(*names).match_list_marker(displayed)
    assert found is not None
    assert found.style == style
    assert found.source == "list_marker"
    assert found.end == len(displayed)


@pytest.mark.parametrize("selector", [VS15, VS16])
def test_list_marker_with_a_variation_selector(selector: str) -> None:
    assert marker("check").match_list_marker(f"✔{selector}") == GlyphMatch(
        source="list_marker", style="check", glyph="✔", end=2
    )


@pytest.mark.parametrize(
    "displayed",
    [
        "",
        "• ",  # a trailing blank is not part of the glyph
        " •",
        "••",
        "•-",
        "1.",  # a numbered marker
        "a)",
        "o",  # Courier New "o" the adapter did not map
        "",  # a symbol-font byte the adapter did not map
        "✗",  # mapped by the adapter, but in no style
        "+",  # a glyph of a style this level did not choose
        f"•{VS16}{VS16}",
    ],
)
def test_list_markers_that_are_not_this_levels_glyph(displayed: str) -> None:
    assert marker("round_bullet", "dash").match_list_marker(displayed) is None


def test_list_marker_ignores_the_blank_rule() -> None:
    # The whole marker is the glyph; nothing has to follow it.
    assert marker("dash").match_list_marker("-") is not None
    assert marker("dash").match_text("-") is None


def test_source_tells_the_two_methods_apart() -> None:
    from_text = marker("dash").match_text("- Nội dung")
    from_list = marker("dash").match_list_marker("-")
    assert from_text is not None and from_list is not None
    assert (from_text.source, from_list.source) == ("text", "list_marker")
    assert (from_text.style, from_text.glyph, from_text.end) == (from_list.style, from_list.glyph, from_list.end)