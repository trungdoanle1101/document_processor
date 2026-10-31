"""Tests for logical/glyph_styles.py: GlyphStyle and the built-in glyph styles.

A glyph style is a named set of characters that mark the same kind of list item:
"-" and "–" are both `dash`. Glyphs are single punctuation or symbol characters,
so letters, digits, spaces, controls and unmapped private-use bytes can never be
glyphs. Styles never share a glyph, so one character always means one style.
"""

import dataclasses
import unicodedata

import pytest

from document_processor.logical.glyph_styles import BUILTIN_GLYPH_STYLES, GlyphStyle


def make_style(
    name: str = "test_style", glyphs: frozenset[str] = frozenset({"-"})
) -> GlyphStyle:
    return GlyphStyle(name=name, glyphs=glyphs)


# ============================================================ GlyphStyle


def test_glyph_style_keeps_its_fields() -> None:
    style = make_style(name="dashes", glyphs=frozenset({"-", "–"}))
    assert style.name == "dashes"
    assert style.glyphs == frozenset({"-", "–"})


def test_glyph_style_is_frozen() -> None:
    style = make_style()
    with pytest.raises(dataclasses.FrozenInstanceError):
        style.name = "other"  # type: ignore[misc]


def test_glyph_style_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        GlyphStyle("dash", frozenset({"-"}))  # type: ignore[call-arg]


# ---------- name


@pytest.mark.parametrize("name", ["dash", "round_bullet", "x", "style2"])
def test_valid_names(name: str) -> None:
    assert make_style(name=name).name == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Dash",
        "1dash",
        "_dash",
        "round-bullet",
        "round bullet",
        "chữ",
        "word_bullet ",
    ],
)
def test_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        make_style(name=name)


# ---------- glyphs


def test_glyphs_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        make_style(glyphs=frozenset())


@pytest.mark.parametrize("glyph", ["", "--", "->", "✓️"])
def test_each_glyph_is_exactly_one_character(glyph: str) -> None:
    with pytest.raises(ValueError, match="exactly one character"):
        make_style(glyphs=frozenset({"-", glyph}))


@pytest.mark.parametrize(
    "glyph",
    [
        "-",  # Pd
        "–",  # Pd en dash
        "•",  # Po bullet
        "*",  # Po
        "§",  # Po section sign
        "+",  # Sm
        "→",  # Sm arrow
        "●",  # So black circle
        "✓",  # So check mark
        "★",  # So black star (Google Docs)
        "➔",  # So heavy arrow (Google Docs)
        "❏",  # So drop-shadowed square (Google Docs)
        "(",  # Ps
        ")",  # Pe
    ],
)
def test_punctuation_and_symbols_are_glyphs(glyph: str) -> None:
    assert glyph in make_style(glyphs=frozenset({glyph})).glyphs


@pytest.mark.parametrize(
    ("glyph", "why"),
    [
        ("o", "letter: Word's Courier New bullet must be mapped to ◦ first"),
        ("x", "letter"),
        ("đ", "letter"),
        ("1", "digit"),
        ("①", "circled digit, category No"),
        (" ", "space"),
        (" ", "no-break space"),
        ("\t", "tab"),
        ("\x01", "control"),
        ("​", "zero-width space, category Cf"),
        ("️", "variation selector, a combining mark"),
        ("́", "combining acute accent"),
        ("", "private use: an unmapped symbol-font byte"),
        ("͸", "unassigned"),
    ],
)
def test_other_characters_are_not_glyphs(glyph: str, why: str) -> None:
    with pytest.raises(ValueError, match="punctuation mark or a symbol"):
        make_style(glyphs=frozenset({glyph}))


def test_error_names_the_style_the_glyph_and_its_code_point() -> None:
    with pytest.raises(ValueError) as excinfo:
        make_style(name="bullets", glyphs=frozenset({"-", "o"}))
    message = str(excinfo.value)
    assert "'bullets'" in message
    assert "'o'" in message
    assert "U+006F" in message


def test_the_reported_glyph_is_the_same_on_every_run() -> None:
    # Several bad glyphs: the first in sorted order is reported, not a random one.
    with pytest.raises(ValueError, match="'1'"):
        make_style(glyphs=frozenset({"z", "a", "1", "-"}))


# ============================================================ built-in styles


BUILTIN_NAMES = (
    "arrow",
    "check",
    "dash",
    "diamond",
    "plus",
    "round_bullet",
    "square",
    "white_bullet",
)


def test_builtin_names() -> None:
    assert BUILTIN_GLYPH_STYLES.names() == BUILTIN_NAMES


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_builtin_is_registered_under_its_own_name(name: str) -> None:
    assert BUILTIN_GLYPH_STYLES.get(name).name == name


def test_word_bullet_is_not_a_style() -> None:
    # word_bullet is a reserved name the profile resolves, not a set of glyphs.
    assert "word_bullet" not in BUILTIN_GLYPH_STYLES


def test_unknown_style_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="glyph style 'dahs'.*known: arrow, check, dash"):
        BUILTIN_GLYPH_STYLES.get("dahs")


def test_builtin_styles_do_not_overlap() -> None:
    owner: dict[str, str] = {}
    for name in BUILTIN_GLYPH_STYLES.names():
        for glyph in BUILTIN_GLYPH_STYLES.get(name).glyphs:
            assert glyph not in owner, f"{glyph!r} is in both {owner[glyph]} and {name}"
            owner[glyph] = name


@pytest.mark.parametrize(
    ("name", "glyphs"),
    [
        ("dash", {"-", "‐", "‑", "–", "—", "−"}),
        ("plus", {"+", "＋"}),
        ("round_bullet", {"•", "●"}),
        ("white_bullet", {"◦", "○"}),
        ("square", {"▪", "■", "□", "❑"}),
        ("check", {"✓", "✔", "☑"}),
        ("arrow", {"➢", "➤", "►", "→"}),
        ("diamond", {"❖", "◆", "◇"}),
    ],
)
def test_builtin_glyphs(name: str, glyphs: set[str]) -> None:
    assert BUILTIN_GLYPH_STYLES.get(name).glyphs == frozenset(glyphs)


def style_of(glyph: str) -> str | None:
    for name in BUILTIN_GLYPH_STYLES.names():
        if glyph in BUILTIN_GLYPH_STYLES.get(name).glyphs:
            return name
    return None


@pytest.mark.parametrize(
    ("glyph", "style"),
    [
        # What the docx adapter produces for Word's bullet library:
        ("•", "round_bullet"),  # Symbol 0xB7, Word's default level-1 bullet
        ("◦", "white_bullet"),  # Courier New "o", Word's default level-2 bullet
        ("▪", "square"),  # Wingdings 0xA7, Word's default level-3 bullet
        ("❖", "diamond"),  # Wingdings 0x76
        ("➢", "arrow"),  # Wingdings 0xD8
        ("✓", "check"),  # Wingdings 0xFC, the check mark in the real document
        ("☑", "check"),  # Wingdings 0xFE
        ("●", "round_bullet"),  # Wingdings 0x6C
        ("■", "square"),  # Wingdings 0x6E
        ("❑", "square"),  # Wingdings 0x71
        # Google Docs' default bullets:
        ("●", "round_bullet"),
        ("○", "white_bullet"),
        ("■", "square"),
        # Typed markers:
        ("-", "dash"),
        ("–", "dash"),
        ("+", "plus"),
    ],
)
def test_bullets_from_real_sources_have_a_style(glyph: str, style: str) -> None:
    assert style_of(glyph) == style


@pytest.mark.parametrize("glyph", ["✗", "☒"])
def test_cross_marks_have_no_style_yet(glyph: str) -> None:
    # Wingdings 0xFB (✗) and 0xFD (☒) are mapped by the adapter but belong to no style;
    # only word_bullet takes them for now.
    assert style_of(glyph) is None


def test_every_builtin_glyph_is_punctuation_or_a_symbol() -> None:
    for name in BUILTIN_GLYPH_STYLES.names():
        for glyph in BUILTIN_GLYPH_STYLES.get(name).glyphs:
            assert unicodedata.category(glyph)[0] in "PS", (name, glyph)
