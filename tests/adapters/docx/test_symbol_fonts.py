"""Tests for adapters/docx/symbol_fonts.py: symbol-font characters -> Unicode.

Word stores a Symbol or Wingdings bullet as the font's own byte, usually in the
private-use form U+F0xx (U+F0FC is Wingdings 0xFC). Older files may store the
plain byte (U+00FC, "ü"). Both forms map to the same Unicode character.

Courier New is not a symbol font, but Word's default level-2 bullet is the
letter "o" in Courier New, drawn as a white bullet; it maps to "◦".
"""

import pytest

from document_processor.adapters.docx.symbol_fonts import map_symbol_text

# ---------- the two characters seen in real files


def test_wingdings_check_mark_in_private_use_form() -> None:
    # The check-mark bullet from the real document: lvlText U+F0FC, font Wingdings.
    assert map_symbol_text("\uf0fc", "Wingdings") == "✓"


def test_symbol_bullet_in_private_use_form() -> None:
    # Word's default bullet: lvlText U+F0B7, font Symbol.
    assert map_symbol_text("\uf0b7", "Symbol") == "•"


# ---------- both storage forms


@pytest.mark.parametrize(
    ("text", "font", "expected"),
    [
        ("\uf0fc", "Wingdings", "✓"),
        ("\u00fc", "Wingdings", "✓"),  # plain byte 0xFC ("ü")
        ("\uf0b7", "Symbol", "•"),
        ("\u00b7", "Symbol", "•"),  # plain byte 0xB7 ("·")
    ],
    ids=["wingdings-pua", "wingdings-byte", "symbol-pua", "symbol-byte"],
)
def test_private_use_and_plain_byte_forms_map_the_same(
    text: str, font: str, expected: str
) -> None:
    assert map_symbol_text(text, font) == expected


# ---------- Word's bullet library and check boxes


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0xA7, "▪"),
        (0x76, "❖"),
        (0xD8, "➢"),
        (0xFC, "✓"),
        (0xFB, "✗"),
        (0xFE, "☑"),
        (0xFD, "☒"),
        (0x6C, "●"),
        (0x6E, "■"),
        (0x71, "❑"),
    ],
)
def test_wingdings_table(code: int, expected: str) -> None:
    assert map_symbol_text(chr(0xF000 + code), "Wingdings") == expected


# ---------- font names


@pytest.mark.parametrize("font", ["Wingdings", "wingdings", "WINGDINGS", "WingDings"])
def test_font_name_ignores_case(font: str) -> None:
    assert map_symbol_text("\uf0fc", font) == "✓"


@pytest.mark.parametrize("font", [None, ""], ids=["none", "empty"])
def test_no_font_leaves_text_unchanged(font: str | None) -> None:
    assert map_symbol_text("\uf0fc", font) == "\uf0fc"


@pytest.mark.parametrize(
    "font", ["Arial", "Times New Roman", "Wingdings 2", "Webdings"]
)
def test_unknown_font_leaves_text_unchanged(font: str) -> None:
    # Fonts without a table are left alone; generated_bullets can still catch them.
    assert map_symbol_text("\uf0fc", font) == "\uf0fc"


def test_fonts_have_separate_tables() -> None:
    # Symbol 0xFC is not a check mark.
    assert map_symbol_text("\uf0fc", "Symbol") == "\uf0fc"
    assert map_symbol_text("\uf0b7", "Wingdings") == "\uf0b7"


# ---------- characters outside the table


def test_unknown_code_in_a_known_font_is_unchanged() -> None:
    assert map_symbol_text("\uf021", "Wingdings") == "\uf021"


def test_ordinary_characters_are_unchanged() -> None:
    # "-" and "o" are not in the Wingdings or Symbol tables.
    assert map_symbol_text("-", "Wingdings") == "-"
    assert map_symbol_text("o", "Symbol") == "o"
    assert map_symbol_text("o", "Wingdings") == "o"


def test_private_use_outside_the_f0xx_block_is_unchanged() -> None:
    # Only U+F000..U+F0FF is Word's symbol-font range.
    assert map_symbol_text("\uf1fc", "Wingdings") == "\uf1fc"
    assert map_symbol_text("\ue0fc", "Wingdings") == "\ue0fc"


def test_every_character_is_mapped() -> None:
    # A bullet lvlText is usually one character, but nothing stops it being longer.
    assert map_symbol_text("\uf0fc \uf0fe", "Wingdings") == "✓ ☑"


def test_empty_text() -> None:
    assert map_symbol_text("", "Wingdings") == ""


# ---------- Courier New: Word's default level-2 bullet


def test_courier_new_o_is_a_white_bullet() -> None:
    assert map_symbol_text("o", "Courier New") == "◦"


@pytest.mark.parametrize("font", ["Courier New", "courier new", "COURIER NEW"])
def test_courier_new_font_name_ignores_case(font: str) -> None:
    assert map_symbol_text("o", font) == "◦"


@pytest.mark.parametrize("font", ["Arial", "Times New Roman", "Courier", "CourierNew"])
def test_o_in_other_fonts_is_unchanged(font: str) -> None:
    assert map_symbol_text("o", font) == "o"


@pytest.mark.parametrize("text", ["-", "O", "·", "•"])
def test_courier_new_only_maps_lowercase_o(text: str) -> None:
    assert map_symbol_text(text, "Courier New") == text
