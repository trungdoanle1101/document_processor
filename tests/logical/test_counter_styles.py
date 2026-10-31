"""Tests for logical/counter_styles.py: CounterStyle, alphabet_style, and the built-ins.

A counter style has three jobs:

* ``numeral_pattern``: what a numeral looks like. It is placed inside a template's
  ``{style}`` slot, so it must have no capturing groups or flags and must not
  match empty text. Letter styles match ONE letter.
* ``to_number``: numeral -> number, strict (``None`` if not valid).
* ``to_text``: number -> numeral.

The conversions themselves are tested in tests/core/test_numerals.py. Here we
check that each style is put together correctly: the right pattern with the
right functions, under the right name.
"""

import dataclasses
import re
from collections.abc import Callable

import pytest

from document_processor.core.numerals import (
    LOWER_LATIN,
    LOWER_VN,
    MAX_ROMAN,
    UPPER_LATIN,
    UPPER_VN,
)
from document_processor.logical.counter_styles import (
    BUILTIN_COUNTER_STYLES,
    CounterStyle,
    alphabet_style,
)


def never(text: str) -> int | None:
    return None


def make_style(
    name: str = "test_style",
    numeral_pattern: str = "[0-9]+",
    to_number: Callable[[str], int | None] = never,
    to_text: Callable[[int], str] = str,
) -> CounterStyle:
    return CounterStyle(
        name=name, numeral_pattern=numeral_pattern, to_number=to_number, to_text=to_text
    )


def fullmatches(style: CounterStyle, text: str) -> bool:
    return re.fullmatch(style.numeral_pattern, text) is not None


# ============================================================ CounterStyle


def test_counter_style_keeps_its_fields() -> None:
    style = make_style(
        name="digits", numeral_pattern="[0-9]+", to_number=int, to_text=str
    )
    assert style.name == "digits"
    assert style.numeral_pattern == "[0-9]+"
    assert style.to_number("12") == 12
    assert style.to_text(12) == "12"


def test_counter_style_is_frozen() -> None:
    style = make_style()
    with pytest.raises(dataclasses.FrozenInstanceError):
        style.name = "other"  # type: ignore[misc]


def test_counter_style_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        CounterStyle("digits", "[0-9]+", never, str)  # type: ignore[call-arg]


# ---------- name: must be usable as {name} in a template


@pytest.mark.parametrize(
    "name", ["a", "arabic", "lower_vn", "upper_roman", "x1", "style_2_b"]
)
def test_valid_names(name: str) -> None:
    assert make_style(name=name).name == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Arabic",  # uppercase
        "ARABIC",
        "1st",  # starts with a digit
        "_private",  # starts with an underscore
        "lower-roman",  # hyphen
        "lower roman",  # space
        " arabic",
        "arabic ",
        "a|b",  # the template's alternative separator
        "{arabic}",  # the template's slot braces
        "chữ",  # non-ASCII
        "đ",
    ],
)
def test_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        make_style(name=name)


# ---------- numeral_pattern


@pytest.mark.parametrize(
    "pattern",
    [
        "[0-9]+",
        "[IVXLCDM]+",
        "[abc]",
        "x",
        "(?:ab)+",  # non-capturing groups are fine
        "[0-9]{1,3}",
        "\\d+",
    ],
)
def test_valid_numeral_patterns(pattern: str) -> None:
    assert make_style(numeral_pattern=pattern).numeral_pattern == pattern


@pytest.mark.parametrize("pattern", ["[", "(", "[0-9", "+", "a{2,1}"])
def test_numeral_pattern_must_be_a_valid_regex(pattern: str) -> None:
    with pytest.raises(ValueError, match="not a valid regex"):
        make_style(numeral_pattern=pattern)


def test_invalid_regex_error_keeps_the_original_cause() -> None:
    with pytest.raises(ValueError) as excinfo:
        make_style(numeral_pattern="[")
    assert isinstance(excinfo.value.__cause__, re.error)


@pytest.mark.parametrize("pattern", ["([0-9])+", "([0-9]+)", "(?P<n>[0-9]+)", "(a)(b)"])
def test_numeral_pattern_must_not_have_capturing_groups(pattern: str) -> None:
    # The template wraps each slot in its own named group; extra groups would collide.
    with pytest.raises(ValueError, match="capturing groups"):
        make_style(numeral_pattern=pattern)


@pytest.mark.parametrize(
    "pattern", ["(?i)[a-z]", "(?x)[0-9] +", "(?s).", "(?m)[0-9]+", "(?a)\\d+"]
)
def test_numeral_pattern_must_not_have_flags(pattern: str) -> None:
    # A global flag would change the whole template regex, e.g. make it case-insensitive.
    with pytest.raises(ValueError, match="flags"):
        make_style(numeral_pattern=pattern)


@pytest.mark.parametrize("pattern", ["", "[0-9]*", "x?", "(?:)", "a|"])
def test_numeral_pattern_must_not_match_empty_text(pattern: str) -> None:
    # A slot that can match nothing would "find" a number in every line.
    with pytest.raises(ValueError, match="empty text"):
        make_style(numeral_pattern=pattern)


def test_pattern_errors_name_the_style() -> None:
    with pytest.raises(ValueError, match="'my_style'"):
        make_style(name="my_style", numeral_pattern="([0-9])")


# ============================================================ alphabet_style


def test_alphabet_style_builds_a_counter_style() -> None:
    style = alphabet_style("greek", "αβγ")
    assert isinstance(style, CounterStyle)
    assert style.name == "greek"


def test_alphabet_style_converts_with_the_alphabet() -> None:
    style = alphabet_style("greek", "αβγ")
    assert [style.to_text(n) for n in range(1, 7)] == ["α", "β", "γ", "αα", "ββ", "γγ"]
    assert [style.to_number(c) for c in "αβγ"] == [1, 2, 3]
    assert style.to_number("αα") == 4


def test_alphabet_style_to_number_is_strict() -> None:
    style = alphabet_style("greek", "αβγ")
    for text in ["", "δ", "a", "αβ", " α", "α "]:
        assert style.to_number(text) is None, repr(text)


def test_alphabet_style_pattern_matches_one_letter_of_the_alphabet() -> None:
    style = alphabet_style("greek", "αβγ")
    for letter in "αβγ":
        assert fullmatches(style, letter)
    for text in ["αα", "αβ", "δ", "a", "", " "]:
        assert not fullmatches(style, text), repr(text)


@pytest.mark.parametrize("alphabet", ["-]^\\", "a-z", "]a", "^a", ".*+?", "[]"])
def test_alphabet_style_escapes_regex_characters(alphabet: str) -> None:
    # "a-z" must mean the three letters a, -, z, not the range a to z.
    style = alphabet_style("symbols", alphabet)
    for letter in alphabet:
        assert fullmatches(style, letter), repr(letter)
    for other in ["b", "m", "y", "x", "Q", "0"]:
        if other not in alphabet:
            assert not fullmatches(style, other), repr(other)


def test_alphabet_style_accepts_a_single_letter() -> None:
    style = alphabet_style("one", "x")
    assert style.to_text(3) == "xxx"
    assert fullmatches(style, "x")


def test_alphabet_style_rejects_an_empty_alphabet() -> None:
    with pytest.raises(ValueError, match="empty"):
        alphabet_style("nothing", "")


@pytest.mark.parametrize(
    "alphabet",
    [
        "ab c",  # space
        "ab\tc",  # tab
        "ab\nc",  # line break
        "ab c",  # no-break space
        "ab​c",  # zero-width space
        "ab­c",  # soft hyphen
    ],
)
def test_alphabet_style_rejects_blank_or_invisible_characters(alphabet: str) -> None:
    with pytest.raises(ValueError, match="blank"):
        alphabet_style("blanks", alphabet)


@pytest.mark.parametrize("alphabet", ["aba", "aa", "abcdec"])
def test_alphabet_style_rejects_repeated_letters(alphabet: str) -> None:
    with pytest.raises(ValueError, match="repeated"):
        alphabet_style("repeats", alphabet)


def test_alphabet_style_error_names_the_style() -> None:
    with pytest.raises(ValueError, match="'repeats'"):
        alphabet_style("repeats", "aba")


def test_alphabet_style_checks_the_name() -> None:
    with pytest.raises(ValueError, match="name"):
        alphabet_style("Greek", "αβγ")


def test_alphabet_style_matches_the_builtin_for_the_same_alphabet() -> None:
    custom = alphabet_style("my_vn", LOWER_VN)
    builtin = BUILTIN_COUNTER_STYLES.get("lower_vn")
    for n in range(1, 3 * len(LOWER_VN) + 1):
        assert custom.to_text(n) == builtin.to_text(n)
    for text in [*LOWER_VN, "aa", "f", "ab", ""]:
        assert custom.to_number(text) == builtin.to_number(text)
        assert fullmatches(custom, text) == fullmatches(builtin, text)


# ============================================================ built-ins


BUILTIN_NAMES = (
    "arabic",
    "lower_latin",
    "lower_roman",
    "lower_vn",
    "upper_latin",
    "upper_roman",
    "upper_vn",
)
LETTER_STYLES = {
    "lower_latin": LOWER_LATIN,
    "upper_latin": UPPER_LATIN,
    "lower_vn": LOWER_VN,
    "upper_vn": UPPER_VN,
}


def test_builtin_names() -> None:
    assert BUILTIN_COUNTER_STYLES.names() == BUILTIN_NAMES


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_builtin_is_registered_under_its_own_name(name: str) -> None:
    assert BUILTIN_COUNTER_STYLES.get(name).name == name


@pytest.mark.parametrize("name", BUILTIN_NAMES)
def test_builtin_pattern_works_inside_a_template(name: str) -> None:
    # The template parser wraps the pattern in one named group per slot.
    pattern = BUILTIN_COUNTER_STYLES.get(name).numeral_pattern
    template = re.compile(f"Điều[ \t]+(?P<s0>{pattern})\\.")
    assert template.groups == 1


def test_unknown_builtin_lists_the_known_names() -> None:
    with pytest.raises(KeyError, match="arabic, lower_latin"):
        BUILTIN_COUNTER_STYLES.get("roman")


# ---------- each built-in: pattern and functions agree


def test_arabic() -> None:
    style = BUILTIN_COUNTER_STYLES.get("arabic")
    for n in range(0, 1001):
        text = style.to_text(n)
        assert fullmatches(style, text)
        assert style.to_number(text) == n
    assert style.to_number("01") == 1
    for text in ["", "+1", "-1", "1.5", "IV", "a", " 1"]:
        assert not fullmatches(style, text), repr(text)


@pytest.mark.parametrize("name", ["lower_roman", "upper_roman"])
def test_roman(name: str) -> None:
    style = BUILTIN_COUNTER_STYLES.get(name)
    for n in range(1, MAX_ROMAN + 1):
        text = style.to_text(n)
        assert fullmatches(style, text), text
        assert style.to_number(text) == n


@pytest.mark.parametrize(
    ("name", "four"), [("upper_roman", "IV"), ("lower_roman", "iv")]
)
def test_roman_case(name: str, four: str) -> None:
    style = BUILTIN_COUNTER_STYLES.get(name)
    assert style.to_text(4) == four
    assert style.to_number(four) == 4
    assert not fullmatches(style, four.swapcase())
    assert style.to_number(four.swapcase()) is None


@pytest.mark.parametrize("name", ["lower_roman", "upper_roman"])
def test_roman_pattern_is_loose_and_to_number_is_strict(name: str) -> None:
    style = BUILTIN_COUNTER_STYLES.get(name)
    for bad in ["IIII", "VV", "IM", "VX", "MMMM"]:
        text = bad if name == "upper_roman" else bad.lower()
        assert fullmatches(style, text), text  # the pattern lets it through ...
        assert style.to_number(text) is None, text  # ... and to_number rejects it


@pytest.mark.parametrize(("name", "alphabet"), LETTER_STYLES.items())
def test_letter_style_matches_its_alphabet(name: str, alphabet: str) -> None:
    style = BUILTIN_COUNTER_STYLES.get(name)
    for n, letter in enumerate(alphabet, start=1):
        assert style.to_text(n) == letter
        assert fullmatches(style, letter)
        assert style.to_number(letter) == n


@pytest.mark.parametrize(("name", "alphabet"), LETTER_STYLES.items())
def test_letter_style_pattern_matches_one_letter_only(name: str, alphabet: str) -> None:
    # to_text still writes repeated letters past the end of the alphabet,
    # but the pattern does not read them: "ii" stays a roman numeral.
    style = BUILTIN_COUNTER_STYLES.get(name)
    doubled = style.to_text(len(alphabet) + 1)
    assert len(doubled) == 2
    assert not fullmatches(style, doubled)


@pytest.mark.parametrize(("name", "alphabet"), LETTER_STYLES.items())
def test_letter_style_pattern_rejects_letters_outside_its_alphabet(
    name: str, alphabet: str
) -> None:
    style = BUILTIN_COUNTER_STYLES.get(name)
    for letter in "abcdefghijklmnopqrstuvwxyzđABCDEFGHIJKLMNOPQRSTUVWXYZĐ":
        assert fullmatches(style, letter) == (letter in alphabet), letter


def test_vietnamese_letter_patterns_skip_f_j_w_z() -> None:
    for name in ("lower_vn", "upper_vn"):
        style = BUILTIN_COUNTER_STYLES.get(name)
        for letter in "fjwzFJWZ":
            assert not fullmatches(style, letter), (name, letter)


def test_latin_letter_patterns_reject_đ() -> None:
    assert not fullmatches(BUILTIN_COUNTER_STYLES.get("lower_latin"), "đ")
    assert not fullmatches(BUILTIN_COUNTER_STYLES.get("upper_latin"), "Đ")


# ---------- overlaps between built-ins (resolved later, in pass 2)


@pytest.mark.parametrize("letter", list("ivxlcdm"))
def test_single_roman_letters_are_valid_lower_roman_and_lower_vn(letter: str) -> None:
    roman = BUILTIN_COUNTER_STYLES.get("lower_roman")
    vn = BUILTIN_COUNTER_STYLES.get("lower_vn")
    assert fullmatches(roman, letter) and roman.to_number(letter) is not None
    assert fullmatches(vn, letter) and vn.to_number(letter) is not None


@pytest.mark.parametrize("text", ["ii", "iii", "xx", "xxx", "cc", "mm"])
def test_repeated_roman_letters_are_only_lower_roman(text: str) -> None:
    assert fullmatches(BUILTIN_COUNTER_STYLES.get("lower_roman"), text)
    assert not fullmatches(BUILTIN_COUNTER_STYLES.get("lower_vn"), text)
    assert not fullmatches(BUILTIN_COUNTER_STYLES.get("lower_latin"), text)


# ---------- profiles add their own styles


def test_profile_can_add_an_alphabet_style() -> None:
    styles = BUILTIN_COUNTER_STYLES.extend({"greek": alphabet_style("greek", "αβγ")})
    assert styles.get("greek").to_number("β") == 2
    assert styles.get("arabic").to_number("7") == 7
    assert "greek" not in BUILTIN_COUNTER_STYLES


def test_profile_cannot_replace_a_builtin() -> None:
    with pytest.raises(ValueError, match="already defined: arabic"):
        BUILTIN_COUNTER_STYLES.extend({"arabic": alphabet_style("arabic", "abc")})
