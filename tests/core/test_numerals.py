"""Tests for core/numerals.py: number <-> text conversions for each counter style.

Two directions, two contracts:

* ``int_to_*`` (to_text) gets numbers from our own code. A number the style
  cannot write is a bug in the caller, so it raises ``ValueError``.
* ``*_to_int`` (to_number) gets text from documents. Text that is not exactly a
  number in the style is normal, so it returns ``None``. It never strips,
  never changes case, and never accepts look-alike characters.
"""

from collections.abc import Callable

import pytest

from document_processor.core.numerals import (
    LOWER_LATIN,
    LOWER_VN,
    MAX_ROMAN,
    UPPER_LATIN,
    UPPER_VN,
    arabic_to_int,
    int_to_arabic,
    int_to_letters,
    int_to_lower_latin,
    int_to_lower_roman,
    int_to_lower_vn,
    int_to_upper_latin,
    int_to_upper_roman,
    int_to_upper_vn,
    letters_to_int,
    lower_latin_to_int,
    lower_roman_to_int,
    lower_vn_to_int,
    upper_latin_to_int,
    upper_roman_to_int,
    upper_vn_to_int,
)

type ToText = Callable[[int], str]
type ToNumber = Callable[[str], int | None]

# Text that is never a number in any style: empty, padded, or not a numeral.
NEVER_A_NUMBER = ["", " ", "\t", "1 2", ".", ")", "-"]


# ---------- alphabets


def test_latin_alphabets() -> None:
    assert LOWER_LATIN == "abcdefghijklmnopqrstuvwxyz"
    assert UPPER_LATIN == LOWER_LATIN.upper()


def test_vietnamese_alphabet_is_the_legal_list_sequence() -> None:
    # The sequence used for "điểm" in Vietnamese legal documents:
    # no f, j, w, z, and đ comes right after d.
    assert LOWER_VN == "abcdđeghiklmnopqrstuvxy"
    assert len(LOWER_VN) == 23
    for letter in "fjwz":
        assert letter not in LOWER_VN


def test_upper_vietnamese_alphabet() -> None:
    assert UPPER_VN == "ABCDĐEGHIKLMNOPQRSTUVXY"


@pytest.mark.parametrize("alphabet", [LOWER_LATIN, UPPER_LATIN, LOWER_VN, UPPER_VN])
def test_alphabets_have_no_repeated_letters(alphabet: str) -> None:
    assert len(set(alphabet)) == len(alphabet)


# ---------- arabic: to_text


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, "0"), (1, "1"), (9, "9"), (10, "10"), (42, "42"), (2026, "2026")],
)
def test_int_to_arabic(n: int, expected: str) -> None:
    assert int_to_arabic(n) == expected


@pytest.mark.parametrize("n", [-1, -42])
def test_int_to_arabic_rejects_negative(n: int) -> None:
    with pytest.raises(ValueError, match="negative"):
        int_to_arabic(n)


# ---------- arabic: to_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0),
        ("1", 1),
        ("9", 9),
        ("42", 42),
        ("2026", 2026),
        # Leading zeros are accepted: "Điều 01" and "01." occur in real documents.
        ("01", 1),
        ("007", 7),
    ],
)
def test_arabic_to_int(text: str, expected: int) -> None:
    assert arabic_to_int(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        *NEVER_A_NUMBER,
        # Things Python's int() would accept but a marker number must not have:
        " 12",
        "12 ",
        "\t12",
        "12\n",
        "+12",
        "-12",
        "1_000",
        # Separators and decimals are not one number:
        "1.5",
        "5.000",
        "1,5",
        # Digits from other scripts and digit-like symbols:
        "١٢",  # Arabic-Indic digits ١٢
        "۱۲",  # Extended Arabic-Indic digits ۱۲
        "１２",  # fullwidth digits １２
        "²",  # superscript two ² (isdigit() is True)
        "①",  # circled one ① (isdigit() is True)
        # Mixed with letters:
        "1a",
        "a1",
        "IV",
    ],
)
def test_arabic_to_int_rejects(text: str) -> None:
    assert arabic_to_int(text) is None


# ---------- int_to_letters (generic)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "A"),
        (2, "B"),
        (26, "Z"),
        # After the last letter, the letter repeats (Word style):
        (27, "AA"),
        (28, "BB"),  # spreadsheet style would give AB; this row tells them apart
        (52, "ZZ"),
        (53, "AAA"),
        (78, "ZZZ"),
    ],
)
def test_int_to_letters_word_style(n: int, expected: str) -> None:
    assert int_to_letters(n, UPPER_LATIN) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "a"),
        (4, "d"),
        (5, "đ"),  # đ is its own letter, right after d
        (6, "e"),
        (7, "g"),  # no f
        (9, "i"),
        (10, "k"),  # no j
        (21, "v"),
        (22, "x"),  # no w
        (23, "y"),  # last letter; no z
        (24, "aa"),
        (28, "đđ"),
        (46, "yy"),
        (47, "aaa"),
    ],
)
def test_int_to_letters_vietnamese(n: int, expected: str) -> None:
    assert int_to_letters(n, LOWER_VN) == expected


def test_int_to_letters_uses_the_given_alphabet() -> None:
    assert int_to_letters(1, "xyz") == "x"
    assert int_to_letters(3, "xyz") == "z"
    assert int_to_letters(4, "xyz") == "xx"


@pytest.mark.parametrize("n", [0, -1, -27])
def test_int_to_letters_rejects_non_positive(n: int) -> None:
    with pytest.raises(ValueError, match="non-positive"):
        int_to_letters(n, UPPER_LATIN)


# ---------- letters_to_int (generic)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", 1),
        ("B", 2),
        ("Z", 26),
        ("AA", 27),
        ("BB", 28),
        ("ZZ", 52),
        ("AAA", 53),
    ],
)
def test_letters_to_int_latin(text: str, expected: int) -> None:
    assert letters_to_int(text, UPPER_LATIN) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("a", 1), ("d", 4), ("đ", 5), ("e", 6), ("g", 7), ("y", 23), ("aa", 24), ("đđ", 28)],
)
def test_letters_to_int_vietnamese(text: str, expected: int) -> None:
    assert letters_to_int(text, LOWER_VN) == expected


@pytest.mark.parametrize(
    "text",
    [
        *NEVER_A_NUMBER,
        # Mixed letters are not a Word-style number (spreadsheet "AB" included):
        "AB",
        "AAB",
        "ABA",
        "BA",
        # Padded:
        " A",
        "A ",
        "A.",
        "A)",
        # Wrong case for this alphabet:
        "a",
        "aa",
        # Not ASCII Latin, even if Python calls them uppercase letters:
        "Đ",
        "É",
        "Ä",
        "Α",  # Greek capital alpha, looks like A
        "А",  # Cyrillic capital A, looks like A
        "Ａ",  # fullwidth A
        "1",
    ],
)
def test_letters_to_int_rejects_latin(text: str) -> None:
    assert letters_to_int(text, UPPER_LATIN) is None


@pytest.mark.parametrize("text", ["f", "j", "w", "z", "ff", "A", "Đ", "ă", "â", "ê", "ô", "ơ", "ư"])
def test_letters_to_int_rejects_letters_outside_the_vietnamese_sequence(text: str) -> None:
    assert letters_to_int(text, LOWER_VN) is None


# ---------- named letter styles


LETTER_STYLES: list[tuple[str, ToText, ToNumber, str]] = [
    ("upper_latin", int_to_upper_latin, upper_latin_to_int, UPPER_LATIN),
    ("lower_latin", int_to_lower_latin, lower_latin_to_int, LOWER_LATIN),
    ("upper_vn", int_to_upper_vn, upper_vn_to_int, UPPER_VN),
    ("lower_vn", int_to_lower_vn, lower_vn_to_int, LOWER_VN),
]
LETTER_STYLE_IDS = [name for name, *_ in LETTER_STYLES]


@pytest.mark.parametrize(("name", "to_text", "to_number", "alphabet"), LETTER_STYLES, ids=LETTER_STYLE_IDS)
def test_letter_style_matches_its_alphabet(
    name: str, to_text: ToText, to_number: ToNumber, alphabet: str
) -> None:
    for i, letter in enumerate(alphabet, start=1):
        assert to_text(i) == letter
        assert to_number(letter) == i


@pytest.mark.parametrize(("name", "to_text", "to_number", "alphabet"), LETTER_STYLES, ids=LETTER_STYLE_IDS)
def test_letter_style_round_trips(name: str, to_text: ToText, to_number: ToNumber, alphabet: str) -> None:
    # Three full passes through the alphabet, so wrap-around is covered.
    for n in range(1, 3 * len(alphabet) + 1):
        assert to_number(to_text(n)) == n


@pytest.mark.parametrize(("name", "to_text", "to_number", "alphabet"), LETTER_STYLES, ids=LETTER_STYLE_IDS)
def test_letter_style_rejects_non_positive(
    name: str, to_text: ToText, to_number: ToNumber, alphabet: str
) -> None:
    with pytest.raises(ValueError):
        to_text(0)


@pytest.mark.parametrize(("name", "to_text", "to_number", "alphabet"), LETTER_STYLES, ids=LETTER_STYLE_IDS)
def test_letter_style_rejects_text_that_is_never_a_number(
    name: str, to_text: ToText, to_number: ToNumber, alphabet: str
) -> None:
    for text in NEVER_A_NUMBER:
        assert to_number(text) is None, repr(text)


def test_letter_styles_are_case_strict() -> None:
    assert upper_latin_to_int("a") is None
    assert lower_latin_to_int("A") is None
    assert upper_vn_to_int("đ") is None
    assert lower_vn_to_int("Đ") is None


def test_latin_styles_reject_đ() -> None:
    assert lower_latin_to_int("đ") is None
    assert upper_latin_to_int("Đ") is None


@pytest.mark.parametrize(
    "text",
    [
        "ß",  # .upper() is "SS"
        "ﬀ",  # .upper() is "FF"
        "ı",  # dotless i: .upper() is "I"
        "ſ",  # long s: .upper() is "S"
    ],
)
def test_lower_latin_rejects_letters_that_change_under_upper(text: str) -> None:
    # A parser that upper-cased its input first would turn these into Latin letters.
    assert lower_latin_to_int(text) is None
    assert lower_vn_to_int(text) is None


def test_upper_latin_rejects_kelvin_sign() -> None:
    # U+212A KELVIN SIGN looks like K and lower-cases to "k".
    assert upper_latin_to_int("K") is None


# ---------- roman: to_text


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "I"),
        (2, "II"),
        (3, "III"),
        (4, "IV"),  # subtractive forms are the classic mistakes
        (5, "V"),
        (9, "IX"),
        (10, "X"),
        (14, "XIV"),
        (19, "XIX"),
        (40, "XL"),
        (49, "XLIX"),
        (90, "XC"),
        (99, "XCIX"),
        (400, "CD"),
        (444, "CDXLIV"),
        (900, "CM"),
        (1994, "MCMXCIV"),
        (2026, "MMXXVI"),
        (3888, "MMMDCCCLXXXVIII"),  # the longest numeral
        (3999, "MMMCMXCIX"),
    ],
)
def test_int_to_upper_roman(n: int, expected: str) -> None:
    assert int_to_upper_roman(n) == expected


@pytest.mark.parametrize(("n", "expected"), [(1, "i"), (4, "iv"), (9, "ix"), (14, "xiv"), (3999, "mmmcmxcix")])
def test_int_to_lower_roman(n: int, expected: str) -> None:
    assert int_to_lower_roman(n) == expected


def test_max_roman() -> None:
    assert MAX_ROMAN == 3999


@pytest.mark.parametrize("to_text", [int_to_upper_roman, int_to_lower_roman])
@pytest.mark.parametrize("n", [0, -1, -4, MAX_ROMAN + 1, 10_000])
def test_roman_to_text_rejects_out_of_range(to_text: ToText, n: int) -> None:
    with pytest.raises(ValueError, match="1 to 3999"):
        to_text(n)


def test_int_to_upper_roman_uses_canonical_forms() -> None:
    # No symbol repeats four times, and V, L, D never repeat.
    for n in range(1, MAX_ROMAN + 1):
        roman = int_to_upper_roman(n)
        for bad in ("IIII", "XXXX", "CCCC", "MMMM", "VV", "LL", "DD"):
            assert bad not in roman, (n, roman)


# ---------- roman: to_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I", 1),
        ("IV", 4),
        ("IX", 9),
        ("XIV", 14),
        ("XL", 40),
        ("XC", 90),
        ("CD", 400),
        ("CM", 900),
        ("MCMXCIV", 1994),
        ("MMMCMXCIX", 3999),
    ],
)
def test_upper_roman_to_int(text: str, expected: int) -> None:
    assert upper_roman_to_int(text) == expected
    assert lower_roman_to_int(text.lower()) == expected


@pytest.mark.parametrize(
    ("to_text", "to_number"),
    [(int_to_upper_roman, upper_roman_to_int), (int_to_lower_roman, lower_roman_to_int)],
    ids=["upper_roman", "lower_roman"],
)
def test_roman_round_trips_over_the_whole_range(to_text: ToText, to_number: ToNumber) -> None:
    for n in range(1, MAX_ROMAN + 1):
        assert to_number(to_text(n)) == n


@pytest.mark.parametrize(
    "text",
    [
        *NEVER_A_NUMBER,
        # Non-canonical forms: a symbol four times, V/L/D repeated.
        "IIII",
        "XXXX",
        "CCCC",
        "MMMM",  # also above MAX_ROMAN
        "VV",
        "LL",
        "DD",
        # Subtractive pairs that are not allowed:
        "IL",
        "IC",
        "IM",
        "VX",
        "XD",
        "XM",
        "LC",
        "DM",
        # Two subtractions in a row, or a subtraction then a repeat:
        "IIV",
        "IXI X",
        "IXX",
        "XCX",
        # Wrong order:
        "IIX",
        "VIV",
        # Letters that are not roman symbols:
        "ABC",
        "IVA",
        "A",
        # Padded or punctuated:
        " IV",
        "IV ",
        "IV.",
        "I V",
        # Wrong case, or mixed case:
        "iv",
        "Iv",
        # Unicode roman numeral characters look the same but are single code points:
        "Ⅳ",  # Ⅳ ROMAN NUMERAL FOUR
        "Ⅰ",  # Ⅰ ROMAN NUMERAL ONE
        # Look-alikes:
        "І",  # І Cyrillic capital Byelorussian-Ukrainian I
        "1",
    ],
)
def test_upper_roman_to_int_rejects(text: str) -> None:
    assert upper_roman_to_int(text) is None


@pytest.mark.parametrize(
    "text",
    [
        *NEVER_A_NUMBER,
        "iiii",
        "vv",
        "im",
        "vx",
        "IV",
        "iV",
        "ı",  # dotless i: .upper() is "I"
        "ıv",
        "ⅳ",  # ⅳ SMALL ROMAN NUMERAL FOUR
        "і",  # і Cyrillic small Byelorussian-Ukrainian i
    ],
)
def test_lower_roman_to_int_rejects(text: str) -> None:
    assert lower_roman_to_int(text) is None


def test_every_canonical_roman_numeral_parses_to_a_different_number() -> None:
    numbers = {upper_roman_to_int(int_to_upper_roman(n)) for n in range(1, MAX_ROMAN + 1)}
    assert numbers == set(range(1, MAX_ROMAN + 1))


# ---------- overlaps between styles


@pytest.mark.parametrize(
    ("text", "roman", "letter"),
    [
        ("i", 1, 9),
        ("v", 5, 21),
        ("x", 10, 22),
        ("c", 100, 3),
        ("d", 500, 4),
        ("l", 50, 11),
        ("m", 1000, 12),
    ],
)
def test_single_letters_that_are_valid_in_two_styles(text: str, roman: int, letter: int) -> None:
    # Both readings are valid at this layer; choosing one is pass 2's job.
    assert lower_roman_to_int(text) == roman
    assert lower_vn_to_int(text) == letter


@pytest.mark.parametrize(("text", "roman", "letter"), [("ii", 2, 32), ("xx", 20, 45), ("ccc", 300, 49)])
def test_repeated_letters_that_are_valid_in_two_styles(text: str, roman: int, letter: int) -> None:
    # Only a single-letter candidate pattern keeps "ii" from being read as letter 32.
    assert lower_roman_to_int(text) == roman
    assert lower_vn_to_int(text) == letter