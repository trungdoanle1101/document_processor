"""Tests for core/numerals.py: plain number-to-text conversions."""

import pytest

from document_processor.core.numerals import int_to_letters, int_to_roman

# ---------- int_to_letters


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "A"),
        (2, "B"),
        (26, "Z"),
        # After Z the letter repeats (Word style), not spreadsheet style:
        (27, "AA"),  # spreadsheet would also give AA ...
        (28, "BB"),  # ... but AB here; this row tells the two schemes apart
        (52, "ZZ"),
        (53, "AAA"),
        (78, "ZZZ"),
    ],
)
def test_int_to_letters(n: int, expected: str) -> None:
    assert int_to_letters(n) == expected


def test_int_to_letters_is_uppercase() -> None:
    # Callers lowercase it themselves for lowerLetter.
    assert int_to_letters(3).isupper()


@pytest.mark.parametrize("n", [0, -1, -27])
def test_int_to_letters_rejects_non_positive(n: int) -> None:
    with pytest.raises(ValueError):
        int_to_letters(n)


# ---------- int_to_roman


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
        (900, "CM"),
        (1994, "MCMXCIV"),
        (3999, "MMMCMXCIX"),
    ],
)
def test_int_to_roman(n: int, expected: str) -> None:
    assert int_to_roman(n) == expected


@pytest.mark.parametrize("n", [0, -1, -4])
def test_int_to_roman_rejects_non_positive(n: int) -> None:
    with pytest.raises(ValueError):
        int_to_roman(n)


def test_int_to_roman_uses_subtractive_forms() -> None:
    # Up to 39, no symbol may repeat four times ("IIII") and V never doubles ("VV").
    for n in range(1, 40):
        roman = int_to_roman(n)
        assert "IIII" not in roman
        assert "VV" not in roman
        assert "XXXX" not in roman
