"""Tests for the shared text helpers."""

import unicodedata

import pytest

from document_processor.core.text_utils import (
    is_invisible,
    normalize_nfc,
    strip_edges,
    to_single_line,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Phạm vi điều chỉnh", "Phạm vi điều chỉnh"),
        ("", ""),
        ("a\tb", "a b"),
        ("a\nb", "a b"),
        ("a\rb", "a b"),
        ("a\r\nb", "a b"),  # one CRLF line break -> one space
        ("a\n\nb", "a  b"),  # two real line breaks -> two spaces
        ("a\r\n\r\nb", "a  b"),
        ("\tlead and trail\n", " lead and trail "),
    ],
    ids=[
        "vietnamese-unchanged",
        "empty",
        "tab",
        "newline",
        "carriage-return",
        "crlf-is-one-break",
        "two-newlines",
        "two-crlf",
        "ends-not-stripped",
    ],
)
def test_to_single_line(text: str, expected: str) -> None:
    assert to_single_line(text) == expected


@pytest.mark.parametrize("text", ["a\tb\nc\r\nd", "Điều 1.\tPhạm vi\nđiều chỉnh"])
def test_to_single_line_leaves_no_separators(text: str) -> None:
    result = to_single_line(text)
    assert "\t" not in result
    assert "\n" not in result
    assert "\r" not in result


def test_to_single_line_keeps_other_whitespace() -> None:
    # Only separators are replaced: spaces and NBSP are content and stay as they are.
    assert to_single_line("a  b c") == "a  b c"


# ---------- is_invisible / strip_edges / normalize_nfc


@pytest.mark.parametrize(
    "text",
    ["", " ", "\t\n\r", " ", "​", "﻿", "­", " ​\t‎"],
    ids=["empty", "space", "separators", "nbsp", "zwsp", "bom", "soft-hyphen", "mixed"],
)
def test_is_invisible_true(text: str) -> None:
    assert is_invisible(text)


@pytest.mark.parametrize("text", ["a", " a ", "​Đ", ".", "-"])
def test_is_invisible_false(text: str) -> None:
    assert not is_invisible(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("abc", "abc"),
        ("  abc  ", "abc"),
        ("\t​\tĐiều 1.", "Điều 1."),
        ("a ​ b", "a ​ b"),  # inner characters kept
        (" ​\t ", ""),  # all blank
        (" x ", "x"),
    ],
)
def test_strip_edges(text: str, expected: str) -> None:
    assert strip_edges(text) == expected


def test_strip_edges_agrees_with_is_invisible() -> None:
    for text in ["", " ", "​", "\t­﻿", " x "]:
        assert (strip_edges(text) == "") == is_invisible(text)


def test_normalize_nfc() -> None:
    nfd = unicodedata.normalize("NFD", "Điều khoản")
    assert nfd != "Điều khoản"
    assert normalize_nfc(nfd) == "Điều khoản"
    assert normalize_nfc("Điều khoản") == "Điều khoản"
