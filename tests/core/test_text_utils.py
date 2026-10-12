import pytest

from document_processor.core.text_utils import to_single_line


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