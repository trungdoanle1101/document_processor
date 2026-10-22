import unicodedata

_LINE_SEP_TO_SPACE_MAPPING = str.maketrans(
    {
        "\n": " ",
        "\t": " ",
        "\r": " ",
    }
)


def is_blank_char(c: str) -> bool:
    if len(c) != 1:
        raise ValueError("chr must be a string of length 1")
    return c.isspace() or unicodedata.category(c) == "Cf"


def is_invisible(text: str) -> bool:
    """
    True if the text has only whitespace or Cf characters
    """
    for c in text:
        if not is_blank_char(c):
            return False

    return True


def to_single_line(text: str) -> str:
    """Replace tabs and line breaks with spaces (a CRLF pair counts as one break); nothing else changes."""
    text = text.replace("\r\n", " ")
    return text.translate(_LINE_SEP_TO_SPACE_MAPPING)


def normalize_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_edges(text: str) -> str:
    # Remove invisible characters from both sides of text
    start = 0
    end = len(text)
    n = len(text)
    while (start < n) and is_blank_char(text[start]):
        start += 1

    while (end > start) and is_blank_char(text[end - 1]):
        end -= 1

    return text[start:end]
