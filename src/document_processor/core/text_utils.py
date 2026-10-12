import unicodedata

_LINE_SEP_TO_SPACE_MAPPING = str.maketrans(
    {
        "\n": " ",
        "\t": " ",
        "\r": " ",
    }
)


def is_invisible(text: str) -> bool:
    """
    True if the text has only whitespace or Cf characters
    """
    for c in text:
        if not c.isspace() and unicodedata.category(c) != "Cf":
            return False

    return True


def to_single_line(text: str) -> str:
    """Replace tabs and line breaks with spaces (a CRLF pair counts as one break); nothing else changes."""
    text = text.replace("\r\n", " ")
    return text.translate(_LINE_SEP_TO_SPACE_MAPPING)
