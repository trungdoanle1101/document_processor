import unicodedata

def is_invisible(text: str) -> bool:
    """
    True if the text has only whitespace or Cf characters
    """
    for c in text:
        if not c.isspace() and unicodedata.category(c) != "Cf":
            return False

    return True