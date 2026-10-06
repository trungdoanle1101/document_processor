import unicodedata

def is_invisible(text: str) -> bool:
    """
    True if the text has no invisible characters: only whitespace and Cf characters
    """
    for c in text:
        if not c.isspace() and unicodedata.category(c) != "Cf":
            return False

    return True