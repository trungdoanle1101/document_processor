"""Checks shared by the prompt templates of answering and conversation."""

import string


def check_fields(name: str, text: str, allowed: frozenset[str]) -> None:
    """The template uses exactly these {fields}: none unknown, none missing."""
    used = {
        field for _, field, _, _ in string.Formatter().parse(text) if field is not None
    }
    unknown = sorted(used - allowed)
    if unknown:
        raise ValueError(
            f"{name} template uses unknown field(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
        )
    missing = sorted(allowed - used)
    if missing:
        raise ValueError(f"{name} template must use {', '.join(missing)}")