import re
from dataclasses import dataclass
from functools import partial
from collections.abc import Callable

from document_processor.core.numerals import (
    letters_to_int, int_to_letters,
    lower_roman_to_int, upper_roman_to_int,
    int_to_lower_roman, int_to_upper_roman,
    arabic_to_int, int_to_arabic,
    lower_latin_to_int, int_to_lower_latin,
    upper_latin_to_int, int_to_upper_latin,
    lower_vn_to_int,  int_to_lower_vn,
    upper_vn_to_int, int_to_upper_vn,
    LOWER_LATIN, UPPER_LATIN,
    LOWER_VN, UPPER_VN,
    
)
from document_processor.core.text_utils import is_blank_char
from .registry import Registry
from .markers import NAME_PATTERN


@dataclass(frozen=True, kw_only=True)
class CounterStyle:
    name: str

    numeral_pattern: str
    """Regex for how a numeral in this style is written ("[0-9]+", "[IVXLCDM]+").
    Placed in each {style} slot of a template to find which characters are the
    numeral. Deliberately loose: to_number makes the final decision."""
    
    # strict: None if not valid in this style
    to_number: Callable[[str], int | None]
    to_text: Callable[[int], str]

    def __post_init__(self) -> None:
        # Check 'name' is a valid slot name
        if not NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid value for attribute 'name': got {self.name!r}. Must fully match '[a-z][a-z0-9_]*'")


        owner = f"{type(self).__name__} {self.name!r} 'numeral_pattern'"

        # Check valid pattern
        try:
            compiled_pattern = re.compile(self.numeral_pattern)
        except re.error as e:
            raise ValueError(f"{owner}: {self.numeral_pattern!r} is not a valid regex pattern: {e}") from e

        # Check no capturing groups
        if compiled_pattern.groups != 0:
            raise ValueError(f"{owner}: must not contain capturing groups, got {compiled_pattern.groups}")

        # Check no flags
        if compiled_pattern.flags != re.compile("").flags:
            raise ValueError(f"{owner}: must not contain flags, got {compiled_pattern.flags}")

        # Pattern must not match empty text
        if compiled_pattern.fullmatch(""):
            raise ValueError(f"{owner}: must not match empty text, got {self.numeral_pattern!r}")


def alphabet_style(name: str, alphabet: str) -> CounterStyle:
    if not alphabet:
        raise ValueError(f"style {name!r}: alphabet must not be empty")

    seen_char: set[str] = set()
    for c in alphabet:
        if is_blank_char(c):
            raise ValueError(f"style {name!r}: alphabet must not contain blank/invisible character")
        if c in seen_char:
            raise ValueError(f"style {name!r}: alphabet must not contain repeated characters - duplicate {c!r}")
        seen_char.add(c)

    pattern = f"[{''.join(re.escape(c) for c in alphabet)}]"
    return CounterStyle(
        name=name,
        numeral_pattern=pattern,
        to_number=partial(letters_to_int, alphabet=alphabet),
        to_text=partial(int_to_letters, alphabet=alphabet)
    )

_BUILTINS = (
     CounterStyle(
        name="arabic",
        numeral_pattern="[0-9]+",
        to_number=arabic_to_int,
        to_text=int_to_arabic
    ),
    CounterStyle(
        name="lower_roman",
        numeral_pattern="[ivxlcdm]+",
        to_number=lower_roman_to_int,
        to_text=int_to_lower_roman
    ),
    CounterStyle(
        name="upper_roman",
        numeral_pattern="[IVXLCDM]+",
        to_number=upper_roman_to_int,
        to_text=int_to_upper_roman
    ),
    CounterStyle(
        name="lower_latin",
        numeral_pattern=f"[{LOWER_LATIN}]",
        to_number=lower_latin_to_int,
        to_text=int_to_lower_latin
    ),
    CounterStyle(
        name="upper_latin",
        numeral_pattern=f"[{UPPER_LATIN}]",
        to_number=upper_latin_to_int,
        to_text=int_to_upper_latin
    ),
    CounterStyle(
        name="lower_vn",
        numeral_pattern=f"[{LOWER_VN}]",
        to_number=lower_vn_to_int,
        to_text=int_to_lower_vn
    ),
    CounterStyle(
        name="upper_vn",
        numeral_pattern=f"[{UPPER_VN}]",
        to_number=upper_vn_to_int,
        to_text=int_to_upper_vn
    )
)

BUILTIN_COUNTER_STYLES = Registry(
    kind="counter style",
    items={
        style.name: style for style in _BUILTINS
    }
)
