from document_processor.core.numerals import (
    int_to_arabic,
    int_to_upper_latin,
    int_to_lower_latin,
    int_to_upper_roman,
    int_to_lower_roman
)


def format_number(num: int, num_fmt: str) -> str:
    if num_fmt == "decimal":
        return int_to_arabic(num)

    if num_fmt == "decimalZero":
        return f"0{num}"

    if num_fmt == "upperLetter":
        return int_to_upper_latin(num)

    if num_fmt == "lowerLetter":
        return int_to_lower_latin(num)

    if num_fmt == "upperRoman":
        return int_to_upper_roman(num)

    if num_fmt == "lowerRoman":
        return int_to_lower_roman(num)

    raise ValueError(f"Unknown num_fmt: {num_fmt}")
