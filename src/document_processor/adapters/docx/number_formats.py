from document_processor.core.numerals import int_to_letters, int_to_roman


def format_number(num: int, num_fmt: str) -> str:
    if num_fmt == "decimal":
        return str(num)

    if num_fmt == "decimalZero":
        return f"0{num}"

    if num_fmt == "upperLetter":
        return int_to_letters(num)

    if num_fmt == "lowerLetter":
        return int_to_letters(num).lower()

    if num_fmt == "upperRoman":
        return int_to_roman(num)

    if num_fmt == "lowerRoman":
        return int_to_roman(num).lower()

    raise ValueError(f"Unknown num_fmt: {num_fmt}")
