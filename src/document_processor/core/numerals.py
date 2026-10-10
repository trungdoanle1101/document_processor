from string import ascii_uppercase

ROMAN_NUMERALS = (
    ("M", 1000),
    ("CM", 900),
    ("D", 500),
    ("CD", 400),
    ("C", 100),
    ("XC", 90),
    ("L", 50),
    ("XL", 40),
    ("X", 10),
    ("IX", 9),
    ("V", 5),
    ("IV", 4),
    ("I", 1),
)


def int_to_letters(n: int) -> str:
    """Word style: the letter repeats; not spreadsheet-style AA, AB.
    Example: 1:A … 26:Z, 27:AA, 28:BB
    """
    num_alphabet = len(ascii_uppercase)
    if n <= 0:
        raise ValueError(f"Cannot convert a non-positive number to letter. Got {n}")

    i = (n - 1) % num_alphabet
    letter = ascii_uppercase[i]
    rep = (n - 1) // num_alphabet + 1

    return letter * rep


def int_to_roman(n: int) -> str:
    if n <= 0:
        raise ValueError(
            f"Cannot convert a non-positive number to roman numerals. Got {n}"
        )
    s = []
    for symbol, value in ROMAN_NUMERALS:
        while n >= value:
            s.append(symbol)
            n -= value

    return "".join(s)
