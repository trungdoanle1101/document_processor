from string import ascii_uppercase, ascii_lowercase


_ROMAN_NUMERALS = (
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

MAX_ROMAN = 3999

LOWER_LATIN = ascii_lowercase
UPPER_LATIN = ascii_uppercase
LOWER_VN = "abcdđeghiklmnopqrstuvxy"
UPPER_VN = LOWER_VN.upper()

# === Arabic
def int_to_arabic(n: int) -> str:
    if n < 0:
        raise ValueError(f"Cannot convert a negative number to arabic. Got {n}")
    return str(n)


def arabic_to_int(text: str) -> int | None:
    if not (text.isascii() and text.isdigit()):
        return None
    return int(text)

# === Letters
def int_to_letters(n: int, alphabet: str) -> str:
    """Word style: the letter repeats; not spreadsheet-style AA, AB.
    Example: 1:A … 26:Z, 27:AA, 28:BB
    """
    if n <= 0:
        raise ValueError(f"Cannot convert a non-positive number to letters. Got {n}")
    size = len(alphabet)

    i = (n - 1) % size
    letter = alphabet[i]
    rep = (n - 1) // size + 1
    return letter * rep


def letters_to_int(text: str, alphabet: str) -> int | None:
    if not text:
        return None
    first = text[0]
    index = alphabet.find(first)
    if (index < 0) or (text != first * len(text)):
        return None
    return len(alphabet) * (len(text) - 1) + index + 1



# === Upper Latin
def int_to_upper_latin(n: int) -> str:
    return int_to_letters(n, UPPER_LATIN)


def upper_latin_to_int(text: str) -> int | None:
    return letters_to_int(text, UPPER_LATIN)


# === Lower Latin
def int_to_lower_latin(n: int) -> str:
    return int_to_letters(n, LOWER_LATIN)


def lower_latin_to_int(text: str) -> int | None:
    return letters_to_int(text, LOWER_LATIN)


# === Upper VN
def int_to_upper_vn(n: int) -> str:
    return int_to_letters(n, UPPER_VN)


def upper_vn_to_int(text: str) -> int | None:
    return letters_to_int(text, UPPER_VN)


# === Lower VN
def int_to_lower_vn(n: int) -> str:
    return int_to_letters(n, LOWER_VN)


def lower_vn_to_int(text: str) -> int | None:
    return letters_to_int(text, LOWER_VN)


# === Upper Roman

def int_to_upper_roman(n: int) -> str:
    if not (1 <= n <= MAX_ROMAN):
        raise ValueError(
            f"Roman numerals cover 1 to {MAX_ROMAN}. Got {n}"
        )
    s = []
    for symbol, value in _ROMAN_NUMERALS:
        while n >= value:
            s.append(symbol)
            n -= value

    return "".join(s)


_UPPER_ROMAN_TO_INT = {int_to_upper_roman(n): n for n in range(1, MAX_ROMAN + 1)}


def upper_roman_to_int(text: str) -> int | None:
    return _UPPER_ROMAN_TO_INT.get(text)


# === Lower Roman

def int_to_lower_roman(n: int) -> str:
    return int_to_upper_roman(n).lower()


_LOWER_ROMAN_TO_INT = {s.lower(): n for s, n in _UPPER_ROMAN_TO_INT.items()}


def lower_roman_to_int(text: str) -> int | None:
    return _LOWER_ROMAN_TO_INT.get(text)