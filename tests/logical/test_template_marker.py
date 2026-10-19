"""Tests for logical/template_marker.py: _parse_template (private; imported as parse_template).

parse_template checks the SHAPE of one marker template and splits it into parts:

* a ``str`` is literal text; the string ``" "`` stands for one or more blanks
  in the document, and no other literal ever contains a space;
* a ``Slot`` is a ``{style}`` or ``{a|b}`` position for a numeral.

It needs no registry: whether a style exists is checked later, by
compile_template. Error tests match keywords, not whole messages.
"""

import pytest

from document_processor.logical.template_marker import _parse_template as parse_template
from document_processor.logical.template_marker import _Slot as Slot
from document_processor.logical.template_marker import _TemplatePart as TemplatePart


def slot(*styles: str) -> Slot:
    return Slot(styles=styles)


# ============================================================ valid templates


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("Điều {arabic}.", ("Điều", " ", slot("arabic"), ".")),
        ("Điều {arabic}:", ("Điều", " ", slot("arabic"), ":")),
        ("Chương {upper_roman|arabic}", ("Chương", " ", slot("upper_roman", "arabic"))),
        ("Mục {upper_roman}", ("Mục", " ", slot("upper_roman"))),
        ("{arabic}.", (slot("arabic"), ".")),
        ("{arabic}.{arabic}", (slot("arabic"), ".", slot("arabic"))),
        ("{arabic}.{arabic}.", (slot("arabic"), ".", slot("arabic"), ".")),
        ("{arabic}.{arabic}.{arabic}", (slot("arabic"), ".", slot("arabic"), ".", slot("arabic"))),
        ("{lower_vn})", (slot("lower_vn"), ")")),
        ("({lower_roman})", ("(", slot("lower_roman"), ")")),
        ("Phụ lục {upper_latin}", ("Phụ", " ", "lục", " ", slot("upper_latin"))),
        ("Article {arabic}er", ("Article", " ", slot("arabic"), "er")),
        ("§ {arabic}", ("§", " ", slot("arabic"))),
        ("{upper_latin}.{arabic}", (slot("upper_latin"), ".", slot("arabic"))),
    ],
)
def test_parses_templates_with_slots(template: str, expected: tuple[TemplatePart, ...]) -> None:
    assert parse_template(template) == expected


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("MỞ ĐẦU", ("MỞ", " ", "ĐẦU")),
        ("Nơi nhận:", ("Nơi", " ", "nhận:")),
        ("PHỤ LỤC", ("PHỤ", " ", "LỤC")),
        ("x", ("x",)),
    ],
)
def test_parses_word_only_templates(template: str, expected: tuple[TemplatePart, ...]) -> None:
    assert parse_template(template) == expected


def test_returns_a_tuple() -> None:
    assert isinstance(parse_template("Điều {arabic}."), tuple)


def test_slot_keeps_the_order_of_alternatives() -> None:
    assert parse_template("Chương {arabic|upper_roman}") == ("Chương", " ", slot("arabic", "upper_roman"))
    assert parse_template("Chương {upper_roman|arabic}") == ("Chương", " ", slot("upper_roman", "arabic"))


def test_slot_with_many_alternatives() -> None:
    assert parse_template("{a|b|c|d})") == (slot("a", "b", "c", "d"), ")")


@pytest.mark.parametrize("name", ["arabic", "lower_vn", "x", "style2", "my_style_3"])
def test_valid_style_names(name: str) -> None:
    assert parse_template(f"{{{name}}}.") == (slot(name), ".")


# ---------- spaces


def test_neighbouring_literal_characters_merge_into_one_part() -> None:
    parts = parse_template("Điều {arabic}.")
    assert parts[0] == "Điều"


def test_a_run_of_spaces_becomes_one_space_part() -> None:
    assert parse_template("Chương  {upper_roman}") == ("Chương", " ", slot("upper_roman"))
    assert parse_template("MỞ     ĐẦU") == ("MỞ", " ", "ĐẦU")


def test_space_after_a_slot_is_kept() -> None:
    assert parse_template("{arabic} {lower_vn})") == (slot("arabic"), " ", slot("lower_vn"), ")")
    assert parse_template("Chương {upper_roman} PHẦN") == ("Chương", " ", slot("upper_roman"), " ", "PHẦN")


def test_space_between_slots_separates_them() -> None:
    # Not "side by side": the blanks are the separator.
    assert parse_template("{arabic} - {arabic}") == (slot("arabic"), " ", "-", " ", slot("arabic"))


@pytest.mark.parametrize(
    "template",
    ["Điều {arabic}.", "Chương  {upper_roman}", "{arabic} {lower_vn})", "MỞ   ĐẦU", "a b  c   d"],
)
def test_no_literal_part_contains_a_space(template: str) -> None:
    for part in parse_template(template):
        if isinstance(part, str) and part != " ":
            assert " " not in part, part


@pytest.mark.parametrize("template", ["Điều {arabic}.", "MỞ  ĐẦU", "{arabic} {lower_vn})"])
def test_never_two_space_parts_in_a_row(template: str) -> None:
    parts = parse_template(template)
    for left, right in zip(parts, parts[1:]):
        assert not (left == " " and right == " ")


@pytest.mark.parametrize("template", ["Điều {arabic}.", "Chương {upper_roman}", "{arabic}.", "MỞ ĐẦU"])
def test_no_empty_parts(template: str) -> None:
    for part in parse_template(template):
        assert part != ""


def test_ends_with_a_slot_has_no_trailing_empty_part() -> None:
    assert parse_template("Chương {upper_roman}")[-1] == slot("upper_roman")


# ---------- escaped braces: pairs are taken from the left, as in str.format


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("{{arabic}}", ("{arabic}",)),
        ("{{{arabic}}}", ("{", slot("arabic"), "}")),
        ("({{{lower_roman}}})", ("({", slot("lower_roman"), "})")),
        ("{arabic}}}", (slot("arabic"), "}")),
        ("{{{arabic}", ("{", slot("arabic"))),
        ("{{{{arabic}}}}", ("{{arabic}}",)),
        ("Điều {{x}} {arabic}", ("Điều", " ", "{x}", " ", slot("arabic"))),
        ("{{", ("{",)),
        ("}}", ("}",)),
        ("{{}}", ("{}",)),
    ],
)
def test_escaped_braces(template: str, expected: tuple[TemplatePart, ...]) -> None:
    assert parse_template(template) == expected


def test_escaped_brace_between_slots_is_a_separator() -> None:
    assert parse_template("{a}{{{b}") == (slot("a"), "{", slot("b"))


# ============================================================ errors


def assert_error(template: str, *keywords: str) -> str:
    with pytest.raises(ValueError) as excinfo:
        parse_template(template)
    message = str(excinfo.value)
    for keyword in keywords:
        assert keyword in message, f"{keyword!r} not in {message!r}"
    return message


# ---------- whole template


def test_empty_template() -> None:
    assert_error("", "empty")


@pytest.mark.parametrize("template", [" ", "   "])
def test_spaces_only_template(template: str) -> None:
    with pytest.raises(ValueError):
        parse_template(template)


@pytest.mark.parametrize("template", [" Điều {arabic}.", "Điều {arabic}. ", "  MỞ ĐẦU", "MỞ ĐẦU  "])
def test_leading_or_trailing_space(template: str) -> None:
    assert_error(template, "leading or trailing")


@pytest.mark.parametrize(
    ("template", "column"),
    [
        ("Điều\t{arabic}.", 5),  # tab
        ("Điều {arabic}.", 5),  # no-break space
        ("Điều {arabic}\n.", 14),  # line break
        ("MỞ​ĐẦU", 3),  # zero-width space
        ("MỞ ĐẦU", 3),  # em space
    ],
)
def test_whitespace_other_than_a_plain_space(template: str, column: int) -> None:
    message = assert_error(template, "plain space")
    assert f"column {column}" in message


@pytest.mark.parametrize("template", ["\tĐiều {arabic}.", "Điều {arabic}.\n"])
def test_other_whitespace_at_the_edges_is_rejected(template: str) -> None:
    with pytest.raises(ValueError):
        parse_template(template)


# ---------- braces


@pytest.mark.parametrize(
    ("template", "column"),
    [
        ("Điều {arabic", 6),
        ("{", 1),
        ("{arabic", 1),
        ("{{{", 3),  # "{{" is a literal, then a "{" that never closes
        ("Điều {{{arabic", 8),
    ],
)
def test_brace_never_closed(template: str, column: int) -> None:
    message = assert_error(template, "never closed")
    assert f"column {column}" in message


@pytest.mark.parametrize(
    ("template", "column"),
    [
        ("Điều {arabic}}", 14),
        ("}", 1),
        ("arabic}", 7),
        ("}}}", 3),  # "}}" is a literal, then a lone "}"
        ("{{arabic}", 9),  # "{{" is a literal, so the "}" is alone
        ("{{{arabic}}", 11),  # "{{", slot, then a lone "}"
    ],
)
def test_lone_closing_brace(template: str, column: int) -> None:
    message = assert_error(template, "lone")
    assert f"column {column}" in message


# ---------- slot contents


@pytest.mark.parametrize(
    "template",
    [
        "Điều {}.",  # empty slot
        "Điều {arabic|}.",  # empty alternative
        "Điều {|arabic}.",
        "Điều {arabic||lower_vn}.",
        "Điều { arabic }.",  # spaces inside
        "Điều {Arabic}.",  # uppercase
        "Điều {1st}.",  # starts with a digit
        "Điều {_x}.",  # starts with an underscore
        "Điều {lower-roman}.",  # hyphen
        "Điều {chữ}.",  # non-ASCII
        "{a{b}",  # a brace inside a slot
    ],
)
def test_invalid_style_name(template: str) -> None:
    assert_error(template, "invalid style")


def test_invalid_style_error_names_the_template_and_the_slot_column() -> None:
    message = assert_error("Điều {arabik!}.", "invalid style")
    assert "Điều {arabik!}." in message
    assert "column 6" in message


@pytest.mark.parametrize("template", ["Điều {arabic|arabic}.", "{a|b|a})", "{x|x|x}."])
def test_style_repeated_in_one_slot(template: str) -> None:
    assert_error(template, "multiple times")


def test_slot_errors_are_reported_before_later_brace_errors() -> None:
    # The scan stops at the first problem.
    assert_error("{Bad} }", "invalid style")


# ---------- the finished parts


@pytest.mark.parametrize("template", ["{arabic}{arabic}", "{arabic}{lower_vn})", "Điều {a}{b}.", "{a}{b}{c}."])
def test_two_slots_side_by_side(template: str) -> None:
    assert_error(template, "side by side")


@pytest.mark.parametrize("template", ["{arabic}", "{upper_roman|arabic}", "{arabic} {lower_vn}", "{a} {b} {c}"])
def test_no_literal_text(template: str) -> None:
    # Would match every line that starts with a number.
    assert_error(template, "no literal text")


def test_escaped_brace_counts_as_literal_text() -> None:
    assert parse_template("{{{arabic}}}") == ("{", slot("arabic"), "}")


# ---------- every error names the template


@pytest.mark.parametrize(
    "template",
    [
        " Điều {arabic}.",
        "Điều\t{arabic}.",
        "Điều {arabic",
        "Điều {arabic}}",
        "Điều {Arabic}.",
        "Điều {arabic|arabic}.",
        "Điều {a}{b}.",
        "{arabic}",
    ],
)
def test_error_messages_include_the_template(template: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_template(template)
    assert repr(template) in str(excinfo.value)