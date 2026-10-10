"""Tests for adapters/docx/numbering.py.

All numbering is built from small XML strings. SAMPLE_NUMBERING reproduces the
lists in sample_doc.docx (bullets, "1. / a)" numbering, and the two table lists),
so the expected markers are the ones Word shows for that file.
"""

import pytest
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml.etree import _Element

from document_processor.adapters.docx.numbering import (
    LevelDefinition,
    ListCounter,
    Numberer,
    build_level_definition,
    find_level,
    find_level_override,
    find_replacement_level,
    read_abstract_num_id,
    read_ilvl,
    read_lvl_text,
    read_num_fmt,
    read_num_id,
    read_start,
    read_start_override,
    resolve_level_definition,
    resolve_marker,
)

W = nsdecls("w")


# ---------- builders


def numbering(inner: str) -> _Element:
    """A <w:numbering> root, as in word/numbering.xml."""
    return parse_xml(f"<w:numbering {W}>{inner}</w:numbering>")


def lvl(ilvl: int, num_fmt: str | None, lvl_text: str | None, start: int | None = 1) -> str:
    """One <w:lvl>; pass None to leave a child element out."""
    parts = [f'<w:lvl w:ilvl="{ilvl}">']
    if start is not None:
        parts.append(f'<w:start w:val="{start}"/>')
    if num_fmt is not None:
        parts.append(f'<w:numFmt w:val="{num_fmt}"/>')
    if lvl_text is not None:
        parts.append(f'<w:lvlText w:val="{lvl_text}"/>')
    parts.append("</w:lvl>")
    return "".join(parts)


def abstract_num(abstract_num_id: int, *levels: str) -> str:
    return f'<w:abstractNum w:abstractNumId="{abstract_num_id}">{"".join(levels)}</w:abstractNum>'


def num(num_id: int, abstract_num_id: int, *overrides: str) -> str:
    return (
        f'<w:num w:numId="{num_id}"><w:abstractNumId w:val="{abstract_num_id}"/>'
        f"{''.join(overrides)}</w:num>"
    )


def paragraph(inner: str = "") -> _Element:
    return parse_xml(f"<w:p {W}>{inner}</w:p>")


def level_element(xml: str) -> _Element:
    """A single <w:lvl> element, parsed on its own."""
    return parse_xml(xml.replace("<w:lvl ", f"<w:lvl {W} ", 1))


# The lists in sample_doc.docx (LibreOffice writes one abstractNum per num).
SAMPLE_NUMBERING = numbering(
    abstract_num(2, lvl(0, "bullet", ""), lvl(1, "bullet", "◦"))
    + abstract_num(3, lvl(0, "decimal", "%1."), lvl(1, "lowerLetter", "%2)"))
    + abstract_num(4, lvl(0, "bullet", ""))
    + abstract_num(5, lvl(0, "decimal", "%1."))
    + num(2, 2)
    + num(3, 3)
    + num(4, 4)
    + num(5, 5)
)

# One abstractNum shared by two lists; list 7 overrides two of its levels.
OVERRIDE_NUMBERING = numbering(
    abstract_num(
        3,
        lvl(0, "decimal", "%1."),
        lvl(1, "lowerLetter", "%2)"),
    )
    + num(8, 3)
    + num(
        7,
        3,
        '<w:lvlOverride w:ilvl="0"><w:startOverride w:val="5"/></w:lvlOverride>',
        '<w:lvlOverride w:ilvl="1">' + lvl(1, "upperRoman", "%2.", start=4) + "</w:lvlOverride>",
        '<w:lvlOverride w:ilvl="2">' + lvl(2, "lowerRoman", "(%3)") + "</w:lvlOverride>",
    )
    + num(9, 99)  # points to an abstractNum that does not exist
)

# Templates that combine several levels, each with its own format.
MULTILEVEL_NUMBERING = numbering(
    abstract_num(
        1,
        lvl(0, "decimal", "%1."),
        lvl(1, "lowerLetter", "%1.%2."),
        lvl(2, "upperRoman", "Điều %1.%2.%3"),
    )
    + num(1, 1)
)


# ---------- paragraph readers


def test_read_num_id_and_ilvl() -> None:
    p = paragraph('<w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="3"/></w:numPr></w:pPr>')
    assert read_num_id(p) == 3
    assert read_ilvl(p) == 1


def test_readers_return_none_when_not_written() -> None:
    # Missing must stay None (not 0): the style chain may still supply it.
    p = paragraph("<w:pPr/>")
    assert read_num_id(p) is None
    assert read_ilvl(p) is None


def test_num_id_zero_is_passed_through() -> None:
    # 0 means "numbering switched off"; deciding that is the caller's job.
    p = paragraph('<w:pPr><w:numPr><w:numId w:val="0"/></w:numPr></w:pPr>')
    assert read_num_id(p) == 0


# ---------- numbering.xml lookups


def test_read_abstract_num_id() -> None:
    assert read_abstract_num_id(SAMPLE_NUMBERING, 3) == 3


def test_read_abstract_num_id_rejects_unknown_num_id() -> None:
    with pytest.raises(ValueError, match="numId 42"):
        read_abstract_num_id(SAMPLE_NUMBERING, 42)


def test_find_level_returns_the_right_level() -> None:
    level = find_level(SAMPLE_NUMBERING, 3, 1)
    assert level.tag == qn("w:lvl")
    assert read_num_fmt(level) == "lowerLetter"


@pytest.mark.parametrize(("abstract_num_id", "ilvl"), [(3, 5), (99, 0)])
def test_find_level_rejects_missing_level(abstract_num_id: int, ilvl: int) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        find_level(SAMPLE_NUMBERING, abstract_num_id, ilvl)


def test_level_readers() -> None:
    level = level_element(lvl(1, "lowerLetter", "%2)", start=3))
    assert read_num_fmt(level) == "lowerLetter"
    assert read_lvl_text(level) == "%2)"
    assert read_start(level) == 3


def test_level_readers_return_none_when_not_written() -> None:
    level = level_element(lvl(0, None, None, start=None))
    assert read_num_fmt(level) is None
    assert read_lvl_text(level) is None
    assert read_start(level) is None


def test_find_level_override() -> None:
    assert find_level_override(OVERRIDE_NUMBERING, 7, 0) is not None
    assert find_level_override(OVERRIDE_NUMBERING, 8, 0) is None  # no override is normal


def test_override_readers() -> None:
    start_only = find_level_override(OVERRIDE_NUMBERING, 7, 0)
    with_level = find_level_override(OVERRIDE_NUMBERING, 7, 1)
    assert start_only is not None and with_level is not None

    assert read_start_override(start_only) == 5
    assert find_replacement_level(start_only) is None

    assert read_start_override(with_level) is None
    replacement = find_replacement_level(with_level)
    assert replacement is not None
    assert read_num_fmt(replacement) == "upperRoman"


# ---------- build_level_definition


def test_build_level_definition_reads_all_values() -> None:
    definition = build_level_definition(level_element(lvl(1, "lowerLetter", "%2)", start=3)))
    assert definition == LevelDefinition(num_fmt="lowerLetter", lvl_text="%2)", start=3)


def test_missing_num_fmt_means_decimal() -> None:
    # ECMA-376 default.
    definition = build_level_definition(level_element(lvl(0, None, "%1.")))
    assert definition.num_fmt == "decimal"


def test_missing_start_means_zero() -> None:
    # ECMA-376 default: 0, not 1.
    definition = build_level_definition(level_element(lvl(0, "decimal", "%1.", start=None)))
    assert definition.start == 0


def test_missing_lvl_text_stays_none() -> None:
    definition = build_level_definition(level_element(lvl(0, "decimal", None)))
    assert definition.lvl_text is None


def test_empty_lvl_text_is_kept() -> None:
    # "" is written in the file; it is not the same as missing.
    definition = build_level_definition(level_element(lvl(0, "decimal", "")))
    assert definition.lvl_text == ""


def test_num_fmt_none_is_kept() -> None:
    # "none" is an explicit value ("show no number"), not a missing one.
    definition = build_level_definition(level_element(lvl(0, "none", "")))
    assert definition.num_fmt == "none"


@pytest.mark.parametrize(
    ("num_fmt", "lvl_text", "expected"),
    [
        ("decimal", "%1.", True),
        ("lowerLetter", "%2)", True),
        ("bullet", "•", False),
        ("none", "", False),
        ("decimal", None, False),
    ],
)
def test_is_ordered(num_fmt: str, lvl_text: str | None, expected: bool) -> None:
    definition = LevelDefinition(num_fmt=num_fmt, lvl_text=lvl_text, start=1)
    assert definition.is_ordered is expected


# ---------- resolve_level_definition


def test_resolve_without_override() -> None:
    definition = resolve_level_definition(OVERRIDE_NUMBERING, 8, 0)
    assert definition == LevelDefinition(num_fmt="decimal", lvl_text="%1.", start=1)


def test_resolve_with_start_override_only() -> None:
    # "Restart at…": the abstract level, with a new start.
    definition = resolve_level_definition(OVERRIDE_NUMBERING, 7, 0)
    assert definition == LevelDefinition(num_fmt="decimal", lvl_text="%1.", start=5)


def test_resolve_with_replacement_level() -> None:
    definition = resolve_level_definition(OVERRIDE_NUMBERING, 7, 1)
    assert definition == LevelDefinition(num_fmt="upperRoman", lvl_text="%2.", start=4)


def test_replacement_level_works_without_an_abstract_level() -> None:
    # abstractNum 3 has no level 2; the override supplies it.
    definition = resolve_level_definition(OVERRIDE_NUMBERING, 7, 2)
    assert definition.num_fmt == "lowerRoman"


def test_resolve_rejects_level_missing_everywhere() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        resolve_level_definition(OVERRIDE_NUMBERING, 8, 2)


def test_resolve_rejects_unknown_num_id() -> None:
    with pytest.raises(ValueError, match="numId 42"):
        resolve_level_definition(OVERRIDE_NUMBERING, 42, 0)


def test_resolve_rejects_dangling_abstract_num_id() -> None:
    with pytest.raises(ValueError):
        resolve_level_definition(OVERRIDE_NUMBERING, 9, 0)


# ---------- ListCounter


def test_counter_counts_from_start() -> None:
    counter = ListCounter()
    assert [counter.advance(1, 0, 1)[0] for _ in range(3)] == [1, 2, 3]


def test_counter_respects_start() -> None:
    assert ListCounter().advance(1, 0, 5) == {0: 5}


def test_counter_restarts_deeper_levels() -> None:
    counter = ListCounter()
    results = [counter.advance(1, ilvl, 1) for ilvl in (0, 1, 1, 0, 1, 0)]
    assert results == [
        {0: 1},
        {0: 1, 1: 1},
        {0: 1, 1: 2},
        {0: 2},  # going back up forgets level 1 (and must not crash)
        {0: 2, 1: 1},
        {0: 3},
    ]


def test_counter_keeps_lists_independent() -> None:
    counter = ListCounter()
    results = [counter.advance(num_id, 0, 1)[0] for num_id in (3, 5, 3, 5)]
    assert results == [1, 1, 2, 2]


def test_counter_returns_a_copy() -> None:
    counter = ListCounter()
    returned = counter.advance(1, 0, 1)
    returned[0] = 99
    assert counter.advance(1, 0, 1) == {0: 2}


# ---------- resolve_marker


def definition(num_fmt: str, lvl_text: str | None) -> LevelDefinition:
    return LevelDefinition(num_fmt=num_fmt, lvl_text=lvl_text, start=1)


@pytest.mark.parametrize(
    ("lvl_text", "counts", "formats", "expected"),
    [
        ("%1.", {0: 3}, {0: "decimal"}, "3."),
        ("%2)", {0: 1, 1: 2}, {1: "lowerLetter"}, "b)"),
        ("%1.%2.", {0: 1, 1: 2}, {0: "decimal", 1: "decimal"}, "1.2."),
        ("%1.%2.%3.", {0: 1, 1: 1, 2: 1}, {0: "decimal", 1: "lowerLetter", 2: "upperRoman"}, "1.a.I."),
        ("Điều %1.", {0: 7}, {0: "decimal"}, "Điều 7."),
        ("Chương %1", {0: 4}, {0: "upperRoman"}, "Chương IV"),
        ("(%1)", {0: 1}, {0: "decimal"}, "(1)"),
        (" %1 ", {0: 1}, {0: "decimal"}, " 1 "),  # surrounding text kept exactly
    ],
)
def test_resolve_marker(
    lvl_text: str, counts: dict[int, int], formats: dict[int, str], expected: str
) -> None:
    assert resolve_marker(definition("decimal", lvl_text), counts, formats) == expected


def test_resolve_marker_none_format_shows_nothing() -> None:
    assert resolve_marker(definition("none", "%1."), {0: 1}, {0: "none"}) is None


def test_resolve_marker_without_lvl_text_shows_nothing() -> None:
    assert resolve_marker(definition("decimal", None), {0: 1}, {0: "decimal"}) is None


def test_resolve_marker_bullet_returns_its_character() -> None:
    assert resolve_marker(definition("bullet", "◦"), {0: 1}, {}) == "◦"


def test_resolve_marker_missing_count() -> None:
    with pytest.raises(ValueError, match=r"no count for level 0 \(%1\)"):
        resolve_marker(definition("decimal", "%1.%2."), {1: 1}, {0: "decimal", 1: "decimal"})


def test_resolve_marker_missing_format() -> None:
    with pytest.raises(ValueError, match=r"no format for level 1 \(%2\)"):
        resolve_marker(definition("decimal", "%2."), {1: 1}, {})


# ---------- Numberer


def markers(numberer: Numberer, calls: list[tuple[int, int]]) -> list[str | None]:
    return [numberer.next_list_info(num_id, ilvl).displayed_marker for num_id, ilvl in calls]


def test_numberer_sample_numbered_list() -> None:
    numberer = Numberer(SAMPLE_NUMBERING)
    assert markers(numberer, [(3, 0), (3, 1), (3, 1), (3, 0), (3, 0)]) == [
        "1.",
        "a)",
        "b)",
        "2.",
        "3.",
    ]


def test_numberer_sample_lists_are_independent() -> None:
    numberer = Numberer(SAMPLE_NUMBERING)
    assert markers(numberer, [(3, 0), (5, 0), (3, 0), (5, 0)]) == ["1.", "1.", "2.", "2."]


def test_numberer_sample_bullets() -> None:
    numberer = Numberer(SAMPLE_NUMBERING)
    first = numberer.next_list_info(2, 0)
    nested = numberer.next_list_info(2, 1)
    assert first.displayed_marker == ""  # kept raw for now (see TODO)
    assert nested.displayed_marker == "◦"
    assert first.is_ordered is False
    assert nested.is_ordered is False


def test_numberer_list_info_fields() -> None:
    info = Numberer(SAMPLE_NUMBERING).next_list_info(3, 0)
    assert info.id == "3"  # a string, so it survives the JSON round trip unchanged
    assert info.is_ordered is True
    assert info.nested_level == 0


def test_numberer_multilevel_templates() -> None:
    numberer = Numberer(MULTILEVEL_NUMBERING)
    assert markers(numberer, [(1, 0), (1, 1), (1, 2), (1, 2), (1, 1), (1, 2)]) == [
        "1.",
        "1.a.",
        "Điều 1.a.I",
        "Điều 1.a.II",
        "1.b.",
        "Điều 1.b.I",  # level 2 restarts under the new level 1
    ]


def test_numberer_uses_overrides() -> None:
    numberer = Numberer(OVERRIDE_NUMBERING)
    assert markers(numberer, [(7, 0), (7, 1), (7, 0)]) == ["5.", "IV.", "6."]


def test_numberer_list_starting_at_a_sub_level_raises() -> None:
    # "%1.%2." needs a count for level 0 that does not exist yet.
    with pytest.raises(ValueError, match="no count for level 0"):
        Numberer(MULTILEVEL_NUMBERING).next_list_info(1, 1)


def test_numberers_do_not_share_counts() -> None:
    # One Numberer per document: a new one starts from scratch.
    first = Numberer(SAMPLE_NUMBERING)
    markers(first, [(3, 0), (3, 0)])
    assert markers(Numberer(SAMPLE_NUMBERING), [(3, 0)]) == ["1."]