"""Tests for paragraph style and alignment resolution in the docx adapter.

Everything is built from small XML strings: a <w:styles> root (as in
word/styles.xml) and <w:p> paragraphs. No .docx file is needed.
"""

import pytest
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from lxml.etree import _Element

from document_processor.adapters.docx.styles import (
    Stylist,
    find_default_paragraph_style,
    find_p_pr_defaults,
    map_alignment,
    read_jc,
    read_p_style,
    read_style_based_on,
    read_style_id,
    read_style_name,
)
from document_processor.core.models.physical import ParagraphAlignment, ParagraphFormat

W = nsdecls("w")

LEFT = ParagraphAlignment.LEFT
RIGHT = ParagraphAlignment.RIGHT
CENTER = ParagraphAlignment.CENTER
JUSTIFIED = ParagraphAlignment.JUSTIFIED


# ---------- builders


def jc_xml(jc: str | None) -> str:
    return f'<w:pPr><w:jc w:val="{jc}"/></w:pPr>' if jc is not None else ""


def style(
    style_id: str | None,
    *,
    name: str | None = None,
    based_on: str | None = None,
    jc: str | None = None,
    style_type: str = "paragraph",
    default: str | None = None,
) -> str:
    """One <w:style>. `name` defaults to the id; pass name="" to leave <w:name> out."""
    id_attr = f' w:styleId="{style_id}"' if style_id is not None else ""
    default_attr = f' w:default="{default}"' if default is not None else ""
    if name is None:
        name = style_id or ""
    name_xml = f'<w:name w:val="{name}"/>' if name else ""
    based_on_xml = f'<w:basedOn w:val="{based_on}"/>' if based_on is not None else ""
    return (
        f'<w:style w:type="{style_type}"{default_attr}{id_attr}>'
        f"{name_xml}{based_on_xml}{jc_xml(jc)}"
        "</w:style>"
    )


def styles_root(*styles: str, default_jc: str | None = None) -> _Element:
    """A <w:styles> root, optionally with a document-default jc."""
    doc_defaults = (
        f"<w:docDefaults><w:pPrDefault>{jc_xml(default_jc)}</w:pPrDefault></w:docDefaults>"
        if default_jc is not None
        else ""
    )
    return parse_xml(f"<w:styles {W}>{doc_defaults}{''.join(styles)}</w:styles>")


def paragraph(p_style: str | None = None, jc: str | None = None) -> _Element:
    p_style_xml = f'<w:pStyle w:val="{p_style}"/>' if p_style is not None else ""
    jc_inner = f'<w:jc w:val="{jc}"/>' if jc is not None else ""
    ppr = f"<w:pPr>{p_style_xml}{jc_inner}</w:pPr>" if (p_style_xml or jc_inner) else ""
    return parse_xml(f"<w:p {W}>{ppr}<w:r><w:t>text</w:t></w:r></w:p>")


def resolve(root: _Element | None, p: _Element) -> ParagraphFormat:
    return Stylist(root).resolve_paragraph_format(p)


# A small realistic chain used by several tests:
#   Normal (default, no jc) <- BodyText (both) <- DieuHeading (no jc)
#   Title (center), not based on anything
CHAIN = (
    style("Normal", default="1"),
    style("BodyText", based_on="Normal", jc="both"),
    style("DieuHeading", based_on="BodyText"),
    style("Title", jc="center"),
)


# ---------- map_alignment


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("left", LEFT),
        ("start", LEFT),  # what LibreOffice writes
        ("right", RIGHT),
        ("end", RIGHT),
        ("center", CENTER),
        ("both", JUSTIFIED),
        ("distribute", JUSTIFIED),
    ],
)
def test_map_alignment(raw: str, expected: ParagraphAlignment) -> None:
    assert map_alignment(raw) is expected


@pytest.mark.parametrize("raw", ["numTab", "highKashida", "Center", ""])
def test_map_alignment_rejects_unknown_values(raw: str) -> None:
    with pytest.raises(ValueError, match=f"Unknown alignment type {raw!r}"):
        map_alignment(raw)


# ---------- readers


def test_read_jc_on_paragraph_and_style() -> None:
    assert read_jc(paragraph(jc="center")) == "center"
    assert read_jc(parse_xml(f"<w:style {W}>{jc_xml('both')}</w:style>")) == "both"


def test_read_jc_keeps_the_raw_value() -> None:
    # Raw means raw: mapping happens later, in map_alignment.
    assert read_jc(paragraph(jc="start")) == "start"


def test_read_jc_returns_none_when_not_set() -> None:
    assert read_jc(paragraph()) is None


def test_read_p_style() -> None:
    assert read_p_style(paragraph(p_style="Heading1")) == "Heading1"
    assert read_p_style(paragraph()) is None


def test_style_readers() -> None:
    el = parse_xml(f"<w:styles {W}>{style('Dieu', name='heading 2', based_on='Normal')}</w:styles>")[0]
    assert read_style_id(el) == "Dieu"
    assert read_style_name(el) == "heading 2"
    assert read_style_based_on(el) == "Normal"


def test_style_readers_return_none_when_missing() -> None:
    el = parse_xml(f"<w:styles {W}>{style(None, name='')}</w:styles>")[0]
    assert read_style_id(el) is None
    assert read_style_name(el) is None
    assert read_style_based_on(el) is None


# ---------- find_default_paragraph_style


def test_default_style_found() -> None:
    root = styles_root(style("Normal", default="1"), style("Other"))
    found = find_default_paragraph_style(root)
    assert found is not None
    assert read_style_id(found) == "Normal"


@pytest.mark.parametrize("value", ["1", "true", "on"])
def test_default_style_accepts_all_on_values(value: str) -> None:
    root = styles_root(style("Normal", default=value))
    assert find_default_paragraph_style(root) is not None


def test_default_style_ignores_off_values() -> None:
    root = styles_root(style("Normal", default="0"))
    assert find_default_paragraph_style(root) is None


def test_no_default_style() -> None:
    assert find_default_paragraph_style(styles_root(style("Normal"))) is None


def test_default_style_ignores_other_style_types() -> None:
    # Character and table styles have their own defaults.
    root = styles_root(
        style("DefaultParagraphFont", style_type="character", default="1"),
        style("TableNormal", style_type="table", default="1"),
    )
    assert find_default_paragraph_style(root) is None


def test_last_default_style_wins() -> None:
    root = styles_root(
        style("First", default="1"),
        style("Middle"),
        style("Second", default="1"),
    )
    found = find_default_paragraph_style(root)
    assert found is not None
    assert read_style_id(found) == "Second"


# ---------- find_p_pr_defaults


def test_p_pr_defaults_found() -> None:
    p_pr_default = find_p_pr_defaults(styles_root(default_jc="both"))
    assert p_pr_default is not None
    # read_jc works on it, because its path starts at ./w:pPr
    assert read_jc(p_pr_default) == "both"


def test_p_pr_defaults_missing() -> None:
    assert find_p_pr_defaults(styles_root()) is None


# ---------- Stylist: building


def test_duplicate_style_ids_rejected() -> None:
    root = styles_root(style("Normal"), style("Normal"))
    with pytest.raises(ValueError, match="Duplicate style_id found: 'Normal'"):
        Stylist(root)


def test_style_without_id_is_ignored() -> None:
    root = styles_root(style(None, name="orphan", jc="center"), style("Normal", default="1"))
    assert resolve(root, paragraph()) == ParagraphFormat(alignment=LEFT, style_name="Normal")


def test_unknown_doc_default_jc_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="Unknown alignment type 'weird'"):
        Stylist(styles_root(default_jc="weird"))


# ---------- Stylist: alignment, step by step


def test_alignment_direct_jc_wins_over_style() -> None:
    root = styles_root(*CHAIN)
    assert resolve(root, paragraph(p_style="Title", jc="right")).alignment is RIGHT


def test_alignment_from_style() -> None:
    root = styles_root(*CHAIN)
    assert resolve(root, paragraph(p_style="Title")).alignment is CENTER


def test_alignment_from_base_style() -> None:
    root = styles_root(*CHAIN)
    # DieuHeading has no jc; its base BodyText has "both".
    assert resolve(root, paragraph(p_style="DieuHeading")).alignment is JUSTIFIED


def test_alignment_nearest_style_in_chain_wins() -> None:
    root = styles_root(
        style("Base", jc="right"),
        style("Middle", based_on="Base", jc="center"),
        style("Leaf", based_on="Middle"),
    )
    assert resolve(root, paragraph(p_style="Leaf")).alignment is CENTER


def test_alignment_from_doc_defaults() -> None:
    root = styles_root(style("Normal", default="1"), default_jc="both")
    assert resolve(root, paragraph()).alignment is JUSTIFIED


def test_alignment_style_wins_over_doc_defaults() -> None:
    root = styles_root(*CHAIN, default_jc="right")
    assert resolve(root, paragraph(p_style="Title")).alignment is CENTER


def test_alignment_falls_back_to_left() -> None:
    root = styles_root(*CHAIN)
    # No direct jc, Normal has none, no docDefaults.
    assert resolve(root, paragraph()).alignment is LEFT


def test_alignment_libreoffice_start_maps_to_left() -> None:
    root = styles_root(style("Normal", default="1", jc="start"))
    assert resolve(root, paragraph()).alignment is LEFT


def test_alignment_from_default_style_when_no_p_style() -> None:
    root = styles_root(style("Normal", default="1", jc="center"))
    assert resolve(root, paragraph()).alignment is CENTER


def test_alignment_from_default_style_when_p_style_is_unknown() -> None:
    root = styles_root(style("Normal", default="1", jc="center"))
    assert resolve(root, paragraph(p_style="DoesNotExist")).alignment is CENTER


# ---------- Stylist: broken chains


def test_missing_based_on_ends_the_chain() -> None:
    root = styles_root(style("Leaf", based_on="Gone"), default_jc="right")
    # The chain stops at the missing id and continues with docDefaults.
    assert resolve(root, paragraph(p_style="Leaf")).alignment is RIGHT


def test_based_on_cycle_terminates() -> None:
    root = styles_root(
        style("A", based_on="B"),
        style("B", based_on="A"),
        default_jc="center",
    )
    assert resolve(root, paragraph(p_style="A")).alignment is CENTER


def test_self_referencing_based_on_terminates() -> None:
    root = styles_root(style("A", based_on="A"))
    assert resolve(root, paragraph(p_style="A")).alignment is LEFT


def test_cycle_still_finds_jc_before_looping() -> None:
    root = styles_root(
        style("A", based_on="B"),
        style("B", based_on="A", jc="right"),
    )
    assert resolve(root, paragraph(p_style="A")).alignment is RIGHT


# ---------- Stylist: unknown values


def test_unknown_jc_on_paragraph_raises() -> None:
    with pytest.raises(ValueError, match="Unknown alignment type 'numTab'"):
        resolve(styles_root(*CHAIN), paragraph(jc="numTab"))


def test_unknown_jc_on_style_raises() -> None:
    root = styles_root(style("Odd", jc="lowKashida"))
    with pytest.raises(ValueError, match="Unknown alignment type 'lowKashida'"):
        resolve(root, paragraph(p_style="Odd"))


# ---------- Stylist: style_name


def test_style_name_from_p_style() -> None:
    root = styles_root(style("Heading1", name="heading 1"), style("Normal", default="1"))
    assert resolve(root, paragraph(p_style="Heading1")).style_name == "heading 1"


def test_style_name_is_the_name_not_the_id() -> None:
    # Ids can be generated by the editor; the name is what we store.
    root = styles_root(style("u1", name="heading 2"))
    assert resolve(root, paragraph(p_style="u1")).style_name == "heading 2"


def test_style_name_falls_back_to_default_when_no_p_style() -> None:
    root = styles_root(style("Normal", default="1"))
    assert resolve(root, paragraph()).style_name == "Normal"


def test_style_name_falls_back_to_default_when_p_style_is_unknown() -> None:
    root = styles_root(style("Normal", default="1"))
    assert resolve(root, paragraph(p_style="DoesNotExist")).style_name == "Normal"


def test_style_name_none_without_default_style() -> None:
    root = styles_root(style("Heading1"))
    assert resolve(root, paragraph()).style_name is None


def test_style_name_none_when_p_style_unknown_and_no_default() -> None:
    root = styles_root(style("Heading1"))
    fmt = resolve(root, paragraph(p_style="DoesNotExist"))
    assert fmt == ParagraphFormat(alignment=LEFT, style_name=None)


def test_style_name_from_last_default_style() -> None:
    root = styles_root(style("First", default="1"), style("Second", default="1"))
    assert resolve(root, paragraph()).style_name == "Second"


def test_style_name_none_when_style_has_no_name() -> None:
    root = styles_root(style("Nameless", name=""))
    assert resolve(root, paragraph(p_style="Nameless")).style_name is None


# ---------- Stylist without styles.xml


def test_no_styles_uses_paragraph_jc() -> None:
    assert resolve(None, paragraph(jc="center")) == ParagraphFormat(
        alignment=CENTER, style_name=None
    )


def test_no_styles_falls_back_to_left() -> None:
    assert resolve(None, paragraph()) == ParagraphFormat(alignment=LEFT, style_name=None)


def test_no_styles_ignores_p_style() -> None:
    # A pStyle pointing nowhere is not an error.
    assert resolve(None, paragraph(p_style="Heading1")) == ParagraphFormat(
        alignment=LEFT, style_name=None
    )


# ---------- the full example


@pytest.mark.parametrize(
    ("p_style", "jc", "expected"),
    [
        ("DieuHeading", "center", ParagraphFormat(alignment=CENTER, style_name="DieuHeading")),
        ("DieuHeading", None, ParagraphFormat(alignment=JUSTIFIED, style_name="DieuHeading")),
        (None, None, ParagraphFormat(alignment=LEFT, style_name="Normal")),
    ],
    ids=["direct-jc", "inherited-from-base", "default-style-then-left"],
)
def test_worked_example(p_style: str | None, jc: str | None, expected: ParagraphFormat) -> None:
    assert resolve(styles_root(*CHAIN), paragraph(p_style=p_style, jc=jc)) == expected