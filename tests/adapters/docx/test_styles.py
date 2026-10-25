"""Tests for paragraph style, alignment and bold resolution in the docx adapter.

Everything is built from small XML strings: a <w:styles> root (as in
word/styles.xml) and <w:p> paragraphs. No .docx file is needed.

Bold of a run: the first level that sets w:b decides — the run itself, its
character style (and what that is based on), its paragraph's style (or the
default paragraph style; and what that is based on), the document default —
otherwise not bold. In the spec w:b in styles is a toggle; this resolver lets
the nearest level win instead.
"""

import pytest
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml.etree import _Element

from document_processor.adapters.docx.styles import (
    Stylist,
    find_default_paragraph_style,
    find_p_pr_defaults,
    map_alignment,
    read_bold,
    read_jc,
    read_p_style,
    read_r_style,
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
    bold: str | None = None,
) -> str:
    """One <w:style>. `name` defaults to the id; pass name="" to leave <w:name> out.

    `bold`: None (no w:b), "" (bare <w:b/>), or a w:val.
    """
    id_attr = f' w:styleId="{style_id}"' if style_id is not None else ""
    default_attr = f' w:default="{default}"' if default is not None else ""
    if name is None:
        name = style_id or ""
    name_xml = f'<w:name w:val="{name}"/>' if name else ""
    based_on_xml = f'<w:basedOn w:val="{based_on}"/>' if based_on is not None else ""
    return (
        f'<w:style w:type="{style_type}"{default_attr}{id_attr}>'
        f"{name_xml}{based_on_xml}{jc_xml(jc)}{bold_xml(bold)}"
        "</w:style>"
    )


def bold_xml(bold: str | None) -> str:
    """A <w:rPr> holding w:b. None: nothing; "": bare <w:b/>; otherwise the w:val."""
    if bold is None:
        return ""
    val = f' w:val="{bold}"' if bold else ""
    return f"<w:rPr><w:b{val}/></w:rPr>"


def styles_root(*styles: str, default_jc: str | None = None, default_bold: str | None = None) -> _Element:
    """A <w:styles> root, optionally with a document-default jc and bold."""
    p_pr_default = f"<w:pPrDefault>{jc_xml(default_jc)}</w:pPrDefault>" if default_jc is not None else ""
    r_pr_default = f"<w:rPrDefault>{bold_xml(default_bold)}</w:rPrDefault>" if default_bold is not None else ""
    doc_defaults = (
        f"<w:docDefaults>{r_pr_default}{p_pr_default}</w:docDefaults>"
        if (p_pr_default or r_pr_default)
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


# ======================================================================
# Bold
# ======================================================================


def run(
    *,
    bold: str | None = None,
    r_style: str | None = None,
    p_style: str | None = None,
    in_hyperlink: bool = False,
) -> _Element:
    """A <w:r> inside a <w:p>; returns the run."""
    r_style_xml = f'<w:rStyle w:val="{r_style}"/>' if r_style is not None else ""
    b_xml = ""
    if bold is not None:
        b_xml = f'<w:b w:val="{bold}"/>' if bold else "<w:b/>"
    r_pr = f"<w:rPr>{r_style_xml}{b_xml}</w:rPr>" if (r_style_xml or b_xml) else ""
    r_xml = f"<w:r>{r_pr}<w:t>TT</w:t></w:r>"
    if in_hyperlink:
        r_xml = f"<w:hyperlink>{r_xml}</w:hyperlink>"
    p_pr = f'<w:pPr><w:pStyle w:val="{p_style}"/></w:pPr>' if p_style is not None else ""
    p = parse_xml(f"<w:p {W}>{p_pr}{r_xml}</w:p>")
    found = p.find(f".//{qn('w:r')}")
    assert found is not None
    return found


def is_bold(root: _Element | None, r: _Element) -> bool:
    return Stylist(root).resolve_run_is_bold(r)


def character_style(style_id: str, **kwargs: str) -> str:
    return style(style_id, style_type="character", **kwargs)


# ---------- readers


def test_read_bold_on_a_run_a_style_and_rpr_default() -> None:
    assert read_bold(run(bold="")) is True
    assert read_bold(parse_xml(f"<w:style {W}>{bold_xml('0')}</w:style>")) is False
    assert read_bold(parse_xml(f"<w:rPrDefault {W}>{bold_xml('')}</w:rPrDefault>")) is True


def test_read_bold_is_none_when_not_set() -> None:
    assert read_bold(run()) is None
    assert read_bold(run(r_style="Strong")) is None  # rPr without w:b


def test_read_r_style() -> None:
    assert read_r_style(run(r_style="Strong")) == "Strong"
    assert read_r_style(run()) is None


# ---------- direct formatting


def test_plain_run_without_styles_is_not_bold() -> None:
    assert is_bold(None, run()) is False


@pytest.mark.parametrize("value", ["", "1", "true", "on"], ids=["bare", "1", "true", "on"])
def test_direct_bold_on(value: str) -> None:
    assert is_bold(None, run(bold=value)) is True


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_direct_bold_off(value: str) -> None:
    assert is_bold(None, run(bold=value)) is False


def test_direct_bold_unknown_value_rejected() -> None:
    with pytest.raises(ValueError, match="Unrecognized w:b/@w:val 'very'"):
        is_bold(None, run(bold="very"))


def test_direct_off_wins_over_a_bold_paragraph_style() -> None:
    # The LibreOffice pattern: Normal is bold, plain runs switch it off.
    root = styles_root(style("Normal", default="1", bold=""))
    assert is_bold(root, run(bold="false", p_style="Normal")) is False


def test_direct_on_wins_over_a_plain_paragraph_style() -> None:
    root = styles_root(style("Normal", default="1", bold="0"))
    assert is_bold(root, run(bold="", p_style="Normal")) is True


# ---------- character style


def test_bold_from_the_character_style() -> None:
    root = styles_root(character_style("Strong", bold=""))
    assert is_bold(root, run(r_style="Strong")) is True


def test_bold_from_the_character_style_base() -> None:
    root = styles_root(character_style("Strong", bold=""), character_style("Emphasis", based_on="Strong"))
    assert is_bold(root, run(r_style="Emphasis")) is True


def test_character_style_off_wins_over_a_bold_paragraph_style() -> None:
    root = styles_root(style("Normal", default="1", bold=""), character_style("Plain", bold="0"))
    assert is_bold(root, run(r_style="Plain", p_style="Normal")) is False


def test_character_style_without_bold_falls_through_to_the_paragraph_style() -> None:
    root = styles_root(style("Normal", default="1", bold=""), character_style("Link"))
    assert is_bold(root, run(r_style="Link", p_style="Normal")) is True


def test_unknown_character_style_falls_through() -> None:
    root = styles_root(style("Normal", default="1", bold=""))
    assert is_bold(root, run(r_style="Missing", p_style="Normal")) is True


def test_bold_character_style_on_a_bold_paragraph_style_stays_bold() -> None:
    # The spec would toggle this off; the nearest-level rule keeps it bold (documented simplification).
    root = styles_root(style("Normal", default="1", bold=""), character_style("Strong", bold=""))
    assert is_bold(root, run(r_style="Strong", p_style="Normal")) is True


# ---------- paragraph style


def test_bold_from_the_paragraph_style() -> None:
    root = styles_root(style("TableHeading", bold=""))
    assert is_bold(root, run(p_style="TableHeading")) is True


def test_bold_from_the_paragraph_style_base() -> None:
    root = styles_root(style("TableHeading", bold=""), style("TableHeadingCenter", based_on="TableHeading"))
    assert is_bold(root, run(p_style="TableHeadingCenter")) is True


def test_nearest_paragraph_style_in_the_chain_wins() -> None:
    root = styles_root(style("Normal", bold=""), style("BodyText", based_on="Normal", bold="false"))
    assert is_bold(root, run(p_style="BodyText")) is False


def test_default_paragraph_style_without_p_style() -> None:
    root = styles_root(style("Normal", default="1", bold=""))
    assert is_bold(root, run()) is True


def test_default_paragraph_style_when_p_style_is_unknown() -> None:
    root = styles_root(style("Normal", default="1", bold=""))
    assert is_bold(root, run(p_style="Missing")) is True


def test_run_inside_a_hyperlink_finds_its_paragraph() -> None:
    root = styles_root(style("Normal", default="1"), style("TableHeading", bold=""))
    assert is_bold(root, run(p_style="TableHeading", in_hyperlink=True)) is True


def test_run_outside_a_paragraph_uses_the_default_paragraph_style() -> None:
    root = styles_root(style("Normal", default="1", bold=""))
    bare_run = parse_xml(f"<w:r {W}><w:t>TT</w:t></w:r>")
    assert is_bold(root, bare_run) is True


# ---------- document default


def test_bold_from_the_document_default() -> None:
    root = styles_root(style("Normal", default="1"), default_bold="")
    assert is_bold(root, run()) is True


def test_paragraph_style_off_wins_over_a_bold_document_default() -> None:
    root = styles_root(style("Normal", default="1", bold="0"), default_bold="")
    assert is_bold(root, run()) is False


def test_document_default_without_bold() -> None:
    root = parse_xml(f"<w:styles {W}><w:docDefaults><w:rPrDefault><w:rPr/></w:rPrDefault></w:docDefaults></w:styles>")
    assert is_bold(root, run()) is False


def test_nothing_set_anywhere_is_not_bold() -> None:
    assert is_bold(styles_root(*CHAIN), run(p_style="DieuHeading")) is False


# ---------- broken chains and no styles.xml


def test_bold_based_on_cycle_terminates() -> None:
    root = styles_root(style("A", based_on="B"), style("B", based_on="A"), default_bold="")
    assert is_bold(root, run(p_style="A")) is True  # the cycle says nothing; the document default decides


def test_cycle_still_finds_bold_before_looping() -> None:
    root = styles_root(style("A", based_on="B"), style("B", based_on="A", bold=""))
    assert is_bold(root, run(p_style="A")) is True


def test_no_styles_ignores_r_style_and_p_style() -> None:
    assert is_bold(None, run(r_style="Strong", p_style="Heading1")) is False


# ---------- the real document


def test_libreoffice_table_header() -> None:
    # Normal is bold; header runs say nothing, body runs switch bold off.
    root = styles_root(style("Normal", bold=""))
    header = run(p_style="Normal")
    body = run(p_style="Normal", bold="false")
    assert (is_bold(root, header), is_bold(root, body)) == (True, False)