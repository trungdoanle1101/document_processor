"""Tests for the low-level XML helpers of the docx adapter.

The elements are built from small XML strings, so no .docx file is needed.
"""

import pytest
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml import etree
from lxml.etree import _Element

from document_processor.adapters.docx._xml import (
    read_jc,
    xpath_element_or_none,
    xpath_int_or_none,
    xpath_str_or_none,
)

W = nsdecls(
    "w"
)  # 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def paragraph(inner: str = "") -> _Element:
    """A <w:p> as python-docx builds it (a CT_P, which knows the w: prefix)."""
    return parse_xml(f"<w:p {W}>{inner}</w:p>")


def plain_element(xml: str) -> _Element:
    """An element parsed by plain lxml: no python-docx class, no built-in w: prefix."""
    return etree.fromstring(xml)


# ---------- xpath_str_or_none


def test_str_returns_the_single_value() -> None:
    p = paragraph('<w:pPr><w:jc w:val="center"/></w:pPr>')
    assert xpath_str_or_none(p, "./w:pPr/w:jc/@w:val") == "center"


def test_str_returns_a_plain_str() -> None:
    # lxml's "smart strings" are str subclasses that keep the XML tree alive.
    p = paragraph('<w:pPr><w:jc w:val="center"/></w:pPr>')
    value = xpath_str_or_none(p, "./w:pPr/w:jc/@w:val")
    assert type(value) is str


@pytest.mark.parametrize(
    "inner",
    ["", "<w:pPr/>", '<w:pPr><w:pStyle w:val="BodyText"/></w:pPr>'],
    ids=["no-pPr", "empty-pPr", "pPr-without-jc"],
)
def test_str_returns_none_when_missing(inner: str) -> None:
    assert xpath_str_or_none(paragraph(inner), "./w:pPr/w:jc/@w:val") is None


def test_str_rejects_multiple_results() -> None:
    # Invalid for Word, but nothing stops a broken file from containing it.
    p = paragraph('<w:pPr><w:jc w:val="left"/><w:jc w:val="center"/></w:pPr>')
    with pytest.raises(ValueError, match=r"Expected 0 or 1 results, got 2"):
        xpath_str_or_none(p, "./w:pPr/w:jc/@w:val")


def test_str_rejects_a_path_that_selects_an_element() -> None:
    # Forgetting "/@w:val" selects the <w:jc> element, not its value.
    p = paragraph('<w:pPr><w:jc w:val="center"/></w:pPr>')
    with pytest.raises(TypeError, match=r"XPath: \./w:pPr/w:jc\. Expected a str"):
        xpath_str_or_none(p, "./w:pPr/w:jc")


def test_str_rejects_a_path_that_does_not_return_a_list() -> None:
    # count(...) returns a number, not a list of matches.
    with pytest.raises(TypeError, match="Expected list, got float"):
        xpath_str_or_none(paragraph(), "count(./w:r)")


def test_str_works_on_a_plain_lxml_element() -> None:
    # Elements python-docx has no class for (e.g. <w:lvl> in numbering.xml)
    # only work because the helper passes the namespace map itself.
    lvl = plain_element(
        f'<w:lvl {W} w:ilvl="1"><w:numFmt w:val="lowerLetter"/></w:lvl>'
    )
    assert xpath_str_or_none(lvl, "./w:numFmt/@w:val") == "lowerLetter"


# ---------- xpath_int_or_none


def test_int_converts_the_value() -> None:
    p = paragraph('<w:pPr><w:numPr><w:ilvl w:val="1"/></w:numPr></w:pPr>')
    assert xpath_int_or_none(p, "./w:pPr/w:numPr/w:ilvl/@w:val") == 1


def test_int_returns_none_when_missing() -> None:
    p = paragraph("<w:pPr><w:numPr/></w:pPr>")
    assert xpath_int_or_none(p, "./w:pPr/w:numPr/w:ilvl/@w:val") is None


def test_int_rejects_a_non_number() -> None:
    p = paragraph('<w:pPr><w:numPr><w:ilvl w:val="abc"/></w:numPr></w:pPr>')
    with pytest.raises(ValueError, match="'abc'"):
        xpath_int_or_none(p, "./w:pPr/w:numPr/w:ilvl/@w:val")


def test_int_keeps_the_original_error_as_cause() -> None:
    p = paragraph('<w:pPr><w:numPr><w:ilvl w:val="abc"/></w:numPr></w:pPr>')
    with pytest.raises(ValueError) as excinfo:
        xpath_int_or_none(p, "./w:pPr/w:numPr/w:ilvl/@w:val")
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_int_passes_on_multiple_results_error() -> None:
    p = paragraph(
        '<w:pPr><w:numPr><w:ilvl w:val="0"/><w:ilvl w:val="1"/></w:numPr></w:pPr>'
    )
    with pytest.raises(ValueError, match="Expected 0 or 1 results, got 2"):
        xpath_int_or_none(p, "./w:pPr/w:numPr/w:ilvl/@w:val")


# ---------- read_jc


@pytest.mark.parametrize("value", ["left", "start", "center", "both", "distribute"])
def test_read_jc_returns_the_raw_value(value: str) -> None:
    # Raw means raw: no mapping, so LibreOffice's "start" stays "start".
    p = paragraph(f'<w:pPr><w:jc w:val="{value}"/></w:pPr>')
    assert read_jc(p) == value


def test_read_jc_returns_none_when_not_set() -> None:
    assert read_jc(paragraph("<w:pPr/>")) is None


def test_read_jc_works_on_a_style() -> None:
    # Styles have a <w:pPr> too, so the same reader serves the style chain.
    style = parse_xml(
        f'<w:style {W} w:type="paragraph" w:styleId="Heading1">'
        '<w:pPr><w:jc w:val="center"/></w:pPr>'
        "</w:style>"
    )
    assert read_jc(style) == "center"


# ---------- xpath_element_or_none


def numbering(inner: str) -> _Element:
    """A <w:numbering> root, as in word/numbering.xml."""
    return parse_xml(f"<w:numbering {W}>{inner}</w:numbering>")


LVL_PATH = './w:abstractNum[@w:abstractNumId="3"]/w:lvl[@w:ilvl="1"]'


def test_element_returns_the_single_element() -> None:
    root = numbering(
        '<w:abstractNum w:abstractNumId="3">'
        '<w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/></w:lvl>'
        '<w:lvl w:ilvl="1"><w:numFmt w:val="lowerLetter"/></w:lvl>'
        "</w:abstractNum>"
    )
    lvl = xpath_element_or_none(root, LVL_PATH)
    assert lvl is not None
    assert lvl.tag == qn("w:lvl")
    # It is the right level, not just any level:
    assert xpath_str_or_none(lvl, "./w:numFmt/@w:val") == "lowerLetter"


def test_element_returns_none_when_missing() -> None:
    root = numbering(
        '<w:abstractNum w:abstractNumId="3"><w:lvl w:ilvl="0"/></w:abstractNum>'
    )
    assert xpath_element_or_none(root, LVL_PATH) is None


def test_element_rejects_multiple_results() -> None:
    # Two definitions of the same level: invalid, and ambiguous.
    root = numbering(
        '<w:abstractNum w:abstractNumId="3">'
        '<w:lvl w:ilvl="1"/><w:lvl w:ilvl="1"/>'
        "</w:abstractNum>"
    )
    with pytest.raises(ValueError, match="Expected 0 or 1 results, got 2"):
        xpath_element_or_none(root, LVL_PATH)


def test_element_rejects_a_path_that_selects_an_attribute() -> None:
    # The mirror image of the str helper's test: "/@w:ilvl" selects a value.
    root = numbering(
        '<w:abstractNum w:abstractNumId="3"><w:lvl w:ilvl="1"/></w:abstractNum>'
    )
    with pytest.raises(TypeError, match="Expected an _Element, got '1'"):
        xpath_element_or_none(root, f"{LVL_PATH}/@w:ilvl")
