import pytest
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml.etree import _Element

from document_processor.adapters.docx.styles import read_jc

W = nsdecls(
    "w"
)  # 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def paragraph(inner: str = "") -> _Element:
    """A <w:p> as python-docx builds it (a CT_P, which knows the w: prefix)."""
    return parse_xml(f"<w:p {W}>{inner}</w:p>")


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
