"""End-to-end tests for the docx adapter.

Each test builds a small .docx in memory from raw XML parts, so every case
(no styles, no numbering, tables, broken values) is under the test's control
and no binary fixtures are needed. A last test runs on the sample file in
data/ when it is present.
"""

import io
import zipfile
from pathlib import Path
from typing import Any

import pytest
from lxml import etree

from document_processor.adapters.docx.adapter import DocxAdapter
from document_processor.core.identity import compute_hash
from document_processor.core.models.physical import (
    BlockType,
    DocSourceFormat,
    ParagraphAlignment,
    PhysicalDocument,
)

from tests.drawing_xml import R_NS, chart, drawing, pic

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f'xmlns:w="{W_NS}"'
FILENAME = "test.docx"

# ---------- building a .docx in memory

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Default Extension="jpeg" ContentType="image/jpeg"/>
  <Default Extension="emf" ContentType="image/x-emf"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""

PACKAGE_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

REL_STYLES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
REL_NUMBERING = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering"
REL_IMAGE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"

SECT_PR = '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr>'


def build_docx(
    body: str,
    *,
    styles: str | None = None,
    numbering: str | None = None,
    images: dict[str, tuple[str, bytes]] | None = None,
) -> bytes:
    """A minimal .docx. `body` is the inside of <w:body>; a sectPr is appended.

    `styles` / `numbering` are the inside of <w:styles> / <w:numbering>;
    None leaves the part (and its relationship) out of the package.
    `images` maps a relationship id to (file name in word/media, bytes);
    use ids from rId10 up so they don't clash with styles and numbering.
    """
    rels = []
    if styles is not None:
        rels.append(f'<Relationship Id="rId1" Type="{REL_STYLES}" Target="styles.xml"/>')
    if numbering is not None:
        rels.append(f'<Relationship Id="rId2" Type="{REL_NUMBERING}" Target="numbering.xml"/>')
    for rid, (name, _) in (images or {}).items():
        rels.append(f'<Relationship Id="{rid}" Type="{REL_IMAGE}" Target="media/{name}"/>')
    document_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(rels)}</Relationships>"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", PACKAGE_RELS)
        z.writestr("word/_rels/document.xml.rels", document_rels)
        z.writestr("word/document.xml", f"<w:document {W}><w:body>{body}{SECT_PR}</w:body></w:document>")
        if styles is not None:
            z.writestr("word/styles.xml", f"<w:styles {W}>{styles}</w:styles>")
        if numbering is not None:
            z.writestr("word/numbering.xml", f"<w:numbering {W}>{numbering}</w:numbering>")
        for name, blob in (images or {}).values():
            z.writestr(f"word/media/{name}", blob)
    return buf.getvalue()


def p(
    text: str = "",
    *,
    style: str | None = None,
    jc: str | None = None,
    num_id: int | None = None,
    ilvl: int | None = None,
) -> str:
    """One <w:p> with an optional style, alignment and list membership."""
    ppr = ""
    if style is not None:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if num_id is not None or ilvl is not None:
        ppr += "<w:numPr>"
        if ilvl is not None:
            ppr += f'<w:ilvl w:val="{ilvl}"/>'
        if num_id is not None:
            ppr += f'<w:numId w:val="{num_id}"/>'
        ppr += "</w:numPr>"
    if jc is not None:
        ppr += f'<w:jc w:val="{jc}"/>'
    ppr_xml = f"<w:pPr>{ppr}</w:pPr>" if ppr else ""
    run = f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r>' if text else ""
    return f"<w:p>{ppr_xml}{run}</w:p>"


def parse(body: str, **parts: Any) -> PhysicalDocument:
    return DocxAdapter().parse_bytes(build_docx(body, **parts), FILENAME)


def document_root(data: bytes) -> etree._Element:
    """The parsed word/document.xml of a package, to evaluate source references on."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return etree.fromstring(z.read("word/document.xml"))


def paragraph_text(p_el: etree._Element) -> str:
    return "".join(p_el.itertext())


# Styles: Normal (default) <- BodyText (both); Title (center).
STYLES = (
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="BodyText"><w:name w:val="Body Text"/>'
    '<w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="both"/></w:pPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
    '<w:pPr><w:jc w:val="center"/></w:pPr></w:style>'
)

# numId 3 -> abstract 0: level 0 "1.", level 1 "a)".  numId 5 -> abstract 1: bullets.
NUMBERING = (
    '<w:abstractNum w:abstractNumId="0">'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/></w:lvl>'
    '<w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="lowerLetter"/><w:lvlText w:val="%2)"/></w:lvl>'
    "</w:abstractNum>"
    '<w:abstractNum w:abstractNumId="1">'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="-"/></w:lvl>'
    "</w:abstractNum>"
    '<w:num w:numId="3"><w:abstractNumId w:val="0"/></w:num>'
    '<w:num w:numId="5"><w:abstractNumId w:val="1"/></w:num>'
)


# ---------- document-level fields


def test_document_fields() -> None:
    data = build_docx(p("Điều 1. Phạm vi điều chỉnh"))
    doc = DocxAdapter().parse_bytes(data, FILENAME)

    assert doc.id == compute_hash(data)
    assert doc.file_hash == compute_hash(data)
    assert doc.source_filename == FILENAME
    assert doc.source_format is DocSourceFormat.DOCX
    assert doc.adapter_name == "docx"
    assert doc.adapter_version == DocxAdapter.version


def test_same_bytes_give_the_same_document() -> None:
    data = build_docx(p("A") + p("B"))
    assert DocxAdapter().parse_bytes(data, FILENAME) == DocxAdapter().parse_bytes(data, FILENAME)


def test_invalid_bytes_rejected() -> None:
    with pytest.raises(ValueError, match=r"File 'test\.docx': unable to decode"):
        DocxAdapter().parse_bytes(b"not a zip file", FILENAME)


# ---------- paragraphs


def test_plain_paragraphs() -> None:
    doc = parse(p("Chương I") + p("QUY ĐỊNH CHUNG") + p("Điều 1. Phạm vi điều chỉnh"))

    assert [b.text for b in doc.blocks] == ["Chương I", "QUY ĐỊNH CHUNG", "Điều 1. Phạm vi điều chỉnh"]
    assert all(b.block_type is BlockType.PARAGRAPH for b in doc.blocks)
    assert all(b.list_info is None for b in doc.blocks)


def test_empty_document_has_no_blocks() -> None:
    assert parse("").blocks == ()


def test_empty_and_blank_paragraphs_are_skipped() -> None:
    doc = parse(p("A") + p() + p("   ") + p("B"))
    assert [b.text for b in doc.blocks] == ["A", "B"]


def test_ids_are_consecutive_over_emitted_blocks() -> None:
    doc = parse(p() + p("A") + p() + p() + p("B") + p("C"))
    assert [b.id for b in doc.blocks] == ["pb1", "pb2", "pb3"]


def test_source_reference_counts_skipped_paragraphs() -> None:
    # References point at XML positions, so they jump over empty paragraphs.
    doc = parse(p("A") + p() + p("B"))
    assert [b.source_reference for b in doc.blocks] == [
        "/w:document/w:body/w:p[1]",
        "/w:document/w:body/w:p[3]",
    ]


def test_source_reference_of_a_single_paragraph_has_no_index() -> None:
    # lxml only adds "[N]" when there are several siblings with the same name.
    doc = parse(p("Only"))
    assert doc.blocks[0].source_reference == "/w:document/w:body/w:p"


def test_source_reference_finds_the_paragraph_again() -> None:
    data = build_docx(p("A") + p() + p("B") + p("   ") + p("C", num_id=3), numbering=NUMBERING)
    doc = DocxAdapter().parse_bytes(data, FILENAME)
    root = document_root(data)

    for block in doc.blocks:
        found = root.xpath(block.source_reference, namespaces={"w": W_NS})
        assert len(found) == 1, block.source_reference
        assert paragraph_text(found[0]) == block.text


# ---------- lists


def test_nested_list_markers() -> None:
    body = (
        p("Intro")
        + p("First", num_id=3, ilvl=0)
        + p("Sub a", num_id=3, ilvl=1)
        + p("Sub b", num_id=3, ilvl=1)
        + p("Second", num_id=3, ilvl=0)
        + p("Third", num_id=3, ilvl=0)
    )
    doc = parse(body, numbering=NUMBERING)

    items = [(b.text, b.list_info.displayed_marker, b.list_info.nested_level) for b in doc.blocks if b.list_info]
    assert items == [
        ("First", "1.", 0),
        ("Sub a", "a)", 1),
        ("Sub b", "b)", 1),
        ("Second", "2.", 0),
        ("Third", "3.", 0),
    ]
    assert doc.blocks[0].block_type is BlockType.PARAGRAPH
    assert all(b.block_type is BlockType.LIST_ITEM for b in doc.blocks[1:])


def test_list_id_is_the_num_id() -> None:
    doc = parse(p("A", num_id=3), numbering=NUMBERING)
    assert doc.blocks[0].list_info is not None
    assert doc.blocks[0].list_info.id == "3"


def test_missing_ilvl_means_level_zero() -> None:
    doc = parse(p("A", num_id=3), numbering=NUMBERING)
    info = doc.blocks[0].list_info
    assert info is not None
    assert (info.nested_level, info.displayed_marker) == (0, "1.")


def test_bullets_are_unordered() -> None:
    doc = parse(p("Dot", num_id=5), numbering=NUMBERING)
    info = doc.blocks[0].list_info
    assert info is not None
    assert info.is_ordered is False


def test_numbered_items_are_ordered() -> None:
    doc = parse(p("One", num_id=3), numbering=NUMBERING)
    info = doc.blocks[0].list_info
    assert info is not None
    assert info.is_ordered is True


def test_empty_list_item_still_advances_the_counter() -> None:
    # Word shows "1.", "2." (empty) and "3."; the empty one is not emitted.
    doc = parse(p("One", num_id=3) + p(num_id=3) + p("Three", num_id=3), numbering=NUMBERING)
    assert [(b.id, b.list_info.displayed_marker) for b in doc.blocks if b.list_info] == [
        ("pb1", "1."),
        ("pb2", "3."),
    ]


def test_separate_num_ids_count_separately() -> None:
    numbering = NUMBERING + '<w:num w:numId="4"><w:abstractNumId w:val="0"/></w:num>'
    body = p("A1", num_id=3) + p("B1", num_id=4) + p("A2", num_id=3)
    doc = parse(body, numbering=numbering)
    assert [b.list_info.displayed_marker for b in doc.blocks if b.list_info] == ["1.", "1.", "2."]


def test_num_id_zero_is_not_a_list() -> None:
    doc = parse(p("Not a list", num_id=0), numbering=NUMBERING)
    assert doc.blocks[0].block_type is BlockType.PARAGRAPH
    assert doc.blocks[0].list_info is None


def test_no_numbering_is_fine_without_lists() -> None:
    doc = parse(p("A") + p("B"))
    assert len(doc.blocks) == 2


def test_list_paragraph_without_numbering_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=r"File 'test\.docx', at /w:document/w:body/w:p\[2\]: has num_id 3 but the document has no numbering\.xml",
    ):
        parse(p("A") + p("B", num_id=3))


# ---------- paragraph format


def test_paragraph_format_from_styles() -> None:
    body = p("Plain") + p("Body", style="BodyText") + p("Heading", style="Title") + p("Direct", jc="right")
    doc = parse(body, styles=STYLES)

    formats = [(b.paragraph_format.style_name, b.paragraph_format.alignment) for b in doc.blocks]
    assert formats == [
        ("Normal", ParagraphAlignment.LEFT),
        ("Body Text", ParagraphAlignment.JUSTIFIED),
        ("Title", ParagraphAlignment.CENTER),
        ("Normal", ParagraphAlignment.RIGHT),
    ]


def test_paragraph_format_without_styles() -> None:
    doc = parse(p("Plain") + p("Centered", jc="center") + p("Styled", style="Title"))

    formats = [(b.paragraph_format.style_name, b.paragraph_format.alignment) for b in doc.blocks]
    assert formats == [
        (None, ParagraphAlignment.LEFT),
        (None, ParagraphAlignment.CENTER),
        (None, ParagraphAlignment.LEFT),
    ]


def test_list_items_get_a_paragraph_format_too() -> None:
    doc = parse(p("Item", style="BodyText", num_id=3), styles=STYLES, numbering=NUMBERING)
    fmt = doc.blocks[0].paragraph_format
    assert (fmt.style_name, fmt.alignment) == ("Body Text", ParagraphAlignment.JUSTIFIED)


def test_unknown_alignment_names_file_and_paragraph() -> None:
    with pytest.raises(ValueError, match=r"File 'test\.docx', at /w:document/w:body/w:p\[2\]") as excinfo:
        parse(p("A") + p("B", jc="numTab"), styles=STYLES)
    # The original reason is kept as the cause.
    assert "numTab" in str(excinfo.value.__cause__)


def test_unknown_alignment_on_empty_paragraph_is_not_checked() -> None:
    # Empty paragraphs are skipped before their format is resolved.
    doc = parse(p("A") + p(jc="numTab"), styles=STYLES)
    assert len(doc.blocks) == 1


def test_broken_styles_name_the_file() -> None:
    duplicate = '<w:style w:type="paragraph" w:styleId="X"/><w:style w:type="paragraph" w:styleId="X"/>'
    with pytest.raises(ValueError, match=r"File 'test\.docx'"):
        parse(p("A"), styles=duplicate)


# ---------- body elements


def tbl(*rows: list[str]) -> str:
    """A body-level <w:tbl>; each row is a list of cell texts ("" = empty cell)."""
    trs = []
    for cells in rows:
        tcs = "".join(f"<w:tc>{p(text)}</w:tc>" for text in cells)
        trs.append(f"<w:tr>{tcs}</w:tr>")
    return f"<w:tbl><w:tblPr/><w:tblGrid/>{''.join(trs)}</w:tbl>"


def test_table_becomes_one_table_block() -> None:
    doc = parse(p("Before") + tbl(["TT", "Nội dung"], ["1", "Phạm vi"]) + p("After"))

    assert [b.block_type for b in doc.blocks] == [BlockType.PARAGRAPH, BlockType.TABLE, BlockType.PARAGRAPH]
    table = doc.blocks[1]
    assert table.text == "TT\tNội dung\n1\tPhạm vi"
    assert table.table_info is not None
    assert (table.table_info.n_rows, table.table_info.n_cols) == (2, 2)


def test_table_block_ids_and_reference() -> None:
    doc = parse(p("A") + tbl(["x"]) + p("B"))
    assert [b.id for b in doc.blocks] == ["pb1", "pb2", "pb3"]
    assert doc.blocks[1].source_reference == "/w:document/w:body/w:tbl"


def test_table_paragraphs_are_not_body_blocks() -> None:
    doc = parse(tbl(["cell text"]))
    assert len(doc.blocks) == 1
    assert doc.blocks[0].block_type is BlockType.TABLE


def test_table_without_rows_produces_no_block() -> None:
    doc = parse(p("A") + "<w:tbl><w:tblPr/><w:tblGrid/></w:tbl>" + p("B"))
    assert [b.text for b in doc.blocks] == ["A", "B"]
    assert [b.id for b in doc.blocks] == ["pb1", "pb2"]


def test_table_of_empty_cells_produces_no_block() -> None:
    doc = parse(p("A") + tbl(["", ""], ["", ""]))
    assert [b.text for b in doc.blocks] == ["A"]


def test_header_rows_reach_the_block() -> None:
    header_row = f"<w:tr><w:trPr><w:tblHeader/></w:trPr><w:tc>{p('TT')}</w:tc></w:tr>"
    data_row = f"<w:tr><w:tc>{p('1')}</w:tc></w:tr>"
    doc = parse(f"<w:tbl><w:tblPr/><w:tblGrid/>{header_row}{data_row}</w:tbl>")
    info = doc.blocks[0].table_info
    assert info is not None
    assert info.n_header_rows == 1


BOLD_HEADER_TABLE = (
    "<w:tbl><w:tblPr/><w:tblGrid/>"
    "<w:tr><w:tc><w:p><w:r><w:rPr><w:b/></w:rPr><w:t>TT</w:t></w:r></w:p></w:tc></w:tr>"
    f"<w:tr><w:tc>{p('1')}</w:tc></w:tr>"
    "</w:tbl>"
)


def test_bold_first_row_is_the_header_by_default() -> None:
    info = parse(BOLD_HEADER_TABLE).blocks[0].table_info
    assert info is not None
    assert info.n_header_rows == 1


def test_marked_rule_ignores_a_bold_first_row() -> None:
    doc = DocxAdapter(header_row_rule="marked").parse_bytes(build_docx(BOLD_HEADER_TABLE), FILENAME)
    info = doc.blocks[0].table_info
    assert info is not None
    assert info.n_header_rows == 0


def test_default_header_row_rule() -> None:
    assert DocxAdapter().header_row_rule == "marked_or_bold_first"


def test_header_row_rule_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        DocxAdapter("marked")  # type: ignore[call-arg]


def test_merged_table_reaches_the_block() -> None:
    rows = (
        f"<w:tr><w:tc><w:tcPr><w:gridSpan w:val=\"2\"/></w:tcPr>{p('Trường hợp áp dụng')}</w:tc></w:tr>"
        f"<w:tr><w:tc>{p('Báo cáo')}</w:tc><w:tc>{p('Kiểm thử')}</w:tc></w:tr>"
    )
    doc = parse(f"<w:tbl><w:tblPr/><w:tblGrid/>{rows}</w:tbl>")
    assert doc.blocks[0].text == "Trường hợp áp dụng\t\nBáo cáo\tKiểm thử"


def test_cell_text_is_read_like_body_text() -> None:
    # Cell paragraphs go through the same text reading as body paragraphs.
    doc = parse(tbl(["a  b"]))
    assert doc.blocks[0].text == "a  b"


def test_list_inside_cell_shares_the_counter_with_the_body() -> None:
    cell_item = f"<w:tbl><w:tblPr/><w:tblGrid/><w:tr><w:tc>{p('In cell', num_id=3)}</w:tc></w:tr></w:tbl>"
    body = p("First", num_id=3) + cell_item + p("Third", num_id=3)
    doc = parse(body, numbering=NUMBERING)

    first, table, third = doc.blocks
    assert first.list_info is not None and first.list_info.displayed_marker == "1."
    # The cell item takes number 2, and its marker is kept in the cell text.
    assert table.text == "2. In cell"
    assert third.list_info is not None and third.list_info.displayed_marker == "3."


def test_empty_list_item_in_cell_still_counts() -> None:
    cell_item = f"<w:tbl><w:tblPr/><w:tblGrid/><w:tr><w:tc>{p('x')}{p(num_id=3)}</w:tc></w:tr></w:tbl>"
    doc = parse(p("One", num_id=3) + cell_item + p("Three", num_id=3), numbering=NUMBERING)
    assert doc.blocks[-1].list_info is not None
    assert doc.blocks[-1].list_info.displayed_marker == "3."


def test_list_in_cell_without_numbering_rejected() -> None:
    cell_item = f"<w:tbl><w:tblPr/><w:tblGrid/><w:tr><w:tc>{p('x', num_id=3)}</w:tc></w:tr></w:tbl>"
    with pytest.raises(ValueError, match=r"File 'test\.docx', at /w:document/w:body/w:tbl"):
        parse(cell_item)


def test_broken_merge_names_file_and_table() -> None:
    continue_cell = f"<w:tc><w:tcPr><w:vMerge/></w:tcPr>{p()}</w:tc>"
    broken = f"<w:tbl><w:tblPr/><w:tblGrid/><w:tr>{continue_cell}</w:tr></w:tbl>"
    with pytest.raises(ValueError, match=r"File 'test\.docx', at /w:document/w:body/w:tbl") as excinfo:
        parse(p("A") + broken)
    assert "without open cell" in str(excinfo.value.__cause__)


def test_nested_table_not_supported_yet() -> None:
    nested = f"<w:tc>{p('outer')}<w:tbl><w:tr><w:tc>{p('inner')}</w:tc></w:tr></w:tbl>{p()}</w:tc>"
    with pytest.raises(NotImplementedError, match="Nested tables not supported"):
        parse(f"<w:tbl><w:tblPr/><w:tblGrid/><w:tr>{nested}</w:tr></w:tbl>")


def test_table_round_trips_through_json(tmp_path: Path) -> None:
    from document_processor.core.serialization import load_physical_document, save_physical_document

    doc = parse(p("A") + tbl(["TT", "Nội dung"], ["1", ""]))
    path = tmp_path / "doc.json"
    save_physical_document(doc, path)
    assert load_physical_document(path) == doc


def test_unknown_body_element_rejected() -> None:
    # A content control (e.g. a table of contents) directly in the body.
    # lxml's getpath leaves out "[1]" when an element is the only one of its
    # name among its siblings, so the reference here is ".../w:sdt".
    sdt = "<w:sdt><w:sdtContent>" + p("TOC") + "</w:sdtContent></w:sdt>"
    with pytest.raises(
        ValueError,
        match=r"File 'test\.docx', at /w:document/w:body/w:sdt: unknown element tag",
    ):
        parse(p("A") + sdt)


# ---------- parse_file


def test_parse_file_matches_parse_bytes(tmp_path: Path) -> None:
    data = build_docx(p("A") + p("B"))
    path = tmp_path / "nghi-dinh.docx"
    path.write_bytes(data)

    assert DocxAdapter().parse_file(path) == DocxAdapter().parse_bytes(data, "nghi-dinh.docx")


def test_parse_file_accepts_a_str_path(tmp_path: Path) -> None:
    path = tmp_path / "a.docx"
    path.write_bytes(build_docx(p("A")))
    assert DocxAdapter().parse_file(str(path)).source_filename == "a.docx"


def test_parse_file_accepts_upper_case_suffix(tmp_path: Path) -> None:
    path = tmp_path / "A.DOCX"
    path.write_bytes(build_docx(p("A")))
    assert len(DocxAdapter().parse_file(path).blocks) == 1


def test_parse_file_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DocxAdapter().parse_file(tmp_path / "missing.docx")


def test_parse_file_wrong_suffix(tmp_path: Path) -> None:
    path = tmp_path / "a.doc"
    path.write_bytes(build_docx(p("A")))
    with pytest.raises(ValueError, match=r"Wanted \.docx, got \.doc"):
        DocxAdapter().parse_file(path)


# ---------- the real sample file


SAMPLE = Path("data/simple_nested_list.docx")


@pytest.mark.skipif(not SAMPLE.is_file(), reason="sample file not present")
def test_sample_file_invariants() -> None:
    data = SAMPLE.read_bytes()
    doc = DocxAdapter().parse_bytes(data, SAMPLE.name)
    root = document_root(data)

    assert doc.blocks, "expected at least one block"
    assert [b.id for b in doc.blocks] == [f"pb{i}" for i in range(1, len(doc.blocks) + 1)]
    for block in doc.blocks:
        assert block.paragraph_format.alignment is not None
        found = root.xpath(block.source_reference, namespaces={"w": W_NS})
        assert len(found) == 1, block.source_reference
    assert any(b.block_type is BlockType.LIST_ITEM for b in doc.blocks)


# ---------- images
# Drawings come from drawing_xml.py; each image paragraph is built from runs.

PNG_A = b"\x89PNG image A"
PNG_B = b"\x89PNG image B"
IMAGES = {"rId10": ("image1.png", PNG_A), "rId11": ("image2.png", PNG_B)}


def r_text(text: str) -> str:
    return f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r>'


def r_drawing(xml: str) -> str:
    return f"<w:r>{xml}</w:r>"


def p_runs(*runs: str, num_id: int | None = None, jc: str | None = None) -> str:
    """A <w:p> made of the given runs (see r_text / r_drawing)."""
    ppr = ""
    if num_id is not None:
        ppr += f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num_id}"/></w:numPr>'
    if jc is not None:
        ppr += f'<w:jc w:val="{jc}"/>'
    ppr_xml = f"<w:pPr>{ppr}</w:pPr>" if ppr else ""
    return f"<w:p>{ppr_xml}{''.join(runs)}</w:p>"


def img(rid: str = "rId10", **kwargs: Any) -> str:
    """A run holding one picture drawing that embeds `rid`."""
    return r_drawing(drawing(pic(rid), **kwargs))


def test_image_only_paragraph_becomes_an_image_block() -> None:
    doc = parse(p_runs(img()), images=IMAGES)

    assert len(doc.blocks) == 1
    block = doc.blocks[0]
    assert block.block_type is BlockType.IMAGE
    assert block.text == ""  # alt text is not read (empty or a file name in practice)
    assert block.list_info is None
    assert block.image_info is not None
    assert block.image_info.source_path == "word/media/image1.png"
    assert block.image_info.mime_type == "image/png"
    assert block.image_info.content_hash == compute_hash(PNG_A)
    assert block.image_info.displayed_width_pt == 72.0
    assert block.image_info.displayed_height_pt == 36.0
    assert block.source_reference == "/w:document/w:body/w:p"


def test_image_block_gets_the_paragraph_format() -> None:
    doc = parse(p_runs(img(), jc="center"), images=IMAGES, styles=STYLES)
    assert doc.blocks[0].paragraph_format.alignment is ParagraphAlignment.CENTER


def test_heading_then_figure_in_separate_paragraphs() -> None:
    # The real case: "1. Mô hình" followed by a diagram in its own paragraph.
    doc = parse(p("1. Mô hình") + p_runs(img()) + p("Trong đó:"), images=IMAGES)

    assert [b.block_type for b in doc.blocks] == [
        BlockType.PARAGRAPH,
        BlockType.IMAGE,
        BlockType.PARAGRAPH,
    ]
    assert [b.id for b in doc.blocks] == ["pb1", "pb2", "pb3"]


def test_text_image_text_in_one_paragraph_keeps_reading_order() -> None:
    doc = parse(
        p_runs(r_text("1. Mô hình"), img(), r_text("Hình 1: Tổng thể")),
        images=IMAGES,
    )

    assert [(b.block_type, b.text) for b in doc.blocks] == [
        (BlockType.PARAGRAPH, "1. Mô hình"),
        (BlockType.IMAGE, ""),
        (BlockType.PARAGRAPH, "Hình 1: Tổng thể"),
    ]
    assert [b.id for b in doc.blocks] == ["pb1", "pb2", "pb3"]


def test_blocks_from_one_paragraph_get_indexed_references() -> None:
    doc = parse(p_runs(r_text("A"), img(), r_text("B")), images=IMAGES)
    base = "/w:document/w:body/w:p"
    assert [b.source_reference for b in doc.blocks] == [base, f"{base}#1", f"{base}#2"]


def test_image_first_then_text() -> None:
    doc = parse(p_runs(img(), r_text("Hình 1")), images=IMAGES)
    assert [b.block_type for b in doc.blocks] == [BlockType.IMAGE, BlockType.PARAGRAPH]


def test_drawing_in_the_same_run_as_text_splits_it() -> None:
    run = f'<w:r><w:t>before</w:t>{drawing()}<w:t>after</w:t></w:r>'
    doc = parse(p_runs(run), images=IMAGES)
    assert [(b.block_type, b.text) for b in doc.blocks] == [
        (BlockType.PARAGRAPH, "before"),
        (BlockType.IMAGE, ""),
        (BlockType.PARAGRAPH, "after"),
    ]


def test_blank_text_around_an_image_is_dropped() -> None:
    doc = parse(p_runs(r_text("  "), img(), r_text("\t")), images=IMAGES)
    assert [b.block_type for b in doc.blocks] == [BlockType.IMAGE]
    assert doc.blocks[0].source_reference == "/w:document/w:body/w:p"


def test_two_images_in_one_paragraph() -> None:
    doc = parse(p_runs(img("rId10"), img("rId11")), images=IMAGES)

    assert [b.block_type for b in doc.blocks] == [BlockType.IMAGE, BlockType.IMAGE]
    hashes = [b.image_info.content_hash for b in doc.blocks if b.image_info is not None]
    assert hashes == [compute_hash(PNG_A), compute_hash(PNG_B)]


def test_same_image_used_twice_has_one_hash() -> None:
    doc = parse(p_runs(img("rId10")) + p_runs(img("rId10")), images=IMAGES)
    assert len(doc.blocks) == 2
    assert doc.blocks[0].image_info == doc.blocks[1].image_info


def test_anchored_image_is_read_too() -> None:
    doc = parse(p_runs(img(anchor=True)), images=IMAGES)
    assert [b.block_type for b in doc.blocks] == [BlockType.IMAGE]


def test_image_inside_a_hyperlink_is_read() -> None:
    link = f'<w:hyperlink r:id="rId99" xmlns:r="{R_NS}">{img()}</w:hyperlink>'
    doc = parse(p_runs(r_text("See"), link), images=IMAGES)
    assert [b.block_type for b in doc.blocks] == [BlockType.PARAGRAPH, BlockType.IMAGE]


def test_chart_is_not_an_image_block() -> None:
    doc = parse(p_runs(r_text("Biểu đồ"), r_drawing(drawing(chart()))), images=IMAGES)
    assert [(b.block_type, b.text) for b in doc.blocks] == [(BlockType.PARAGRAPH, "Biểu đồ")]


def test_paragraph_with_only_a_chart_produces_no_block() -> None:
    doc = parse(p("A") + p_runs(r_drawing(drawing(chart()))) + p("B"), images=IMAGES)
    assert [b.text for b in doc.blocks] == ["A", "B"]
    assert [b.id for b in doc.blocks] == ["pb1", "pb2"]


def test_linked_image_is_skipped() -> None:
    linked = r_drawing(drawing(pic(embed=None, link="rId30")))
    doc = parse(p("A") + p_runs(linked), images=IMAGES)
    assert [b.text for b in doc.blocks] == ["A"]


def test_unknown_image_relationship_names_file_and_paragraph() -> None:
    with pytest.raises(
        ValueError, match=r"File 'test\.docx', at /w:document/w:body/w:p\[2\]: .*'rId77'"
    ):
        parse(p("A") + p_runs(img("rId77")), images=IMAGES)


# ---------- images and lists


def test_numbered_paragraph_with_image_puts_the_marker_on_the_text() -> None:
    doc = parse(
        p_runs(r_text("Mô hình"), img(), num_id=3) + p("Tiếp theo", num_id=3),
        numbering=NUMBERING,
        images=IMAGES,
    )

    text_block, image_block, next_item = doc.blocks
    assert text_block.block_type is BlockType.LIST_ITEM
    assert text_block.list_info is not None
    assert text_block.list_info.displayed_marker == "1."
    assert image_block.block_type is BlockType.IMAGE
    assert image_block.list_info is None
    # The paragraph counted once, not once per block.
    assert next_item.list_info is not None
    assert next_item.list_info.displayed_marker == "2."


def test_text_after_the_image_is_a_plain_paragraph() -> None:
    doc = parse(
        p_runs(r_text("Mô hình"), img(), r_text("Hình 1"), num_id=3),
        numbering=NUMBERING,
        images=IMAGES,
    )
    assert [b.block_type for b in doc.blocks] == [
        BlockType.LIST_ITEM,
        BlockType.IMAGE,
        BlockType.PARAGRAPH,
    ]
    assert doc.blocks[2].list_info is None


def test_numbered_image_only_paragraph_still_advances_the_counter() -> None:
    doc = parse(
        p("A", num_id=3) + p_runs(img(), num_id=3) + p("C", num_id=3),
        numbering=NUMBERING,
        images=IMAGES,
    )
    markers = [b.list_info.displayed_marker for b in doc.blocks if b.list_info is not None]
    assert markers == ["1.", "3."]  # the image paragraph was "2."
    assert doc.blocks[1].block_type is BlockType.IMAGE


def test_empty_numbered_paragraph_with_broken_format_still_counts() -> None:
    # Numbering is read before the format, and an empty paragraph never
    # reaches the format: it counts, and its broken jc is not an error.
    doc = parse(
        p("A", num_id=3) + p(num_id=3, jc="numTab") + p("C", num_id=3),
        numbering=NUMBERING,
        styles=STYLES,
    )
    markers = [b.list_info.displayed_marker for b in doc.blocks if b.list_info is not None]
    assert markers == ["1.", "3."]


# ---------- images and tables


def test_image_in_a_table_cell_is_ignored() -> None:
    cell = f"<w:tc>{p_runs(r_text('Sơ đồ'), img())}</w:tc>"
    doc = parse(f"<w:tbl><w:tblPr/><w:tblGrid/><w:tr>{cell}</w:tr></w:tbl>", images=IMAGES)

    assert [b.block_type for b in doc.blocks] == [BlockType.TABLE]
    assert "Sơ đồ" in doc.blocks[0].text


def test_ids_stay_consecutive_around_tables_and_images() -> None:
    doc = parse(p("A") + tbl(["x"]) + p_runs(img()) + tbl(["y"]) + p("B"), images=IMAGES)
    assert [b.id for b in doc.blocks] == ["pb1", "pb2", "pb3", "pb4", "pb5"]
    assert [b.block_type for b in doc.blocks] == [
        BlockType.PARAGRAPH,
        BlockType.TABLE,
        BlockType.IMAGE,
        BlockType.TABLE,
        BlockType.PARAGRAPH,
    ]


# ---------- symbol-font bullets


def test_wingdings_check_mark_bullet_reaches_the_block() -> None:
    # As in the real document: abstractNum 22, lvlText U+F0FC in Wingdings.
    wingdings = (
        '<w:abstractNum w:abstractNumId="22"><w:lvl w:ilvl="0"><w:start w:val="1"/>'
        '<w:numFmt w:val="bullet"/><w:lvlText w:val="\uf0fc"/><w:lvlJc w:val="start"/>'
        '<w:rPr><w:rFonts w:ascii="Wingdings" w:hAnsi="Wingdings" w:cs="Wingdings" w:hint="default"/></w:rPr>'
        '</w:lvl></w:abstractNum><w:num w:numId="22"><w:abstractNumId w:val="22"/></w:num>'
    )
    doc = parse(p("Đã hoàn thành", num_id=22), numbering=wingdings)
    info = doc.blocks[0].list_info
    assert info is not None
    assert info.displayed_marker == "✓"
    assert info.is_ordered is False