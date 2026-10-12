"""End-to-end tests for the docx adapter.

Each test builds a small .docx in memory from raw XML parts, so every case
(no styles, no numbering, tables, broken values) is under the test's control
and no binary fixtures are needed. A last test runs on the sample file in
data/ when it is present.
"""

import io
import zipfile
from pathlib import Path

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

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f'xmlns:w="{W_NS}"'
FILENAME = "test.docx"

# ---------- building a .docx in memory

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""

PACKAGE_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

REL_STYLES = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
)
REL_NUMBERING = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering"
)

SECT_PR = '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr>'


def build_docx(
    body: str,
    *,
    styles: str | None = None,
    numbering: str | None = None,
) -> bytes:
    """A minimal .docx. `body` is the inside of <w:body>; a sectPr is appended.

    `styles` / `numbering` are the inside of <w:styles> / <w:numbering>;
    None leaves the part (and its relationship) out of the package.
    """
    rels = []
    if styles is not None:
        rels.append(
            f'<Relationship Id="rId1" Type="{REL_STYLES}" Target="styles.xml"/>'
        )
    if numbering is not None:
        rels.append(
            f'<Relationship Id="rId2" Type="{REL_NUMBERING}" Target="numbering.xml"/>'
        )
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
        z.writestr(
            "word/document.xml",
            f"<w:document {W}><w:body>{body}{SECT_PR}</w:body></w:document>",
        )
        if styles is not None:
            z.writestr("word/styles.xml", f"<w:styles {W}>{styles}</w:styles>")
        if numbering is not None:
            z.writestr(
                "word/numbering.xml", f"<w:numbering {W}>{numbering}</w:numbering>"
            )
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


def parse(body: str, **parts: str | None) -> PhysicalDocument:
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
    assert DocxAdapter().parse_bytes(data, FILENAME) == DocxAdapter().parse_bytes(
        data, FILENAME
    )


def test_invalid_bytes_rejected() -> None:
    with pytest.raises(ValueError, match=r"File 'test\.docx': unable to decode"):
        DocxAdapter().parse_bytes(b"not a zip file", FILENAME)


# ---------- paragraphs


def test_plain_paragraphs() -> None:
    doc = parse(p("Chương I") + p("QUY ĐỊNH CHUNG") + p("Điều 1. Phạm vi điều chỉnh"))

    assert [b.text for b in doc.blocks] == [
        "Chương I",
        "QUY ĐỊNH CHUNG",
        "Điều 1. Phạm vi điều chỉnh",
    ]
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
    data = build_docx(
        p("A") + p() + p("B") + p("   ") + p("C", num_id=3), numbering=NUMBERING
    )
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

    items = [
        (b.text, b.list_info.displayed_marker, b.list_info.nested_level)
        for b in doc.blocks
        if b.list_info
    ]
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
    doc = parse(
        p("One", num_id=3) + p(num_id=3) + p("Three", num_id=3), numbering=NUMBERING
    )
    assert [
        (b.id, b.list_info.displayed_marker) for b in doc.blocks if b.list_info
    ] == [
        ("pb1", "1."),
        ("pb2", "3."),
    ]


def test_separate_num_ids_count_separately() -> None:
    numbering = NUMBERING + '<w:num w:numId="4"><w:abstractNumId w:val="0"/></w:num>'
    body = p("A1", num_id=3) + p("B1", num_id=4) + p("A2", num_id=3)
    doc = parse(body, numbering=numbering)
    assert [b.list_info.displayed_marker for b in doc.blocks if b.list_info] == [
        "1.",
        "1.",
        "2.",
    ]


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
    body = (
        p("Plain")
        + p("Body", style="BodyText")
        + p("Heading", style="Title")
        + p("Direct", jc="right")
    )
    doc = parse(body, styles=STYLES)

    formats = [
        (b.paragraph_format.style_name, b.paragraph_format.alignment)
        for b in doc.blocks
    ]
    assert formats == [
        ("Normal", ParagraphAlignment.LEFT),
        ("Body Text", ParagraphAlignment.JUSTIFIED),
        ("Title", ParagraphAlignment.CENTER),
        ("Normal", ParagraphAlignment.RIGHT),
    ]


def test_paragraph_format_without_styles() -> None:
    doc = parse(p("Plain") + p("Centered", jc="center") + p("Styled", style="Title"))

    formats = [
        (b.paragraph_format.style_name, b.paragraph_format.alignment)
        for b in doc.blocks
    ]
    assert formats == [
        (None, ParagraphAlignment.LEFT),
        (None, ParagraphAlignment.CENTER),
        (None, ParagraphAlignment.LEFT),
    ]


def test_list_items_get_a_paragraph_format_too() -> None:
    doc = parse(
        p("Item", style="BodyText", num_id=3), styles=STYLES, numbering=NUMBERING
    )
    fmt = doc.blocks[0].paragraph_format
    assert (fmt.style_name, fmt.alignment) == (
        "Body Text",
        ParagraphAlignment.JUSTIFIED,
    )


def test_unknown_alignment_names_file_and_paragraph() -> None:
    with pytest.raises(
        ValueError, match=r"File 'test\.docx', at /w:document/w:body/w:p\[2\]"
    ) as excinfo:
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

    assert [b.block_type for b in doc.blocks] == [
        BlockType.PARAGRAPH,
        BlockType.TABLE,
        BlockType.PARAGRAPH,
    ]
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


def test_merged_table_reaches_the_block() -> None:
    rows = (
        f'<w:tr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr>{p("Trường hợp áp dụng")}</w:tc></w:tr>'
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
    doc = parse(
        p("One", num_id=3) + cell_item + p("Three", num_id=3), numbering=NUMBERING
    )
    assert doc.blocks[-1].list_info is not None
    assert doc.blocks[-1].list_info.displayed_marker == "3."


def test_list_in_cell_without_numbering_rejected() -> None:
    cell_item = f"<w:tbl><w:tblPr/><w:tblGrid/><w:tr><w:tc>{p('x', num_id=3)}</w:tc></w:tr></w:tbl>"
    with pytest.raises(
        ValueError, match=r"File 'test\.docx', at /w:document/w:body/w:tbl"
    ):
        parse(cell_item)


def test_broken_merge_names_file_and_table() -> None:
    continue_cell = f"<w:tc><w:tcPr><w:vMerge/></w:tcPr>{p()}</w:tc>"
    broken = f"<w:tbl><w:tblPr/><w:tblGrid/><w:tr>{continue_cell}</w:tr></w:tbl>"
    with pytest.raises(
        ValueError, match=r"File 'test\.docx', at /w:document/w:body/w:tbl"
    ) as excinfo:
        parse(p("A") + broken)
    assert "without open cell" in str(excinfo.value.__cause__)


def test_nested_table_not_supported_yet() -> None:
    nested = f"<w:tc>{p('outer')}<w:tbl><w:tr><w:tc>{p('inner')}</w:tc></w:tr></w:tbl>{p()}</w:tc>"
    with pytest.raises(NotImplementedError, match="Nested tables not supported"):
        parse(f"<w:tbl><w:tblPr/><w:tblGrid/><w:tr>{nested}</w:tr></w:tbl>")


def test_table_round_trips_through_json(tmp_path: Path) -> None:
    from document_processor.core.serialization import (
        load_physical_document,
        save_physical_document,
    )

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

    assert DocxAdapter().parse_file(path) == DocxAdapter().parse_bytes(
        data, "nghi-dinh.docx"
    )


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
    assert [b.id for b in doc.blocks] == [
        f"pb{i}" for i in range(1, len(doc.blocks) + 1)
    ]
    for block in doc.blocks:
        assert block.paragraph_format.alignment is not None
        found = root.xpath(block.source_reference, namespaces={"w": W_NS})
        assert len(found) == 1, block.source_reference
    assert any(b.block_type is BlockType.LIST_ITEM for b in doc.blocks)
