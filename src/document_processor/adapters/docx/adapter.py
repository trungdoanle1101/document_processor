from pathlib import Path
from docx import Document as DocumentFactory
from docx.document import Document
from io import BytesIO
from docx.text.paragraph import Paragraph
from docx.oxml.text.paragraph import CT_P
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.part import XmlPart
from lxml.etree import _Element
from lxml import etree
from document_processor.core.models.physical import (
    PhysicalDocument,
    PhysicalBlock,
    BlockType,
    Region,
    DocSourceFormat,
    ListInfo,
    ParagraphFormat,
)
from document_processor.core.text_utils import is_invisible
from document_processor.core.identity import compute_hash

from .numbering import Numberer, read_num_id, read_ilvl
from .styles import Stylist
from .tables import parse_table

RELATION_TYPE_MAPPER = {"styles": RT.STYLES, "numbering": RT.NUMBERING}


class _DocxReader:
    def __init__(self, doc: Document, filename: str) -> None:
        self._doc = doc
        self._filename = filename
        self._block_count: int = 0
        self._tree = etree.ElementTree(doc.element)

        numbering_root = self._get_root("numbering")

        self._numberer = (
            Numberer(numbering_root) if numbering_root is not None else None
        )

        styles_root = self._get_root("styles")

        try:
            self._stylist = Stylist(styles_root)
        except ValueError as e:
            raise ValueError(
                f"File {filename!r}: unable to create Stylist object"
            ) from e

    def _get_root(self, root_type: str) -> _Element | None:
        if root_type not in RELATION_TYPE_MAPPER:
            raise ValueError(
                f"File {self._filename!r}: invalid root_type. Expected in {sorted(RELATION_TYPE_MAPPER)}"
            )

        try:
            part = self._doc.part.part_related_by(RELATION_TYPE_MAPPER[root_type])
        except KeyError:
            return None
        except ValueError as e:
            raise ValueError(
                f"File {self._filename!r}: multiple conflicting {root_type} relationship found"
            ) from e

        if not isinstance(part, XmlPart):
            raise ValueError(f"File {self._filename!r}: {root_type} part not in XML")

        root: _Element = part.element
        return root

    def read_blocks(self) -> tuple[PhysicalBlock, ...]:
        blocks: list[PhysicalBlock] = []

        for el in self._doc.element.body:
            if el.tag == qn("w:p"):
                # Paragraph
                block = self._create_para_block(el)
                if block is None:
                    continue
                blocks.append(block)

            elif el.tag == qn("w:tbl"):
                # Table
                block = self._create_table_block(el)
                if block is None:
                    continue
                blocks.append(block)
            elif el.tag == qn("w:sectPr"):
                # Skip sectPr intentionally
                continue

            else:
                raise ValueError(
                    f"File {self._filename!r}, at {self._ref(el)}: unknown element tag: {el.tag!r}"
                )

        return tuple(blocks)

    def _read_paragraph(self, p: _Element) -> tuple[str, ListInfo | None] | None:
        if not isinstance(p, CT_P):
            raise ValueError(f"File {self._filename!r}, at {self._ref(p)}: not a CT_P")
        paragraph = Paragraph(p, self._doc)

        num_id = read_num_id(p)
        ilvl = read_ilvl(p)
        if ilvl is None:
            ilvl = 0

        if num_id is not None and num_id != 0:
            if self._numberer is None:
                raise ValueError(
                    f"File {self._filename!r}, at {self._ref(p)}: has num_id {num_id!r} but the document has no numbering.xml"
                )
            try:
                list_info: ListInfo | None = self._numberer.next_list_info(num_id, ilvl)
            except ValueError as e:
                raise ValueError(
                    f"File {self._filename!r}, at {self._ref(p)}, num_id {num_id}, ilvl {ilvl}: error occurred: {e}"
                )  from e

        else:
            list_info = None

        if is_invisible(paragraph.text):
            return None

        return (paragraph.text, list_info)

    def _create_para_block(self, p: _Element) -> PhysicalBlock | None:

        result = self._read_paragraph(p)
        if result is None:
            return None

        text, list_info = result

        self._block_count += 1

        try:
            paragraph_format = self._stylist.resolve_paragraph_format(p)
        except ValueError as e:
            raise ValueError(
                f"File {self._filename!r}, at {self._ref(p)}: unable to resolve paragraph format"
            ) from e

        char_spans = ()
        block_type = (
            BlockType.LIST_ITEM if list_info is not None else BlockType.PARAGRAPH
        )

        return PhysicalBlock(
            id=f"pb{self._block_count}",
            block_type=block_type,
            text=text,
            region=Region.BODY,
            page=None,
            paragraph_format=paragraph_format,
            char_spans=char_spans,
            list_info=list_info,
            source_reference=self._ref(p),
            changelog=(),
        )

    def _create_table_block(self, tbl: _Element) -> PhysicalBlock | None:
        try:
            table_info = parse_table(tbl, self._read_cell_paragraph)
        except ValueError as e:
            raise ValueError(
                f"File {self._filename!r}, at {self._ref(tbl)}: error occurred when parsing table: {e}"
            ) from e

        if table_info is None:
            return None
        table_text = table_info.text()
        if is_invisible(table_text):
            return None

        self._block_count += 1

        return PhysicalBlock(
            id=f"pb{self._block_count}",
            block_type=BlockType.TABLE,
            text=table_text,
            region=Region.BODY,
            page=None,
            paragraph_format=ParagraphFormat(alignment=None, style_name=None),
            char_spans=(),
            table_info=table_info,
            source_reference=self._ref(tbl),
            changelog=(),
        )

    def _read_cell_paragraph(self, p: _Element) -> str | None:
        result = self._read_paragraph(p)
        if result is None:
            return None
        text, list_info = result
        # Prefix marker for paragraph
        # Do this because list_info is lost for list items within table cells
        if list_info is not None and list_info.displayed_marker is not None:
            return f"{list_info.displayed_marker} {text}"
        return text

    def _ref(self, el: _Element) -> str:
        return self._tree.getpath(el)


class DocxAdapter:
    name = "docx"
    version = "1.0.0"

    def parse_bytes(self, data: bytes, filename: str) -> PhysicalDocument:

        hash_str = compute_hash(data)

        try:
            doc = DocumentFactory(BytesIO(data))
        except Exception as e:
            raise ValueError(
                f"File {filename!r}: unable to decode bytes data as docx.Document"
            ) from e

        # Create a reader once per parse
        # Call read_blocks once
        reader = _DocxReader(doc, filename)
        blocks = reader.read_blocks()

        return PhysicalDocument(
            id=hash_str,
            source_filename=filename,
            source_format=DocSourceFormat.DOCX,
            adapter_name=self.name,
            adapter_version=self.version,
            file_hash=hash_str,
            blocks=blocks,
        )

    def parse_file(self, file_path: Path | str) -> PhysicalDocument:
        file_path = Path(file_path)

        # First, check if the format is valid. If not valid, raise and exit
        if not file_path.is_file():
            raise FileNotFoundError(f"{file_path} is not a valid file")
        if file_path.suffix.lower() != ".docx":
            raise ValueError(
                f"Unexpected file type: Wanted .docx, got {file_path.suffix}"
            )

        filename = file_path.name
        file_bytes = file_path.read_bytes()

        return self.parse_bytes(file_bytes, filename)


if __name__ == "__main__":
    # file = "data/no_list.docx"
    # file = "data/simple_nested_list.docx"
    file = "data/2025_Quy dinh quan ly kien truc CNTT.docx"
    adapter = DocxAdapter()
    phys_doc = adapter.parse_file(file)
    from document_processor.core.serialization import save_physical_document

    save_physical_document(phys_doc, f"{file}.json")

    # reader = _DocxReader(DocumentFactory("data/simple_table.docx"), "simple_table.docx")
    # blocks = reader.read_blocks()
