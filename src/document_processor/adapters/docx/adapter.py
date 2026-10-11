from pathlib import Path
from docx import Document as DocumentFactory
from docx.document import Document
from io import BytesIO
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from lxml.etree import _Element
from lxml import etree
from document_processor.core.models.physical import (
    PhysicalDocument,
    PhysicalBlock,
    BlockType,
    Region,
    DocSourceFormat,
    ListInfo,
)
from document_processor.core.text_utils import is_invisible
from document_processor.core.identity import compute_hash

from .numbering import Numberer, read_num_id, read_ilvl
from .styles import Stylist


class _DocxReader:
    def __init__(self, doc: Document, filename: str) -> None:
        self._doc = doc
        self._filename = filename
        self._block_count: int = 0
        self._tree = etree.ElementTree(doc.element)

        try:
            part = doc.part.part_related_by(RT.NUMBERING)
            numbering_root: _Element | None = part.element
        except KeyError:
            numbering_root = None
        except ValueError as e:
            raise ValueError(
                f"File {filename!r}: multiple conflicting numbering relationship found"
            ) from e

        self._numberer = (
            Numberer(numbering_root) if numbering_root is not None else None
        )

        try:
            part = doc.part.part_related_by(RT.STYLES)
            styles_root: _Element | None = part.element
        except KeyError:
            styles_root = None
        except ValueError as e:
            raise ValueError(
                f"File {filename!r}: multiple conflicting styles relationship found"
            ) from e

        try:
            self._stylist = Stylist(styles_root)
        except ValueError as e:
            raise ValueError(f"File {filename!r}: unable to create Stylist object") from e
        

    def read_blocks(self) -> tuple[PhysicalBlock, ...]:
        blocks: list[PhysicalBlock] = []

        for el in self._doc.element.body:
            ref = self._tree.getpath(el)
            if el.tag == qn("w:p"):
                # Paragraph
                block = self._process_para_block(el, ref)
                if block is None:
                    continue
                blocks.append(block)

            elif el.tag == qn("w:tbl"):
                # Table
                raise NotImplementedError("tables are not supported yet")

            elif el.tag == qn("w:sectPr"):
                # Skip sectPr intentionally
                continue

            else:
                raise ValueError(
                    f"File {self._filename!r}, at {ref}: unknown element tag: {el.tag}"
                )

        return tuple(blocks)

    def _process_para_block(self, el: _Element, ref: str) -> PhysicalBlock | None:

        p = Paragraph(el, self._doc)

        num_id = read_num_id(el)
        ilvl = read_ilvl(el)
        if ilvl is None:
            ilvl = 0

        if num_id is not None and num_id != 0:
            block_type = BlockType.LIST_ITEM
            if self._numberer is None:
                raise ValueError(
                    f"File {self._filename!r}, at {ref}: has num_id {num_id!r} but the document has no numbering.xml"
                )

            list_info: ListInfo | None = self._numberer.next_list_info(num_id, ilvl)

        else:
            block_type = BlockType.PARAGRAPH
            list_info = None

        if is_invisible(p.text):
            return None

        self._block_count += 1

        try:
            paragraph_format = self._stylist.resolve_paragraph_format(el)
        except ValueError as e:
            raise ValueError(f"File {self._filename!r}, at {ref}: unable to resolve paragraph format") from e

        char_spans = ()

        return PhysicalBlock(
            id=f"pb{self._block_count}",
            block_type=block_type,
            text=p.text,
            region=Region.BODY,
            page=None,
            paragraph_format=paragraph_format,
            char_spans=char_spans,
            list_info=list_info,
            source_reference=ref,
            changelog=(),
        )


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


# if __name__ == "__main__":
#     adapter = DocxAdapter()
#     # file = "data/no_list.docx"
#     # file = "data/simple_nested_list.docx"
#     file = "data/headings.docx"
#     phys_doc = adapter.parse_file(file)
#     from document_processor.core.serialization import save_physical_document

#     save_physical_document(phys_doc, f"{file}.json")
