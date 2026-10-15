from docx.document import Document
from docx.text.paragraph import Paragraph, Hyperlink
from docx.text.run import Run
from docx.text.hyperlink import Hyperlink
from docx.drawing import Drawing
from docx.oxml.text.paragraph import CT_P
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.part import XmlPart
from lxml.etree import _Element
from lxml import etree
from document_processor.core.models.physical import (
    PhysicalBlock,
    BlockType,
    Region,
    ListInfo,
    ParagraphFormat,
    ImageInfo,
)
from document_processor.core.text_utils import is_invisible
from typing import Any, Iterator

from .numbering import Numberer, read_num_id, read_ilvl
from .styles import Stylist
from .tables import parse_table
from .images import read_drawing_image


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


    def read_blocks(self) -> tuple[PhysicalBlock, ...]:
        blocks: list[PhysicalBlock] = []

        for el in self._doc.element.body:
            if el.tag == qn("w:p"):
                # Paragraph
                pbs = self._paragraph_blocks(el)
                blocks.extend(pbs)

            elif el.tag == qn("w:tbl"):
                # Table
                tbs = self._table_blocks(el)
                blocks.extend(tbs)
            elif el.tag == qn("w:sectPr"):
                # Skip sectPr intentionally
                continue

            else:
                raise ValueError(
                    f"File {self._filename!r}, at {self._ref(el)}: unknown element tag: {el.tag!r}"
                )

        return tuple(blocks)
    
    
    def _paragraph_blocks(self, p: _Element) -> list[PhysicalBlock]:
        paragraph = self._paragraph(p)
        list_info = self._list_info(p)


        segments = [
            s for s in paragraph_segments(paragraph)
            if not (isinstance(s, str) and is_invisible(s))
        ]

        if not segments:
            return []


        paragraph_format = self._paragraph_format(p)

        blocks: list[PhysicalBlock] = []
        pending_list_info: ListInfo | None = list_info
        for segment in segments:
            ref = self._ref(p) if not blocks else f"{self._ref(p)}#{len(blocks)}"

            if isinstance(segment, str):
                if is_invisible(segment):
                    continue
                block = self._text_block(segment, pending_list_info, paragraph_format, ref)
                blocks.append(block)
                pending_list_info = None
            elif isinstance(segment, Drawing):
                image_info = self._image_info(segment, p)
                if image_info is None:
                    continue
                block = self._image_block(image_info, paragraph_format, ref)
                blocks.append(block)

            # Skip everything else intentionally

        return blocks

    def _text_block(self, text: str, list_info: ListInfo | None, paragraph_format: ParagraphFormat, ref: str) -> PhysicalBlock :
        block_type = (
            BlockType.LIST_ITEM if list_info is not None else BlockType.PARAGRAPH
        )

        return self._new_block(
            block_type=block_type,
            text=text,
            paragraph_format=paragraph_format,
            list_info=list_info,
            source_reference=ref
        )

    def _paragraph(self, p: _Element) -> Paragraph:
        if not isinstance(p, CT_P):
            raise ValueError(f"File {self._filename!r}, at {self._ref(p)}: not a CT_P")
        return Paragraph(p, self._doc)
    
    
    def _list_info(self, p:_Element) -> ListInfo | None:
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

        return list_info
    

    def _paragraph_format(self, p: _Element) -> ParagraphFormat:
        try:
            paragraph_format = self._stylist.resolve_paragraph_format(p)
        except ValueError as e:
            raise ValueError(
                f"File {self._filename!r}, at {self._ref(p)}: unable to resolve paragraph format: {e}"
            ) from e
        
        return paragraph_format


    def _table_blocks(self, tbl: _Element) -> list[PhysicalBlock]:
        blocks: list[PhysicalBlock] = []
        try:
            table_info = parse_table(tbl, self._read_cell_paragraph)
        except ValueError as e:
            raise ValueError(
                f"File {self._filename!r}, at {self._ref(tbl)}: error occurred when parsing table: {e}"
            ) from e

        if table_info is None:
            return []
        
        table_text = table_info.text()
        if is_invisible(table_text):
            return []


        block = self._new_block(
            block_type=BlockType.TABLE,
            text=table_text,
            paragraph_format=ParagraphFormat(alignment=None, style_name=None),
            table_info=table_info,
            source_reference=self._ref(tbl)
        )
        blocks.append(block)
    
        return blocks
    

    def _read_cell_paragraph(self, p: _Element) -> str | None:
        paragraph = self._paragraph(p)
        list_info = self._list_info(p)
        text = paragraph.text

        if is_invisible(text):
            return None
        
        # Prefix marker for paragraph
        # Do this because list_info is lost for list items within table cells
        if list_info is not None and list_info.displayed_marker is not None:
            return f"{list_info.displayed_marker} {text}"
        return text
    

    def _image_block(self, image_info: ImageInfo, paragraph_format: ParagraphFormat, ref: str) -> PhysicalBlock:
        return self._new_block(
            block_type=BlockType.IMAGE,
            text="", # Not giving text for IMAGE blocks
            paragraph_format=paragraph_format,
            image_info=image_info,
            source_reference=ref,
        )
    
    def _image_info(self, segment: Drawing, p: _Element) -> ImageInfo | None:
        if not segment.has_picture:
            return None
        
        related_parts = self._doc.part.related_parts
        try: 
            image_info = read_drawing_image(segment._element, related_parts)
        except ValueError as e:
            raise ValueError(f"File {self._filename!r}, at {self._ref(p)}: error occurred: {e}") from e


        return image_info
    

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
    

    def _ref(self, el: _Element) -> str:
        return self._tree.getpath(el)
    

    def _new_block(self, **fields: Any) -> PhysicalBlock:
        self._block_count += 1
        return PhysicalBlock(
            id=f"pb{self._block_count}",
            region=Region.BODY,
            page=None,
            char_spans=(),
            changelog=(),
            **fields
        )


def paragraph_segments(paragraph: Paragraph) -> list[str | Drawing]:
    segments: list[str | _Element] = []
    buffer: list[str] = []

    def runs() -> Iterator[Run]:
        for item in paragraph.iter_inner_content():
            if isinstance(item, Hyperlink):
                yield from item.runs
            else:
                yield item


    for run in runs():
        for piece in run.iter_inner_content():
            if isinstance(piece, str):
                buffer.append(piece)
            elif isinstance(piece, Drawing):
                if buffer:
                    segments.append("".join(buffer))
                    buffer = []
                segments.append(piece)
            # Ignore RenderedPageBreak
        
    if buffer:
        segments.append("".join(buffer))
    return segments