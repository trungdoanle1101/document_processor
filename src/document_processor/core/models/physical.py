from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum
from .ranges import TextRange
from document_processor.core.text_utils import is_invisible
from document_processor.core.identity import HASH_ALGORITHM



class DocSourceFormat(Enum):
    DOCX = "docx"
    PDF = "pdf"
    MARKDOWN = "md"
    TXT = "txt"


class BlockType(Enum):
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    IMAGE = "image"
    TABLE = "table"


class Region(Enum):
    BODY = "body"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"


class ParagraphAlignment(Enum):
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    JUSTIFIED = "justified"


class SpanStyle(Enum):
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"
    STRIKETHROUGH = "strikethrough"


@dataclass(frozen=True)
class ListInfo:
    id: str
    is_ordered: bool
    displayed_marker: str | None
    nested_level: int  # Starting at 0

IMAGE_MIME_TYPE_PATTERN = re.compile(r"image/[a-z0-9][a-z0-9.+-]*")

@dataclass(frozen=True)
class ImageInfo:
    source_path: str
    mime_type: str
    content_hash: str
    displayed_width_pt: float | None
    displayed_height_pt: float | None

    def __post_init__(self) -> None:
        if is_invisible(self.source_path):
            raise ValueError(f"Invisible source_path: {self.source_path!r}")
        if IMAGE_MIME_TYPE_PATTERN.fullmatch(self.mime_type) is None:
            raise ValueError(f"Invalid mime_type: {self.mime_type!r}: expected 'image/<subtype>'")
        if not self.content_hash.startswith(f"{HASH_ALGORITHM}:"):
            raise ValueError(
                f"Invalid content_hash: {self.content_hash!r} - Must start with '{HASH_ALGORITHM}:'"
            )

        if self.displayed_height_pt is not None and self.displayed_height_pt <= 0:
            raise ValueError(f"displayed_height_pt must be > 0, got {self.displayed_height_pt}")
        
        if self.displayed_width_pt is not None and self.displayed_width_pt <= 0:
            raise ValueError(f"displayed_width_pt must be > 0, got {self.displayed_width_pt}")



@dataclass(frozen=True)
class ParagraphFormat:
    alignment: ParagraphAlignment | None
    style_name: str | None


@dataclass(frozen=True)
class CharSpan:
    """
    Each span starts at 0 <= start < end <= length of text
    """

    style: SpanStyle
    text_range: TextRange


@dataclass(frozen=True, kw_only=True)
class PhysicalBlock:
    """
    Physical representation of a block
    """

    # Basic information
    id: str
    block_type: BlockType
    text: str

    # Location
    region: Region
    page: int | None  # Starting at 1

    # Formatting
    paragraph_format: ParagraphFormat
    char_spans: tuple[CharSpan, ...]

    # Only if the block is a list item
    list_info: ListInfo | None = None

    # Only if the block is an image
    image_info: ImageInfo | None = None

    # Provenance
    source_reference: str

    # TODO: Hint

    # Changelog
    changelog: tuple[str, ...]

    def __post_init__(self) -> None:
        # Text must be visible, except for IMAGE
        if self.block_type != BlockType.IMAGE and is_invisible(self.text):
            raise ValueError(f"Block {self.id}: Contains only invisible/empty text")

        # Check orphaned LIST_ITEM block (no list_info)
        if self.block_type == BlockType.LIST_ITEM and self.list_info is None:
            raise ValueError(
                f"Block {self.id}: Invalid {BlockType.LIST_ITEM.value} block: Doesn't contain list_info"
            )

        # list_info must not be present in blocks that are not LIST_ITEM
        if self.block_type != BlockType.LIST_ITEM and self.list_info is not None:
            raise ValueError(
                f"Block {self.id}: Non-LIST_ITEM block must not contain list_info"
            )

        # Check orphaned IMAGE block (no image_info)
        if self.block_type == BlockType.IMAGE and self.image_info is None:
            raise ValueError(
                f"Block {self.id}: Invalid {BlockType.IMAGE.value} block: Doesn't contain image_info"
            )

        # image_info must not be present in blocks that are not IMAGE
        if self.block_type != BlockType.IMAGE and self.image_info is not None:
            raise ValueError(
                f"Block {self.id}: Non-IMAGE block must not contain image_info"
            )

        # Check spans length definition
        for i, span in enumerate(self.char_spans):
            end = span.text_range.end
            if end > len(self.text):
                raise ValueError(
                    f"Block {self.id}: char span {i} ends at {end}, past text length {len(self.text)}"
                )


@dataclass(frozen=True)
class PhysicalDocument:
    """
    Class representing the entire document
    """

    id: str

    source_filename: str
    source_format: DocSourceFormat

    adapter_name: str
    adapter_version: str

    file_hash: str
    blocks: tuple[PhysicalBlock, ...]
