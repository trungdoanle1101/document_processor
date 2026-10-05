from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Literal

class BlockType(Enum):
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    IMAGE = "image"

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
    original_marker: str | None
    nested_level: int # Starting at 0

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
    start: int
    end: int

@dataclass(frozen=True)
class PhysicalBlock:
    """
        Physical representation of text
    """
    # Basic information
    id: str
    block_type: BlockType
    text: str
    
    # Location
    region: Region
    page: int | None # Starting at 1
    
    # Formatting
    paragraph_format: ParagraphFormat 
    char_spans: tuple[CharSpan, ...]

    # Only if the block is a list item
    list_info: ListInfo | None


    # Provenance
    source_reference: str

    # Hints (don't know what to fill yet)

    # Changelog
    changelog: tuple[str, ...]



class DocSourceFormat(Enum):
    DOCX = "docx"
    PDF = "pdf"
    MARKDOWN = "md"
    TXT = "txt"


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



@dataclass
class LogicalNode:
    """
        Logical representation of text
    """ 
    id: str
    block: PhysicalBlock
    parent: LogicalNode
    children: list[LogicalNode]
