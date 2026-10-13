"""Builders for test data.

Every test builds its models through these helpers and passes only the fields
it cares about. When a model gains a field, only this file changes.
"""

from typing import Any

from document_processor.core.identity import compute_hash
from document_processor.core.models import (
    BlockType,
    DocSourceFormat,
    ImageInfo,
    ListInfo,
    ParagraphFormat,
    PhysicalBlock,
    PhysicalDocument,
    Region,
    TableCell,
    TableInfo,
)

from document_processor.core.models.logical import (
    BlockSpan,
    LogicalDocument,
    LogicalNode,
)
from document_processor.core.models.ranges import TextRange


def make_list_info(**overrides: Any) -> ListInfo:
    fields: dict[str, Any] = {
        "id": "list1",
        "is_ordered": True,
        "displayed_marker": "1.",
        "nested_level": 0,
    }
    fields.update(overrides)
    return ListInfo(**fields)


def make_image_info(**overrides: Any) -> ImageInfo:
    fields: dict[str, Any] = {
        "source_path": "word/media/image1.png",
        "mime_type": "image/png",
        "content_hash": compute_hash(b"fake png bytes"),
        "displayed_width_pt": 120.0,
        "displayed_height_pt": 80.0,
    }
    fields.update(overrides)
    return ImageInfo(**fields)


def make_block(**overrides: Any) -> PhysicalBlock:
    """A valid PARAGRAPH block; override any field."""
    fields: dict[str, Any] = {
        "id": "pb1",
        "block_type": BlockType.PARAGRAPH,
        "text": "Điều 1. Phạm vi điều chỉnh",
        "region": Region.BODY,
        "page": None,
        "paragraph_format": ParagraphFormat(alignment=None, style_name=None),
        "char_spans": (),
        "source_reference": "paragraph:1",
        "changelog": (),
    }
    fields.update(overrides)
    return PhysicalBlock(**fields)


def make_document(**overrides: Any) -> PhysicalDocument:
    """A valid document with two blocks; override any field."""
    file_hash = compute_hash(b"fake source bytes")
    fields: dict[str, Any] = {
        "id": file_hash,
        "source_filename": "nghi-dinh-13.docx",
        "source_format": DocSourceFormat.DOCX,
        "adapter_name": "test",
        "adapter_version": "0.0.0",
        "file_hash": file_hash,
        "blocks": (make_block(id="pb1"), make_block(id="pb2")),
    }
    fields.update(overrides)
    return PhysicalDocument(**fields)


def make_table_cell(**overrides: Any) -> TableCell:
    """A valid 1x1 cell at (0, 0) with one paragraph; override any field."""
    fields: dict[str, Any] = {
        "row": 0,
        "col": 0,
        "row_span": 1,
        "col_span": 1,
        "paragraphs": ("Nội dung",),
    }
    fields.update(overrides)
    return TableCell(**fields)


def make_table_info(**overrides: Any) -> TableInfo:
    """A valid 2x2 table with one header row; override any field.

        TT | Nội dung
        1  | Phạm vi điều chỉnh
    """
    fields: dict[str, Any] = {
        "n_rows": 2,
        "n_cols": 2,
        "n_header_rows": 1,
        "cells": (
            make_table_cell(row=0, col=0, paragraphs=("TT",)),
            make_table_cell(row=0, col=1, paragraphs=("Nội dung",)),
            make_table_cell(row=1, col=0, paragraphs=("1",)),
            make_table_cell(row=1, col=1, paragraphs=("Phạm vi điều chỉnh",)),
        ),
    }
    fields.update(overrides)
    return TableInfo(**fields)


def make_table_block(**overrides: Any) -> PhysicalBlock:
    """A valid TABLE block whose text is derived from its table_info."""
    table_info = overrides.pop("table_info", None) or make_table_info()
    fields: dict[str, Any] = {
        "block_type": BlockType.TABLE,
        "table_info": table_info,
        "text": table_info.text(),
    }
    fields.update(overrides)
    return make_block(**fields)


def make_block_span(**overrides: Any) -> BlockSpan:
    fields: dict[str, Any] = {
        "block_id": "pb1",
        "text_range": TextRange(0, 7),
    }
    fields.update(overrides)
    return BlockSpan(**fields)


def make_logical_node(**overrides: Any) -> LogicalNode:
    """A valid leaf node (Điều 1, no children); override any field."""
    fields: dict[str, Any] = {
        "id": "dieu-1",
        "level": "dieu",
        "number": "1",
        "marker": make_block_span(),
        "title": (),
        "body": (),
        "children_ids": (),
    }
    fields.update(overrides)
    return LogicalNode(**fields)


def make_logical_document(**overrides: Any) -> LogicalDocument:
    """A valid document with no nodes; override any field.

    Pass ``nodes`` and ``root_ids`` together, since they must agree.
    """
    fields: dict[str, Any] = {
        "id": "ld1",
        "physical_document_id": "pd1",
        "source_file_hash": "hash1",
        "profile_name": "vn_regulation",
        "profile_version": "1.0.0",
        "front_matter": (),
        "root_ids": (),
        "nodes": (),
        "end_matter": (),
    }
    fields.update(overrides)
    return LogicalDocument(**fields)