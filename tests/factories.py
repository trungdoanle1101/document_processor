"""Builders for test data.

Every test builds its models through these helpers and passes only the fields
it cares about. When a model gains a field, only this file changes.
"""

from typing import Any

from document_processor.core.identity import compute_hash
from document_processor.core.models.physical import (
    BlockType,
    DocSourceFormat,
    ImageInfo,
    ListInfo,
    ParagraphFormat,
    PhysicalBlock,
    PhysicalDocument,
    Region,
)


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
