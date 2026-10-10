import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from document_processor.core.models.physical import (
    BlockType,
    CharSpan,
    DocSourceFormat,
    ImageInfo,
    ListInfo,
    ParagraphAlignment,
    ParagraphFormat,
    PhysicalBlock,
    PhysicalDocument,
    Region,
    SpanStyle,
)
from document_processor.core.models.ranges import TextRange

PHYSICAL_SCHEMA_VERSION = 1


def _unwrap_enum_factory(kv_pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in kv_pairs}


def physical_document_to_dict(doc: PhysicalDocument) -> dict[str, Any]:
    d = {
        "kind": "physical_document",
        "schema_version": PHYSICAL_SCHEMA_VERSION,
        "document": asdict(doc, dict_factory=_unwrap_enum_factory),
    }
    return d


def _paragraph_format_from_dict(d: dict[str, Any]) -> ParagraphFormat:
    paragraph_format = ParagraphFormat(
        alignment=ParagraphAlignment(d["alignment"])
        if d["alignment"] is not None
        else None,
        style_name=d["style_name"],
    )
    return paragraph_format


def _text_range_from_dict(d: dict[str, int]) -> TextRange:
    text_range = TextRange(
        start=d["start"],
        end=d["end"],
    )
    return text_range


def _char_span_from_dict(d: dict[str, Any]) -> CharSpan:
    char_span = CharSpan(
        style=SpanStyle(d["style"]), text_range=_text_range_from_dict(d["text_range"])
    )
    return char_span


def _list_info_from_dict(d: dict[str, Any]) -> ListInfo:
    list_info = ListInfo(
        id=d["id"],
        is_ordered=d["is_ordered"],
        displayed_marker=d["displayed_marker"],
        nested_level=d["nested_level"],
    )
    return list_info


def _image_info_from_dict(d: dict[str, Any]) -> ImageInfo:
    image_info = ImageInfo(
        source_path=d["source_path"],
        mime_type=d["mime_type"],
        content_hash=d["content_hash"],
        displayed_height_pt=d["displayed_height_pt"],
        displayed_width_pt=d["displayed_width_pt"],
    )
    return image_info


def _physical_block_from_dict(d: dict[str, Any]) -> PhysicalBlock:
    block = PhysicalBlock(
        id=d["id"],
        block_type=BlockType(d["block_type"]),
        text=d["text"],
        region=Region(d["region"]),
        page=d["page"],
        paragraph_format=_paragraph_format_from_dict(d["paragraph_format"]),
        char_spans=tuple(_char_span_from_dict(cs) for cs in d["char_spans"]),
        list_info=(
            _list_info_from_dict(d["list_info"]) if d["list_info"] is not None else None
        ),
        image_info=(
            _image_info_from_dict(d["image_info"])
            if d["image_info"] is not None
            else None
        ),
        source_reference=d["source_reference"],
        changelog=tuple(d["changelog"]),
    )
    return block


def physical_document_from_dict(d: dict[str, Any]) -> PhysicalDocument:
    kind = d.get("kind")
    if kind != "physical_document":
        raise ValueError(
            f"Unexpected dictionary kind: Expected: physical_document. Actual: {kind}"
        )
    schema_version = d.get("schema_version")
    if schema_version != PHYSICAL_SCHEMA_VERSION:
        raise ValueError(
            f"Mismatched schema version: Expected {PHYSICAL_SCHEMA_VERSION}. Actual: {schema_version}"
        )

    document_dict = d["document"]
    document = PhysicalDocument(
        id=document_dict["id"],
        source_filename=document_dict["source_filename"],
        source_format=DocSourceFormat(document_dict["source_format"]),
        adapter_name=document_dict["adapter_name"],
        adapter_version=document_dict["adapter_version"],
        file_hash=document_dict["file_hash"],
        blocks=tuple(_physical_block_from_dict(b) for b in document_dict["blocks"]),
    )
    return document


def save_physical_document(doc: PhysicalDocument, file_path: str | Path) -> None:
    """Save a PhysicalDocument as JSON format at file_path.
    Will create parent folders if they don't exist.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    doc_dict = physical_document_to_dict(doc)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2, ensure_ascii=False)


def load_physical_document(file_path: str | Path) -> PhysicalDocument:
    """Load a PhysicalDocument from a JSON file."""
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        doc_dict = json.load(f)
    return physical_document_from_dict(doc_dict)
