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
    TableInfo,
    TableCell,
    ParagraphAlignment,
    ParagraphFormat,
    PhysicalBlock,
    PhysicalDocument,
    Region,
    SpanStyle,
)

from document_processor.core.models.logical import (
    LogicalNode,
    LogicalDocument,
    BlockSpan,
    ContentItem,
    ChildRef
)
from document_processor.core.models.ranges import TextRange

PHYSICAL_SCHEMA_VERSION = 1
LOGICAL_SCHEMA_VERSION = 1
PHYSICAL_DOCUMENT_KIND = "physical_document"
LOGICAL_DOCUMENT_KIND = "logical_document"


def _unwrap_enum_factory(kv_pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in kv_pairs}

# === Physical

def physical_document_to_dict(doc: PhysicalDocument) -> dict[str, Any]:
    d = {
        "kind": PHYSICAL_DOCUMENT_KIND,
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


def _table_cell_from_dict(d: dict[str, Any]) -> TableCell:
    table_cell = TableCell(
        row=d["row"],
        col=d["col"],
        row_span=d["row_span"],
        col_span=d["col_span"],
        paragraphs=tuple(d["paragraphs"]),
    )
    return table_cell


def _table_info_from_dict(d: dict[str, Any]) -> TableInfo:
    table_info = TableInfo(
        n_rows=d["n_rows"],
        n_cols=d["n_cols"],
        n_header_rows=d["n_header_rows"],
        cells=tuple(_table_cell_from_dict(cell) for cell in d["cells"]),
    )
    return table_info


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
        table_info=(
            _table_info_from_dict(d["table_info"])
            if d["table_info"] is not None
            else None
        ),
        source_reference=d["source_reference"],
        changelog=tuple(d["changelog"]),
    )
    return block


def physical_document_from_dict(d: dict[str, Any]) -> PhysicalDocument:
    kind = d.get("kind")
    if kind != PHYSICAL_DOCUMENT_KIND:
        raise ValueError(
            f"Unexpected dictionary kind: Expected: {PHYSICAL_DOCUMENT_KIND!r}. Actual: {kind!r}"
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

# === Logical

def _content_item_to_dict(item: ContentItem) -> dict[str, Any]:
    d = asdict(item)
    if isinstance(item, BlockSpan):
        d["kind"] = "block_span"
    elif isinstance(item, ChildRef):
        d["kind"] = "child_ref"
    else:
        raise TypeError(f"unknown type for argument 'item': expected BlockSpan or ChildRef, got {type(item).__name__}")
    return d


def _logical_node_to_dict(n: LogicalNode) -> dict[str, Any]:
    return {
        "id": n.id,
        "level": n.level,
        "number": n.number,
        "marker": asdict(n.marker) if n.marker is not None else None,
        "title": [asdict(b) for b in n.title],
        "content": [_content_item_to_dict(item) for item in n.content]
    }


def logical_document_to_dict(doc: LogicalDocument) -> dict[str, Any]:
    document_dict = {
        "id": doc.id,
        "physical_document_id": doc.physical_document_id,
        "source_file_hash": doc.source_file_hash,
        "profile_name": doc.profile_name,
        "profile_version": doc.profile_version,
        "front_matter": [asdict(b) for b in doc.front_matter],
        "root_ids": list(doc.root_ids),
        "nodes": [_logical_node_to_dict(n) for n in doc.nodes],
        "end_matter": [asdict(b) for b in doc.end_matter]
    }

    return {
        "kind": LOGICAL_DOCUMENT_KIND,
        "schema_version": LOGICAL_SCHEMA_VERSION,
        "document": document_dict
    }

def _block_span_from_dict(d: dict[str, Any]) -> BlockSpan:
    block_span = BlockSpan(
        block_id=d["block_id"],
        text_range=_text_range_from_dict(d["text_range"]) 
    )
    return block_span

def _child_ref_from_dict(d: dict[str, str]) -> ChildRef:
    return ChildRef(node_id=d["node_id"])

def _logical_node_from_dict(d: dict[str, Any]) -> LogicalNode:
    content: list[ContentItem] = []
    for item in d["content"]:
        if item["kind"] == "block_span":
            content.append(_block_span_from_dict(item))
        elif item["kind"] == "child_ref":
            content.append(_child_ref_from_dict(item))
        else:
            raise ValueError(f"unknown kind: expected 'block_span' | 'child_ref', got {item["kind"]!r}")

    logical_node = LogicalNode(
        id=d["id"],
        level=d["level"],
        number=d["number"],
        marker=_block_span_from_dict(d["marker"]) if d["marker"] is not None else None,
        title=tuple(_block_span_from_dict(b) for b in d["title"]),
        content=tuple(content)
    )
    return logical_node


def logical_document_from_dict(d: dict[str, Any]) -> LogicalDocument:
    kind = d.get("kind")
    if kind != LOGICAL_DOCUMENT_KIND:
        raise ValueError(
            f"Unexpected dictionary kind: Expected: {LOGICAL_DOCUMENT_KIND!r}. Actual: {kind!r}"
        )
    schema_version = d.get("schema_version")
    if schema_version != LOGICAL_SCHEMA_VERSION:
        raise ValueError(
            f"Mismatched schema version: Expected: {LOGICAL_SCHEMA_VERSION}. Actual: {schema_version}"
        )
    document_dict = d["document"]
    document = LogicalDocument(
        id=document_dict["id"],
        physical_document_id=document_dict["physical_document_id"],
        source_file_hash=document_dict["source_file_hash"],
        profile_name=document_dict["profile_name"],
        profile_version=document_dict["profile_version"],
        front_matter=tuple(_block_span_from_dict(b) for b in document_dict["front_matter"]),
        root_ids=tuple(document_dict["root_ids"]),
        nodes=tuple(_logical_node_from_dict(n) for n in document_dict["nodes"]),
        end_matter=tuple(_block_span_from_dict(b) for b in document_dict["end_matter"])
    )
    return document


def save_logical_document(doc: LogicalDocument, file_path: str | Path) -> None:
    """Save a LogicalDocument as JSON format at file_path.
    Will create parent folders if they don't exist.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    doc_dict = logical_document_to_dict(doc)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2, ensure_ascii=False)


def load_logical_document(file_path: str | Path) -> LogicalDocument:
    """Load a LogicalDocument from a JSON file."""
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        doc_dict = json.load(f)
    return logical_document_from_dict(doc_dict)
