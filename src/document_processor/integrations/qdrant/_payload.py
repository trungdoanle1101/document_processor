import json
import uuid
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any, Literal, assert_never, cast

from qdrant_client import models

from document_processor.retrieval.attributes import Attributes, AttributeValue, Scalar
from document_processor.retrieval.embedding import DenseVector, Embedding, SparseVector
from document_processor.retrieval.index import (
    ChunkKey,
    GeneratedEntry,
    GeneratedEntryKey,
    IndexedChunk,
)

_NAMESPACE = uuid.UUID("98e50f22-0424-4e60-9dd9-72fd0cc4e9a8")

# Payload fields: written here, filtered on in _filters.py and index.py
KIND = "kind"
CHUNK_KEY = "chunk_key"
DOCUMENT_ID = "document_id"
CHUNK_ID = "chunk_id"
POSITION = "position"
TEXT = "text"
EMBEDDED_TEXT = "embedded_text"
HEADINGS = "headings"
REFERENCE_KEYS = "reference_keys"
ATTRIBUTES = "attributes"
GENERATOR = "generator"
ENTRY_ID = "entry_id"
ENTRY_CHUNK_IDS = "entry_chunk_ids"
ENTRY_TEXT = "entry_text"

type PointKind = Literal["chunk", "entry"]
CHUNK: PointKind = "chunk"
ENTRY: PointKind = "entry"

type StorageKind = Literal["text", "integer", "float", "boolean", "date", "empty"]
STORAGE_KINDS: tuple[StorageKind, ...] = (
    "text",
    "integer",
    "float",
    "boolean",
    "date",
    "empty",
)

_DECODE_ELEMENT: Mapping[StorageKind, Callable[[Any], Scalar]] = {
    "text": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "date": lambda s: date.fromisoformat(s[:10]),
}

NUMBER_KINDS: tuple[StorageKind, ...] = (
    "integer",
    "float",
)


def attribute_key(kind: StorageKind, name: str) -> str:
    """Where one attribute's value is stored: attributes.<kind>.<name>."""
    return f"{ATTRIBUTES}.{kind}.{name}"


def attribute_group_key(kind: StorageKind) -> str:
    """Where all attributes of one kind are stored, for set_payload's key."""
    return f"{ATTRIBUTES}.{kind}"


def encode_chunk_key(key: ChunkKey) -> str:
    return json.dumps([key.document_id, key.chunk_id], ensure_ascii=False)


def chunk_point_id(key: ChunkKey) -> str:
    name = json.dumps(["chunk", key.document_id, key.chunk_id], ensure_ascii=False)
    return str(uuid.uuid5(namespace=_NAMESPACE, name=name))


def entry_point_id(key: GeneratedEntryKey, chunk_id: str) -> str:
    name = json.dumps(
        ["entry", key.document_id, key.generator, key.entry_id, chunk_id],
        ensure_ascii=False,
    )
    return str(uuid.uuid5(namespace=_NAMESPACE, name=name))


def encode_attributes(attributes: Attributes) -> dict[StorageKind, dict[str, Any]]:
    result: dict[StorageKind, dict[str, Any]] = {kind: {} for kind in STORAGE_KINDS}

    for name, value in attributes.items():
        match value:
            case bool():
                result["boolean"][name] = value
            case int():
                result["integer"][name] = value
            case float():
                result["float"][name] = value
            case str():
                result["text"][name] = value
            case date():
                result["date"][name] = f"{value.isoformat()}T00:00:00Z"
            case tuple():
                if not value:
                    result["empty"][name] = []
                elif type(value[0]) is str:
                    result["text"][name] = list(value)
                elif type(value[0]) is int:
                    result["integer"][name] = list(value)
                else:
                    raise TypeError(
                        f"unsupported tuple with elements of type {type(value[0]).__name__}"
                    )
            case _:
                raise TypeError(f"unsupported type {type(value).__name__}")

    return result


def decode_attributes(
    stored: Mapping[str, Mapping[str, Any]],
) -> dict[str, AttributeValue]:
    result: dict[str, AttributeValue] = {}
    for kind in STORAGE_KINDS:
        for name, value in stored.get(kind, {}).items():
            if kind == "empty":
                result[name] = ()
            elif isinstance(value, list):
                result[name] = cast(
                    AttributeValue,
                    tuple(_DECODE_ELEMENT[kind](element) for element in value),
                )
            else:
                result[name] = _DECODE_ELEMENT[kind](value)

    return result


def chunk_payload(chunk: IndexedChunk) -> dict[str, Any]:
    return {
        KIND: CHUNK,
        DOCUMENT_ID: chunk.document_id,
        CHUNK_ID: chunk.chunk_id,
        CHUNK_KEY: encode_chunk_key(chunk.key),
        POSITION: chunk.position,
        TEXT: chunk.text,
        EMBEDDED_TEXT: chunk.embedded_text,
        HEADINGS: list(chunk.headings),
        REFERENCE_KEYS: list(chunk.reference_keys),
        ATTRIBUTES: encode_attributes(chunk.attributes),
    }


def entry_copy_payload(entry: GeneratedEntry, chunk: IndexedChunk) -> dict[str, Any]:
    return {
        KIND: ENTRY,
        DOCUMENT_ID: chunk.document_id,
        CHUNK_ID: chunk.chunk_id,
        CHUNK_KEY: encode_chunk_key(chunk.key),
        REFERENCE_KEYS: list(chunk.reference_keys),
        ATTRIBUTES: encode_attributes(chunk.attributes),
        GENERATOR: entry.generator,
        ENTRY_ID: entry.entry_id,
        ENTRY_CHUNK_IDS: list(entry.chunk_ids),
        ENTRY_TEXT: entry.text,
    }


def chunk_from_payload(payload: Mapping[str, Any]) -> IndexedChunk:
    return IndexedChunk(
        document_id=payload[DOCUMENT_ID],
        chunk_id=payload[CHUNK_ID],
        position=payload[POSITION],
        text=payload[TEXT],
        embedded_text=payload[EMBEDDED_TEXT],
        headings=tuple(payload[HEADINGS]),
        reference_keys=tuple(payload[REFERENCE_KEYS]),
        attributes=decode_attributes(payload[ATTRIBUTES]),
    )


def entry_from_payload(payload: Mapping[str, Any]) -> GeneratedEntry:
    return GeneratedEntry(
        document_id=payload[DOCUMENT_ID],
        generator=payload[GENERATOR],
        entry_id=payload[ENTRY_ID],
        chunk_ids=tuple(payload[ENTRY_CHUNK_IDS]),
        text=payload[ENTRY_TEXT],
    )


def point_vectors(embedding: Embedding) -> dict[str, models.Vector]:
    result: dict[str, models.Vector] = {}
    for name, vector in embedding.vectors.items():
        match vector:
            case DenseVector(values=values):
                result[name] = list(values)
            case SparseVector(indices=indices, values=values):
                result[name] = models.SparseVector(
                    indices=list(indices),
                    values=list(values),
                )
            case _:
                assert_never(vector)
    return result
