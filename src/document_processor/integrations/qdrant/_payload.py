import json
import uuid
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any, assert_never, cast

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


_DECODE_ELEMENT: Mapping[str, Callable[[Any], Scalar]] = {
    "text": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "date": lambda s: date.fromisoformat(s[:10]),
}


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


def encode_attributes(attributes: Attributes) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        "text": {},
        "integer": {},
        "float": {},
        "boolean": {},
        "date": {},
        "empty": {},
    }

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
    for type_name, fields in stored.items():
        for name, value in fields.items():
            if type_name == "empty":
                result[name] = ()
            elif isinstance(value, list):
                result[name] = cast(
                    AttributeValue,
                    tuple(_DECODE_ELEMENT[type_name](element) for element in value),
                )
            else:
                result[name] = _DECODE_ELEMENT[type_name](value)

    return result


def chunk_payload(chunk: IndexedChunk) -> dict[str, Any]:
    return {
        "kind": "chunk",
        "chunk_key": encode_chunk_key(chunk.key),
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "position": chunk.position,
        "text": chunk.text,
        "embedded_text": chunk.embedded_text,
        "headings": list(chunk.headings),
        "reference_keys": list(chunk.reference_keys),
        "attributes": encode_attributes(chunk.attributes),
    }


def entry_copy_payload(entry: GeneratedEntry, chunk: IndexedChunk) -> dict[str, Any]:
    return {
        "kind": "entry",
        "chunk_key": encode_chunk_key(chunk.key),
        "document_id": entry.document_id,
        "chunk_id": chunk.chunk_id,
        "reference_keys": list(chunk.reference_keys),
        "attributes": encode_attributes(chunk.attributes),
        "generator": entry.generator,
        "entry_id": entry.entry_id,
        "entry_chunk_ids": list(entry.chunk_ids),
        "entry_text": entry.text,
    }


def chunk_from_payload(payload: Mapping[str, Any]) -> IndexedChunk:
    return IndexedChunk(
        document_id=payload["document_id"],
        chunk_id=payload["chunk_id"],
        position=payload["position"],
        text=payload["text"],
        embedded_text=payload["embedded_text"],
        headings=tuple(payload["headings"]),
        reference_keys=tuple(payload["reference_keys"]),
        attributes=decode_attributes(payload["attributes"]),
    )


def entry_from_payload(payload: Mapping[str, Any]) -> GeneratedEntry:
    return GeneratedEntry(
        document_id=payload["document_id"],
        generator=payload["generator"],
        entry_id=payload["entry_id"],
        chunk_ids=tuple(payload["entry_chunk_ids"]),
        text=payload["entry_text"],
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
