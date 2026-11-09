import io
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from document_processor.chunking import Chunk
from document_processor.chunking.serialization import chunk_from_dict, chunk_to_dict
from document_processor.core.models import LogicalDocument
from document_processor.core.serialization import (
    logical_document_from_dict,
    logical_document_to_dict,
)
from kb_server.storage.files import FileStore, prepared_folder, prepared_version_folder


@dataclass(frozen=True, kw_only=True)
class PreparationSettings:
    """Everything that decides how a file is prepared. Different settings, different chunks."""

    # candidate profiles tried in order, as name@version
    profiles: tuple[str, ...]
    # used when no candidate finds structure, e.g. plain@1
    fallback_profile: str
    # kind and parameters for chunker: e.g., {"kind": "outline", "max_tokens": 512}
    chunker: Mapping[str, Any]
    # the identity of the embedder whose tokenizer sized the chunks
    embedder: str

    def describe(self) -> dict[str, Any]:
        """The settings in full, for provenance."""
        return {
            "profiles": list(self.profiles),
            "fallback_profile": self.fallback_profile,
            "chunker": dict(self.chunker),
            "embedder": self.embedder,
        }


@dataclass(frozen=True, kw_only=True)
class PreparedParts:
    logical: LogicalDocument
    chunks: tuple[Chunk, ...]


def save_prepared(
    store: FileStore, document_id: uuid.UUID, version: int, parts: PreparedParts
) -> None:
    """Replace whatever is saved for this version with these parts (logical.json and chunks.jsonl)."""
    folder = prepared_version_folder(document_id, version)
    store.delete_folder(folder)
    _put_json(
        store,
        _logical_key(folder),
        json.dumps(logical_document_to_dict(parts.logical), ensure_ascii=False),
        "application/json",
    )
    _put_json(
        store,
        _chunks_key(folder),
        "".join(
            json.dumps(chunk_to_dict(chunk), ensure_ascii=False) + "\n"
            for chunk in parts.chunks
        ),
        "application/x-ndjson",
    )


def load_prepared(
    store: FileStore, document_id: uuid.UUID, version: int
) -> PreparedParts | None:
    """Load the prepared parts for this version (logical.json and chunks.jsonl).
    None if either file is missing (nothing saved, or a save that stopped halfway).
    """
    folder = prepared_version_folder(document_id, version)
    try:
        with store.open(_logical_key(folder)) as stream:
            logical = logical_document_from_dict(json.load(stream))
        with store.open(_chunks_key(folder)) as stream:
            chunks = tuple(
                chunk_from_dict(json.loads(line)) for line in stream if line.strip()
            )
    except FileNotFoundError:
        return None
    return PreparedParts(logical=logical, chunks=chunks)


def delete_prepared(store: FileStore, document_id: uuid.UUID) -> int:
    """Delete everything prepared for a document; return how many files went."""
    return store.delete_folder(prepared_folder(document_id))


def _put_json(files: FileStore, key: str, text: str, media_type: str) -> None:
    data = text.encode("utf-8")
    files.put(key, io.BytesIO(data), size=len(data), media_type=media_type)


def _logical_key(folder: str) -> str:
    return f"{folder}/logical.json"


def _chunks_key(folder: str) -> str:
    return f"{folder}/chunks.jsonl"
