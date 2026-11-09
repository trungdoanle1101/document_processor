import hashlib
import io
import json
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
from kb_server.storage.files import FileStore


@dataclass(frozen=True, kw_only=True)
class PreparationSettings:
    """Everything that decides how a file is prepared. Different settings, different chunks."""

    profiles: tuple[
        str, ...
    ]  # tuples of name@version - naming the candidate profiles to try
    fallback_profile: str
    chunker: Mapping[
        str, Any
    ]  # kind and parameters for chunker: e.g., {"kind": "outline", "max_tokens": 512}
    embedder: str  # the identity of the embedder whose tokenizer sized the chunks

    def fingerprint(self) -> str:
        """12 hex characters of the sha256 of these settings, as canonical JSON."""
        text = json.dumps(
            self.describe(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def describe(self) -> dict[str, Any]:
        """The settings in full, for provenance."""
        return {
            "profile": list(self.profiles),
            "fallback_profile": self.fallback_profile,
            "chunker": dict(self.chunker),
            "embedder": self.embedder,
        }


@dataclass(frozen=True, kw_only=True)
class PreparedParts:
    logical: LogicalDocument
    chunks: tuple[Chunk, ...]


def save_prepared(files: FileStore, prefix: str, parts: PreparedParts) -> None:
    """Save prepared parts into the cache.
    Prefix is the folder-like prefix under which a file's prepared parts (logical.json, chunks.jsonl) are kept.
    """
    _put_json(
        files,
        _logical_key(prefix),
        json.dumps(logical_document_to_dict(parts.logical), ensure_ascii=False),
        "application/json",
    )
    _put_json(
        files,
        _chunks_key(prefix),
        "".join(
            json.dumps(chunk_to_dict(chunk), ensure_ascii=False) + "\n"
            for chunk in parts.chunks
        ),
        "application/x-ndjson",
    )


def load_prepared(files: FileStore, prefix: str) -> PreparedParts | None:
    """The cached parts, or None if they aren't (all) there.
    Prefix is the folder-like prefix under which a file's prepared parts (logical.json, chunks.jsonl) are kept.
    """
    try:
        with files.open(_logical_key(prefix)) as stream:
            logical = logical_document_from_dict(json.load(stream))
        with files.open(_chunks_key(prefix)) as stream:
            chunks = tuple(
                chunk_from_dict(json.loads(line)) for line in stream if line.strip()
            )
    except FileNotFoundError:
        return None
    return PreparedParts(logical=logical, chunks=chunks)


def _put_json(files: FileStore, key: str, text: str, media_type: str) -> None:
    data = text.encode("utf-8")
    files.put(key, io.BytesIO(data), size=len(data), media_type=media_type)


def _logical_key(prefix: str) -> str:
    return f"{prefix}/logical.json"


def _chunks_key(prefix: str) -> str:
    return f"{prefix}/chunks.jsonl"
