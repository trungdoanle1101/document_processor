"""Which documents the knowledge base holds, and what produced their chunks.

A record is replaced, never edited: each indexing run saves a new version, and
every version is kept, so an old index or answer can always be explained.
Provenance says what produced the chunks; when any part of it changes (a new
file, profile version, chunk size, embedder or collection), the document is stale.
"""

import json
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from document_processor.retrieval.attributes import AttributeValue, check_attribute_name

_SCHEMA_VERSION = 1


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """What produced a document's chunks; if any part differs, they must be produced again."""

    source_hash: str
    profile: str  # name@version
    chunker: str  # its kind and settings, e.g. "outline/512"
    embedder: str  # the embedder's identity
    collection: str


@dataclass(frozen=True, kw_only=True)
class DocumentRecord:
    document_id: str  # stable across versions of the same document
    title: str
    source: str  # where the file was read from
    access_group: str  # who may read it
    attributes: Mapping[str, AttributeValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    version: int  # 1 for the first indexing, then one more each time
    provenance: Provenance
    chunk_count: int
    indexed_at: datetime

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if not self.access_group.strip():
            raise ValueError("access_group must not be blank")
        if self.version < 1:
            raise ValueError(f"version must be at least 1, got {self.version}")
        for name in self.attributes:
            check_attribute_name(name)
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


class DocumentRegistry(Protocol):
    def get(self, document_id: str) -> DocumentRecord | None: ...

    def list_documents(
        self, *, access_groups: Sequence[str] | None = None
    ) -> list[DocumentRecord]:
        """Current versions, by document id; None: every group."""
        ...

    def history(self, document_id: str) -> list[DocumentRecord]:
        """Every version, oldest first."""
        ...

    def save(self, record: DocumentRecord) -> None:
        """Make this the current version; its version must be one more than the last."""
        ...

    def delete(self, document_id: str) -> None:
        """Forget the document; its history is kept."""
        ...


class SqliteDocumentRegistry:
    """A DocumentRegistry in one SQLite file (or ":memory:")."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    access_group TEXT NOT NULL,
                    record TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    record TEXT NOT NULL,
                    PRIMARY KEY (document_id, version)
                );
                """
            )

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT record FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return None if row is None else record_from_json(row[0])

    def list_documents(
        self, *, access_groups: Sequence[str] | None = None
    ) -> list[DocumentRecord]:
        with self._lock:
            if access_groups is None:
                rows = self._connection.execute(
                    "SELECT record FROM documents ORDER BY document_id"
                ).fetchall()
            else:
                marks = ", ".join("?" for _ in access_groups)
                rows = self._connection.execute(
                    f"SELECT record FROM documents WHERE access_group IN ({marks}) ORDER BY document_id",
                    tuple(access_groups),
                ).fetchall()
        return [record_from_json(row[0]) for row in rows]

    def history(self, document_id: str) -> list[DocumentRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT record FROM document_versions WHERE document_id = ? ORDER BY version",
                (document_id,),
            ).fetchall()
        return [record_from_json(row[0]) for row in rows]

    def save(self, record: DocumentRecord) -> None:
        text = record_to_json(record)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT MAX(version) FROM document_versions WHERE document_id = ?",
                (record.document_id,),
            ).fetchone()
            expected = (row[0] or 0) + 1
            if record.version != expected:
                raise ValueError(
                    f"document {record.document_id!r}: version {record.version} saved, but the next one is {expected}"
                )
            self._connection.execute(
                "INSERT INTO document_versions (document_id, version, record) VALUES (?, ?, ?)",
                (record.document_id, record.version, text),
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO documents (document_id, access_group, record) VALUES (?, ?, ?)",
                (record.document_id, record.access_group, text),
            )

    def delete(self, document_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )

    def close(self) -> None:
        self._connection.close()


def record_to_json(record: DocumentRecord) -> str:
    data = {
        "schema_version": _SCHEMA_VERSION,
        "document_id": record.document_id,
        "title": record.title,
        "source": record.source,
        "access_group": record.access_group,
        "attributes": {name: _encode_value(v) for name, v in record.attributes.items()},
        "version": record.version,
        "provenance": {
            "source_hash": record.provenance.source_hash,
            "profile": record.provenance.profile,
            "chunker": record.provenance.chunker,
            "embedder": record.provenance.embedder,
            "collection": record.provenance.collection,
        },
        "chunk_count": record.chunk_count,
        "indexed_at": record.indexed_at.isoformat(),
    }
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def record_from_json(text: str) -> DocumentRecord:
    data = json.loads(text)
    if data.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported document record schema {data.get('schema_version')!r}"
        )
    return DocumentRecord(
        document_id=data["document_id"],
        title=data["title"],
        source=data["source"],
        access_group=data["access_group"],
        attributes={name: _decode_value(v) for name, v in data["attributes"].items()},
        version=data["version"],
        provenance=Provenance(**data["provenance"]),
        chunk_count=data["chunk_count"],
        indexed_at=datetime.fromisoformat(data["indexed_at"]),
    )


# JSON has no dates or tuples: a date is {"date": "YYYY-MM-DD"}, a tuple a list
def _encode_value(value: AttributeValue) -> Any:
    if isinstance(value, date):
        return {"date": value.isoformat()}
    if isinstance(value, tuple):
        return list(value)
    return value


def _decode_value(value: Any) -> AttributeValue:
    if isinstance(value, dict):
        return date.fromisoformat(value["date"])
    if isinstance(value, list):
        return tuple(value)
    result: AttributeValue = value
    return result


def provenance_differences(old: Provenance, new: Provenance) -> list[str]:
    """Which parts of the provenance changed, by name."""
    return [
        name
        for name in ("source_hash", "profile", "chunker", "embedder", "collection")
        if getattr(old, name) != getattr(new, name)
    ]
