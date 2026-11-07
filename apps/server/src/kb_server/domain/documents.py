import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, get_args

DocumentStatus = Literal["draft", "active"]


DOCUMENT_STATUSES = get_args(DocumentStatus)


@dataclass(frozen=True, kw_only=True)
class StoredFile:
    id: uuid.UUID
    sha256: str
    size: int
    media_type: str
    storage_key: str
    created_at: datetime


@dataclass(frozen=True, kw_only=True)
class Document:
    id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    status: DocumentStatus
    current_version: int
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, kw_only=True)
class DocumentVersion:
    document_id: uuid.UUID
    version: int
    file_id: uuid.UUID
    original_filename: str
    provenance: Mapping[str, Any] | None
    uploaded_by: uuid.UUID
    created_at: datetime
