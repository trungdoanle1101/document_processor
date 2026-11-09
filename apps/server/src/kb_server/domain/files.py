import uuid
from dataclasses import dataclass
from datetime import datetime

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT = "text/plain"
HTML = "text/html"


@dataclass(frozen=True, kw_only=True)
class StoredFile:
    id: uuid.UUID
    sha256: str
    size: int
    media_type: str
    storage_key: str
    created_at: datetime
