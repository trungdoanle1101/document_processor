import uuid
from dataclasses import dataclass
from datetime import datetime

EVERYONE = "everyone"


@dataclass(frozen=True, kw_only=True)
class Group:
    id: uuid.UUID
    name: str
    created_at: datetime
