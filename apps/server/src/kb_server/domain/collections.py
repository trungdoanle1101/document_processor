import uuid
from dataclasses import dataclass
from datetime import datetime

from kb_server.domain.groups import Group


@dataclass(frozen=True, kw_only=True)
class Collection:
    id: uuid.UUID
    slug: str
    title: str
    description: str
    created_at: datetime


@dataclass(frozen=True, kw_only=True)
class CollectionDetails:
    """A collection with its default read groups, as callers see it."""

    collection: Collection
    groups: list[Group]


def normalize_slug(slug: str) -> str:
    return slug.strip().lower()
