import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence

from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kb_server.domain.collections import Collection
from kb_server.domain.errors import CollectionSlugTaken
from kb_server.domain.groups import Group
from kb_server.domain.users import User
from kb_server.storage.models import (
    CollectionGroupRow,
    CollectionManagerRow,
    CollectionRow,
    GroupRow,
    UserGroupRow,
    UserRow,
)
from kb_server.storage.repositories.groups import to_group
from kb_server.storage.repositories.users import to_user


class CollectionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, slug: str, title: str, description: str) -> Collection:
        row = CollectionRow(
            slug=slug,
            title=title,
            description=description,
        )
        try:
            with self._session.begin_nested():
                self._session.add(row)
                self._session.flush()
        except IntegrityError:
            if self.get_by_slug(slug) is not None:
                raise CollectionSlugTaken(
                    f"a collection with slug {slug!r} already exists"
                )
            raise
        return to_collection(row)

    def get(self, collection_id: uuid.UUID) -> Collection | None:
        row = self._session.get(CollectionRow, collection_id)
        return to_collection(row) if row is not None else None

    def update(
        self, collection_id: uuid.UUID, *, slug: str, title: str, description: str
    ) -> None:
        try:
            with self._session.begin_nested():
                self._session.execute(
                    update(CollectionRow)
                    .where(CollectionRow.id == collection_id)
                    .values(slug=slug, title=title, description=description)
                )
        except IntegrityError:
            raise CollectionSlugTaken(
                f"a collection with slug {slug!r} already exists"
            ) from None

    def delete(self, collection_id: uuid.UUID) -> None:
        self._session.execute(
            delete(CollectionRow).where(CollectionRow.id == collection_id)
        )

    def get_by_slug(self, slug: str) -> Collection | None:
        row = self._session.scalar(
            select(CollectionRow).where(CollectionRow.slug == slug)
        )
        return to_collection(row) if row is not None else None

    def list_all(self) -> list[Collection]:
        rows = self._session.scalars(select(CollectionRow).order_by(CollectionRow.slug))
        return [to_collection(row) for row in rows]

    def list_groups(self, collection_id: uuid.UUID) -> list[Group]:
        rows = self._session.scalars(
            select(GroupRow)
            .join(CollectionGroupRow, CollectionGroupRow.group_id == GroupRow.id)
            .where(CollectionGroupRow.collection_id == collection_id)
            .order_by(GroupRow.name)
        ).all()
        return [to_group(row) for row in rows]

    def set_groups(
        self, collection_id: uuid.UUID, group_ids: Iterable[uuid.UUID]
    ) -> None:
        self._session.execute(
            delete(CollectionGroupRow).where(
                CollectionGroupRow.collection_id == collection_id
            )
        )
        rows = [
            {"collection_id": collection_id, "group_id": group_id}
            for group_id in set(group_ids)
        ]
        if rows:
            self._session.execute(insert(CollectionGroupRow), rows)

    def groups_by_collection(
        self, collection_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, list[Group]]:
        rows = self._session.execute(
            select(CollectionGroupRow.collection_id, GroupRow)
            .join(GroupRow, GroupRow.id == CollectionGroupRow.group_id)
            .where(CollectionGroupRow.collection_id.in_(collection_ids))
            .order_by(GroupRow.name)
        ).all()

        found: dict[uuid.UUID, list[Group]] = defaultdict(list)
        for collection_id, group_row in rows:
            found[collection_id].append(to_group(group_row))
        return dict(found)

    def is_visible_to(self, collection_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        shared_group = (
            select(CollectionGroupRow.group_id)
            .join(UserGroupRow, CollectionGroupRow.group_id == UserGroupRow.group_id)
            .where(
                CollectionGroupRow.collection_id == collection_id,
                UserGroupRow.user_id == user_id,
            )
        )
        return bool(self._session.scalar(select(shared_group.exists())))

    def list_visible_to(self, user_id: uuid.UUID) -> list[Collection]:
        shared_group = (
            select(CollectionGroupRow.collection_id)
            .join(UserGroupRow, CollectionGroupRow.group_id == UserGroupRow.group_id)
            .where(
                CollectionGroupRow.collection_id == CollectionRow.id,
                UserGroupRow.user_id == user_id,
            )
        )
        rows = self._session.scalars(
            select(CollectionRow)
            .where(shared_group.exists())
            .order_by(CollectionRow.slug)
        ).all()
        return [to_collection(row) for row in rows]

    def list_managers(self, collection_id: uuid.UUID) -> list[User]:
        rows = self._session.scalars(
            select(UserRow)
            .join(CollectionManagerRow, CollectionManagerRow.user_id == UserRow.id)
            .where(CollectionManagerRow.collection_id == collection_id)
            .order_by(UserRow.email)
        ).all()
        return [to_user(row) for row in rows]

    def add_manager(self, *, collection_id: uuid.UUID, user_id: uuid.UUID) -> None:
        row = CollectionManagerRow(collection_id=collection_id, user_id=user_id)
        try:
            with self._session.begin_nested():
                self._session.add(row)
                self._session.flush()
        except IntegrityError:
            pass

    def remove_manager(self, *, collection_id: uuid.UUID, user_id: uuid.UUID) -> None:
        self._session.execute(
            delete(CollectionManagerRow).where(
                CollectionManagerRow.collection_id == collection_id,
                CollectionManagerRow.user_id == user_id,
            )
        )

    def list_manager_ids(self, collection_id: uuid.UUID) -> set[uuid.UUID]:
        manager_ids = self._session.scalars(
            select(CollectionManagerRow.user_id).where(
                CollectionManagerRow.collection_id == collection_id
            )
        ).all()
        return set(manager_ids)


def to_collection(row: CollectionRow) -> Collection:
    return Collection(
        id=row.id,
        slug=row.slug,
        title=row.title,
        description=row.description,
        created_at=row.created_at,
    )
