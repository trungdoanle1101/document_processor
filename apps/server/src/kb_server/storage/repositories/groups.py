import uuid

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kb_server.domain.errors import GroupNameTaken
from kb_server.domain.groups import Group
from kb_server.domain.users import User
from kb_server.storage.models import GroupRow, UserGroupRow, UserRow
from kb_server.storage.repositories.users import to_user


class GroupRepository:
    def __init__(self, session: Session):
        self._session = session

    def add(self, name: str) -> Group:
        row = GroupRow(name=name)
        try:
            with self._session.begin_nested():
                self._session.add(row)
                self._session.flush()
        except IntegrityError:
            raise GroupNameTaken(f"group {name} already exists")
        return to_group(row)

    def get(self, group_id: uuid.UUID) -> Group | None:
        row = self._session.get(GroupRow, group_id)
        if row is None:
            return None
        return to_group(row)

    def get_by_name(self, name: str) -> Group | None:
        return self._session.scalar(select(GroupRow).where(GroupRow.name == name))

    def list_all(self) -> list[Group]:
        rows = self._session.scalars(select(GroupRow).order_by(GroupRow.name)).all()
        return [to_group(row) for row in rows]

    def delete(self, group_id: uuid.UUID) -> None:
        self._session.execute(delete(GroupRow).where(GroupRow.id == group_id))

    def list_members(self, group_id: uuid.UUID) -> list[User]:
        user_rows = self._session.scalars(
            select(UserRow)
            .join(UserGroupRow, UserRow.id == UserGroupRow.user_id)
            .where(UserGroupRow.group_id == group_id)
            .order_by(UserRow.email)
        ).all()
        return [to_user(row) for row in user_rows]

    def add_member(self, *, user_id: uuid.UUID, group_id: uuid.UUID) -> None:
        try:
            with self._session.begin_nested():
                row = UserGroupRow(user_id=user_id, group_id=group_id)
                self._session.add(row)
                self._session.flush()
        except IntegrityError:
            pass

    def remove_member(self, *, user_id: uuid.UUID, group_id: uuid.UUID) -> None:
        self._session.execute(
            delete(UserGroupRow).where(
                UserGroupRow.user_id == user_id, UserGroupRow.group_id == group_id
            )
        )


def to_group(row: GroupRow) -> Group:
    return Group(id=row.id, name=row.name, created_at=row.created_at)
