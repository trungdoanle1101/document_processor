import uuid

from kb_server.domain.credentials import Principal
from kb_server.domain.errors import GroupInUse, GroupNotFound, NotAllowed, UserNotFound
from kb_server.domain.groups import EVERYONE, Group
from kb_server.domain.users import User
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.users import UserRepository


class GroupService:
    def __init__(self, *, groups: GroupRepository, users: UserRepository) -> None:
        self._groups = groups
        self._users = users

    def create_group(self, admin: Principal, name: str) -> Group:
        self._require_admin(admin)
        return self._groups.add(name)

    def list_groups(self, admin: Principal) -> list[Group]:
        self._require_admin(admin)
        return self._groups.list_all()

    def delete_group(self, admin: Principal, group_id: uuid.UUID) -> None:
        self._require_admin(admin)
        group = self._get_group(group_id)
        if group.name == EVERYONE:
            raise NotAllowed(f"group {EVERYONE!r} can't be deleted")
        if self._groups.is_in_use(group_id):
            raise GroupInUse(f"group {group.name!r} is a default group of a collection")
        self._groups.delete(group_id)

    def list_members(self, admin: Principal, group_id: uuid.UUID) -> list[User]:
        self._require_admin(admin)
        self._get_group(group_id)
        return self._groups.list_members(group_id)

    def add_member(
        self, admin: Principal, *, user_id: uuid.UUID, group_id: uuid.UUID
    ) -> None:
        self._require_admin(admin)
        self._get_group(group_id)
        if self._users.get(user_id) is None:
            raise UserNotFound("no user with this id")
        self._groups.add_member(user_id=user_id, group_id=group_id)

    def remove_member(
        self, admin: Principal, *, user_id: uuid.UUID, group_id: uuid.UUID
    ) -> None:
        self._require_admin(admin)
        group = self._get_group(group_id)
        if group.name == EVERYONE:
            raise NotAllowed(f"group {EVERYONE!r} can't lose members")
        if self._users.get(user_id) is None:
            raise UserNotFound("no user with this id")
        self._groups.remove_member(user_id=user_id, group_id=group_id)

    def _get_group(self, group_id: uuid.UUID) -> Group:
        group = self._groups.get(group_id)
        if group is None:
            raise GroupNotFound("no group with this id")
        return group

    def _require_admin(self, principal: Principal) -> None:
        if not principal.is_admin:
            raise NotAllowed("only an admin can manage groups")
