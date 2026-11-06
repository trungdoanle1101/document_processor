import uuid
from collections.abc import Iterable, Sequence
from dataclasses import replace

from kb_server.domain.collections import Collection, CollectionDetails, normalize_slug
from kb_server.domain.credentials import Principal, can_manage
from kb_server.domain.errors import (
    CollectionNotFound,
    InvalidRequest,
    NotAllowed,
    UserNotFound,
)
from kb_server.domain.groups import Group
from kb_server.domain.users import User
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.users import UserRepository


class CollectionService:
    def __init__(
        self,
        *,
        collections: CollectionRepository,
        groups: GroupRepository,
        users: UserRepository,
    ) -> None:
        self._collections = collections
        self._groups = groups
        self._users = users

    def create_collection(
        self,
        admin: Principal,
        *,
        slug: str,
        title: str,
        description: str,
        group_ids: Sequence[uuid.UUID],
    ) -> CollectionDetails:
        if not admin.is_admin:
            raise NotAllowed("only an admin can create collections")
        groups = self._get_groups(group_ids)
        collection = self._collections.add(
            slug=normalize_slug(slug), title=title, description=description
        )
        self._collections.set_groups(collection.id, [group.id for group in groups])
        return CollectionDetails(collection=collection, groups=groups)

    def get_collection(
        self, principal: Principal, collection_id: uuid.UUID
    ) -> CollectionDetails:
        collection = self._get_visible(principal, collection_id)
        groups = self._collections.list_groups(collection_id)
        return CollectionDetails(collection=collection, groups=groups)

    def list_collections(self, principal: Principal) -> list[CollectionDetails]:
        if principal.user.role in ("manager", "admin"):
            collections = self._collections.list_all()
        else:
            collections = self._collections.list_visible_to(principal.user.id)
        groups = self._collections.groups_by_collection([c.id for c in collections])
        return [
            CollectionDetails(collection=c, groups=groups.get(c.id, []))
            for c in collections
        ]

    def update_collection(
        self,
        principal: Principal,
        collection_id: uuid.UUID,
        *,
        slug: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> CollectionDetails:
        slug = normalize_slug(slug) if slug is not None else None
        if not principal.is_admin:
            raise NotAllowed("only an admin can update collections")
        details = self.get_collection(principal, collection_id)
        c = details.collection
        new = replace(
            c,
            slug=slug if slug is not None else c.slug,
            title=title if title is not None else c.title,
            description=description if description is not None else c.description,
        )
        self._collections.update(
            collection_id,
            slug=new.slug,
            title=new.title,
            description=new.description,
        )
        return CollectionDetails(collection=new, groups=details.groups)

    def delete_collection(self, admin: Principal, collection_id: uuid.UUID) -> None:
        if not admin.is_admin:
            raise NotAllowed("only an admin can delete collections")
        self._collections.delete(collection_id)

    def set_groups(
        self,
        principal: Principal,
        collection_id: uuid.UUID,
        group_ids: Iterable[uuid.UUID],
    ) -> CollectionDetails:
        collection = self._get_visible(principal, collection_id)
        manager_ids = self._collections.list_manager_ids(collection_id)
        if not can_manage(principal, manager_ids):
            raise NotAllowed(
                "only an admin or a manager of this collection can change its groups"
            )
        groups = self._get_groups(group_ids)
        self._collections.set_groups(collection_id, [group.id for group in groups])
        return CollectionDetails(collection=collection, groups=groups)

    def list_managers(
        self, principal: Principal, collection_id: uuid.UUID
    ) -> list[User]:
        self._get_visible(principal, collection_id)
        if principal.user.role not in ("manager", "admin"):
            raise NotAllowed("only managers and admins can see a collection's managers")
        return self._collections.list_managers(collection_id)

    def add_manager(
        self, principal: Principal, *, collection_id: uuid.UUID, user_id: uuid.UUID
    ) -> None:
        if not principal.is_admin:
            raise NotAllowed("only an admin can add a manager to a collection")
        if not self._collections.get(collection_id):
            raise CollectionNotFound()
        user = self._users.get(user_id)
        if not user:
            raise UserNotFound()
        if user.role not in ("manager", "admin"):
            raise InvalidRequest("only managers and admins can manage a collection")
        self._collections.add_manager(collection_id=collection_id, user_id=user_id)

    def remove_manager(
        self, principal: Principal, *, collection_id: uuid.UUID, user_id: uuid.UUID
    ) -> None:
        if not principal.is_admin:
            raise NotAllowed("only an admin can add a manager to a collection")
        if not self._collections.get(collection_id):
            raise CollectionNotFound()
        user = self._users.get(user_id)
        if not user:
            raise UserNotFound()
        self._collections.remove_manager(collection_id=collection_id, user_id=user_id)

    def _get_visible(
        self, principal: Principal, collection_id: uuid.UUID
    ) -> Collection:
        collection = self._collections.get(collection_id)
        if collection is None or not self._can_see(principal, collection.id):
            raise CollectionNotFound()
        return collection

    def _can_see(self, principal: Principal, collection_id: uuid.UUID) -> bool:
        if principal.user.role in ("manager", "admin"):
            return True
        return self._collections.is_visible_to(collection_id, principal.user.id)

    def _get_groups(self, group_ids: Iterable[uuid.UUID]) -> list[Group]:
        wanted = set(group_ids)
        found = self._groups.get_many(wanted)
        missing = wanted - {group.id for group in found}
        if missing:
            raise InvalidRequest(
                f"unknown group ids: {', '.join(sorted(map(str, missing)))}"
            )
        return found
