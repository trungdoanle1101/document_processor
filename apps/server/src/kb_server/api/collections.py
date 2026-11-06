import uuid
from datetime import datetime
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter
from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

from kb_server.api.deps import (
    AdminPrincipal,
    CollectionServiceDep,
    CurrentPrincipal,
)
from kb_server.api.groups import GroupOut
from kb_server.api.users import UserOut
from kb_server.domain.collections import CollectionDetails

Slug = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,49}$"
    ),
    AfterValidator(str.lower),
]

Title = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
]

Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=5000)]


class CollectionCreate(BaseModel):
    slug: Slug
    title: Title
    description: Description = ""
    group_ids: list[uuid.UUID]

    model_config = ConfigDict(extra="forbid")


class CollectionUpdate(BaseModel):
    slug: Slug | None = None
    title: Title | None = None
    description: Description | None = None

    model_config = ConfigDict(extra="forbid")


class CollectionGroupSet(BaseModel):
    group_ids: list[uuid.UUID]

    model_config = ConfigDict(extra="forbid")


class CollectionOut(BaseModel):
    id: uuid.UUID
    slug: str
    title: str
    description: str
    created_at: datetime
    groups: list[GroupOut]

    @classmethod
    def from_details(cls, details: CollectionDetails) -> "CollectionOut":
        c = details.collection
        return cls(
            id=c.id,
            slug=c.slug,
            title=c.title,
            description=c.description,
            created_at=c.created_at,
            groups=[GroupOut.model_validate(group) for group in details.groups],
        )


router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", status_code=HTTPStatus.CREATED)
def create_collection(
    body: CollectionCreate,
    principal: AdminPrincipal,
    collection_service: CollectionServiceDep,
) -> CollectionOut:
    details = collection_service.create_collection(
        principal,
        slug=body.slug,
        title=body.title,
        description=body.description,
        group_ids=body.group_ids,
    )
    return CollectionOut.from_details(details)


@router.get("")
def list_collections(
    principal: CurrentPrincipal, collection_service: CollectionServiceDep
) -> list[CollectionOut]:
    return [
        CollectionOut.from_details(details)
        for details in collection_service.list_collections(principal)
    ]


@router.get("/{collection_id}")
def get_collection(
    collection_id: uuid.UUID,
    principal: CurrentPrincipal,
    collection_service: CollectionServiceDep,
) -> CollectionOut:
    details = collection_service.get_collection(principal, collection_id)
    return CollectionOut.from_details(details)


@router.patch("/{collection_id}")
def update_collection(
    collection_id: uuid.UUID,
    body: CollectionUpdate,
    principal: AdminPrincipal,
    collection_service: CollectionServiceDep,
) -> CollectionOut:
    details = collection_service.update_collection(
        principal,
        collection_id,
        slug=body.slug,
        title=body.title,
        description=body.description,
    )
    return CollectionOut.from_details(details)


@router.delete("/{collection_id}", status_code=HTTPStatus.NO_CONTENT)
def delete_collection(
    collection_id: uuid.UUID,
    principal: AdminPrincipal,
    collection_service: CollectionServiceDep,
) -> None:
    collection_service.delete_collection(principal, collection_id)


@router.put("/{collection_id}/groups")
def set_groups(
    collection_id: uuid.UUID,
    body: CollectionGroupSet,
    principal: CurrentPrincipal,
    collection_service: CollectionServiceDep,
) -> CollectionOut:
    details = collection_service.set_groups(principal, collection_id, body.group_ids)
    return CollectionOut.from_details(details)


@router.get("/{collection_id}/managers")
def list_managers(
    collection_id: uuid.UUID,
    principal: CurrentPrincipal,
    collection_service: CollectionServiceDep,
) -> list[UserOut]:
    return [
        UserOut.model_validate(manager)
        for manager in collection_service.list_managers(principal, collection_id)
    ]


@router.put("/{collection_id}/managers/{user_id}", status_code=HTTPStatus.NO_CONTENT)
def add_manager(
    collection_id: uuid.UUID,
    user_id: uuid.UUID,
    principal: AdminPrincipal,
    collection_service: CollectionServiceDep,
) -> None:
    collection_service.add_manager(
        principal, collection_id=collection_id, user_id=user_id
    )


@router.delete("/{collection_id}/managers/{user_id}", status_code=HTTPStatus.NO_CONTENT)
def remove_manager(
    collection_id: uuid.UUID,
    user_id: uuid.UUID,
    principal: AdminPrincipal,
    collection_service: CollectionServiceDep,
) -> None:
    collection_service.remove_manager(
        principal, collection_id=collection_id, user_id=user_id
    )
