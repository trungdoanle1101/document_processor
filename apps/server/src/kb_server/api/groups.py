import uuid
from datetime import datetime
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, StringConstraints

from kb_server.api.deps import AdminPrincipal, GroupServiceDep
from kb_server.api.users import UserOut

router = APIRouter(prefix="/groups", tags=["groups"])

GroupName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, to_lower=True, pattern=r"^[a-z0-9][a-z0-0_-]{0,49}$"
    ),
]


class GroupCreate(BaseModel):
    name: GroupName

    model_config = ConfigDict(extra="forbid")


class GroupOut(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


@router.post("", status_code=HTTPStatus.CREATED)
def create_group(
    body: GroupCreate, principal: AdminPrincipal, group_service: GroupServiceDep
) -> GroupOut:
    return GroupOut.model_validate(group_service.create_group(principal, body.name))


@router.get("")
def list_groups(
    principal: AdminPrincipal, group_service: GroupServiceDep
) -> list[GroupOut]:
    return [
        GroupOut.model_validate(group) for group in group_service.list_groups(principal)
    ]


@router.delete("/{group_id}", status_code=HTTPStatus.NO_CONTENT)
def delete_group(
    group_id: uuid.UUID, principal: AdminPrincipal, group_service: GroupServiceDep
) -> None:
    group_service.delete_group(principal, group_id)


@router.get("/{group_id}/members")
def list_members(
    group_id: uuid.UUID, principal: AdminPrincipal, group_service: GroupServiceDep
) -> list[UserOut]:
    return [
        UserOut.model_validate(user)
        for user in group_service.list_members(principal, group_id)
    ]


@router.put("/{group_id}/members/{user_id}", status_code=HTTPStatus.NO_CONTENT)
def add_member(
    user_id: uuid.UUID,
    group_id: uuid.UUID,
    principal: AdminPrincipal,
    group_service: GroupServiceDep,
) -> None:
    group_service.add_member(principal, user_id=user_id, group_id=group_id)


@router.delete("/{group_id}/members/{user_id}", status_code=HTTPStatus.NO_CONTENT)
def remove_member(
    user_id: uuid.UUID,
    group_id: uuid.UUID,
    principal: AdminPrincipal,
    group_service: GroupServiceDep,
) -> None:
    group_service.remove_member(principal, user_id=user_id, group_id=group_id)
