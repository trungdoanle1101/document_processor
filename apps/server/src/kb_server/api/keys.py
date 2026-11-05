import uuid
from datetime import datetime
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import AwareDatetime, BaseModel, ConfigDict, StringConstraints

from kb_server.api.deps import AdminPrincipal, AuthServiceDep, CurrentPrincipal

KeyName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]


class KeyCreate(BaseModel):
    name: KeyName
    expires_at: AwareDatetime | None = None
    model_config = ConfigDict(extra="forbid")


class KeyOut(BaseModel):
    id: uuid.UUID
    name: str
    prefix: str
    act_for_users: bool
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None
    model_config = ConfigDict(from_attributes=True)


class FrontEndkeyCreate(BaseModel):
    service_user_id: uuid.UUID
    name: KeyName
    expires_at: AwareDatetime | None = None
    model_config = ConfigDict(extra="forbid")


class KeyCreated(KeyOut):
    token: str


router = APIRouter(prefix="/keys", tags=["keys"])
admin_router = APIRouter(prefix="/admin/keys", tags=["admin", "keys"])


@router.post("", status_code=HTTPStatus.CREATED)
def create_key(
    body: KeyCreate, principal: CurrentPrincipal, auth_service: AuthServiceDep
) -> KeyCreated:
    issued = auth_service.create_key(
        principal, name=body.name, expires_at=body.expires_at
    )
    return KeyCreated(
        **KeyOut.model_validate(issued.credential).model_dump(),
        token=issued.token,
    )


@router.get("")
def list_keys(
    principal: CurrentPrincipal, auth_service: AuthServiceDep
) -> list[KeyOut]:
    credentials = auth_service.list_keys(principal)
    return [KeyOut.model_validate(credential) for credential in credentials]


@router.delete("/{key_id}", status_code=HTTPStatus.NO_CONTENT)
def revoke_key(
    key_id: uuid.UUID, principal: CurrentPrincipal, auth_service: AuthServiceDep
) -> None:
    auth_service.revoke_key(principal, key_id)


@admin_router.post("", status_code=HTTPStatus.CREATED)
def create_front_end_key(
    body: FrontEndkeyCreate, admin: AdminPrincipal, auth_service: AuthServiceDep
) -> KeyCreated:
    issued = auth_service.create_front_end_key(
        admin,
        service_user_id=body.service_user_id,
        name=body.name,
        expires_at=body.expires_at,
    )
    return KeyCreated(
        **KeyOut.model_validate(issued.credential).model_dump(),
        token=issued.token,
    )


@admin_router.get("")
def list_user_keys(
    admin: AdminPrincipal,
    auth_service: AuthServiceDep,
    user_id: Annotated[uuid.UUID, Query()],
) -> list[KeyOut]:
    credentials = auth_service.list_user_keys(admin, user_id)
    return [KeyOut.model_validate(credential) for credential in credentials]


@admin_router.delete("/{key_id}", status_code=HTTPStatus.NO_CONTENT)
def revoke_user_key(
    key_id: uuid.UUID, admin: AdminPrincipal, auth_service: AuthServiceDep
) -> None:
    auth_service.revoke_user_key(admin, key_id)
