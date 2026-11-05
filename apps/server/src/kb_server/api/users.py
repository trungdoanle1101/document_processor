import uuid
from datetime import datetime
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
    field_validator,
)

from kb_server.api.deps import AdminPrincipal, CurrentPrincipal, UserServiceDep
from kb_server.domain.users import MIN_PASSWORD_LEN, UserRole, normalize_email

DisplayName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]

Password = Field(min_length=MIN_PASSWORD_LEN, max_length=128)


class PasswordChange(BaseModel):
    current_password: str
    new_password: Annotated[str, Password]
    model_config = ConfigDict(extra="forbid")


class UserCreate(BaseModel):
    email: EmailStr
    display_name: DisplayName
    password: Annotated[str, Password]
    role: UserRole

    model_config = ConfigDict(extra="forbid")

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        return normalize_email(value)


class UserUpdate(BaseModel):
    display_name: DisplayName | None = None
    role: UserRole | None = None
    is_active: bool | None = None
    model_config = ConfigDict(extra="forbid")


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str
    role: UserRole
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


router = APIRouter(prefix="/users", tags=["users"])
me_router = APIRouter(prefix="/me", tags=["me"])


@router.post("", status_code=HTTPStatus.CREATED)
def create_user(
    body: UserCreate, principal: AdminPrincipal, user_service: UserServiceDep
) -> UserOut:
    new_user = user_service.create_user(
        principal,
        email=body.email,
        display_name=body.display_name,
        role=body.role,
        password=body.password,
    )

    return UserOut.model_validate(new_user)


@router.get("")
def list_users(
    principal: AdminPrincipal, user_service: UserServiceDep
) -> list[UserOut]:
    return [UserOut.model_validate(u) for u in user_service.list_users(principal)]


@router.get("/{user_id}")
def get_user(
    user_id: uuid.UUID, principal: AdminPrincipal, user_service: UserServiceDep
) -> UserOut:
    return UserOut.model_validate(user_service.get_user(principal, user_id))


@router.patch("/{user_id}")
def update_user(
    user_id: uuid.UUID,
    body: UserUpdate,
    principal: AdminPrincipal,
    user_service: UserServiceDep,
) -> UserOut:
    updated = user_service.update_user(
        principal,
        user_id,
        display_name=body.display_name,
        role=body.role,
        is_active=body.is_active,
    )
    return UserOut.model_validate(updated)


@me_router.get("")
def get_me(principal: CurrentPrincipal) -> UserOut:
    return UserOut.model_validate(principal.user)


@me_router.post("/password", status_code=HTTPStatus.NO_CONTENT)
def change_password(
    body: PasswordChange, principal: CurrentPrincipal, user_service: UserServiceDep
) -> None:
    user_service.change_password(
        principal, current=body.current_password, new=body.new_password
    )
