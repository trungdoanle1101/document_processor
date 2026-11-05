import uuid
from datetime import datetime
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
    field_validator,
)

from kb_server.domain.users import UserRole, normalize_email

DisplayName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]


class PasswordChange(BaseModel):
    current_password: str
    new_password: Annotated[str, Field(min_length=12, max_length=128)]
    model_config = ConfigDict(extra="forbid")


class UserCreate(BaseModel):
    email: EmailStr
    display_name: DisplayName
    password: Annotated[str, Field(min_length=12, max_length=128)]
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
