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


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str
    role: UserRole
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
