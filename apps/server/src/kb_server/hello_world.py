import uuid
from dataclasses import dataclass
from datetime import datetime

from fastapi import FastAPI
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import DateTime, Enum, Uuid, String, func, MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

app = FastAPI()

type UserRoleLiteral = Literal["reader", "manager", "admin"]
USER_ROLES = get_args(UserRoleLiteral)

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

class UserCreate(BaseModel):
    email: EmailStr
    display_name: Annotated[str, Field(max_length=100)]
    password: Annotated[str, Field(min_length=12)]
    role: UserRoleLiteral

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str
    role: UserRoleLiteral
    is_active: bool
    created_at: datetime

class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    display_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(Enum(*USER_ROLES, name="user_role"))
    is_active: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


@app.get("/")
def create_app():
    return {"Hello": "World"}