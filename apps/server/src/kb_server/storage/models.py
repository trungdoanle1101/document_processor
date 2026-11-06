import uuid
from datetime import datetime
from tkinter import CASCADE

from sqlalchemy import DateTime, Enum, ForeignKey, String, Uuid, false, func, true
from sqlalchemy.orm import Mapped, mapped_column

from kb_server.domain.credentials import CREDENTIAL_KINDS, CredentialKind
from kb_server.domain.users import USER_ROLES, UserRole
from kb_server.storage.database import Base


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    display_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(
        Enum(
            *USER_ROLES,
            name="user_role",
            native_enum=False,
            create_constraint=True,
            length=20,
        )
    )
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class CredentialRow(Base):
    __tablename__ = "credentials"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[CredentialKind] = mapped_column(
        Enum(
            *CREDENTIAL_KINDS,
            name="credential_kind",
            native_enum=False,
            create_constraint=True,
            length=20,
        )
    )
    name: Mapped[str] = mapped_column(String(100))
    prefix: Mapped[str] = mapped_column(String(20))
    token_hash: Mapped[str] = mapped_column(String(255), unique=True)
    act_for_users: Mapped[bool] = mapped_column(default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class GroupRow(Base):
    __tablename__ = "groups"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UserGroupRow(Base):
    __tablename__ = "user_groups"
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete=CASCADE), primary_key=True
    )
    group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("groups.id", ondelete=CASCADE), primary_key=True
    )
