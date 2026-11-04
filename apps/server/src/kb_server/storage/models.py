import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, String, Uuid, func, true
from sqlalchemy.orm import Mapped, mapped_column

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
            length=10,
        )
    )
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
