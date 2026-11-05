import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, get_args

UserRole = Literal["reader", "manager", "admin"]
USER_ROLES = get_args(UserRole)

MIN_PASSWORD_LEN = 12


@dataclass(frozen=True, kw_only=True)
class User:
    id: uuid.UUID
    email: str
    display_name: str
    role: UserRole
    is_active: bool
    created_at: datetime


def normalize_email(email: str) -> str:
    return email.strip().lower()
