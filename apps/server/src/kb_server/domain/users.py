import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, get_args

UserRole = Literal["reader", "manager", "admin"]
USER_ROLES = get_args(UserRole)


@dataclass(frozen=True, kw_only=True)
class User:
    id: uuid.UUID
    email: str
    display_name: str
    role: UserRole
    is_active: bool
    created_at: datetime


class EmailTaken(Exception):
    """A user with this email already exists"""


def normalize_email(email: str) -> str:
    return email.strip().lower()
