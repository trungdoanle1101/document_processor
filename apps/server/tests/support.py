"""Helpers for the tests. Fixtures live in conftest.py."""

import uuid
from datetime import datetime, timedelta

from sqlalchemy import update
from sqlalchemy.orm import Session

from kb_server.domain.credentials import Principal
from kb_server.services.auth import AuthService
from kb_server.storage.models import UserRow

PASSWORD = "correct horse battery"


class FakeClock:
    """A clock that only moves when told to."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, by: timedelta) -> None:
        self.now += by


def sign_in(auth: AuthService, email: str) -> tuple[str, Principal]:
    """Log in as email (whose password is PASSWORD): the session token and its principal."""
    issued = auth.login(email, PASSWORD)
    assert issued is not None, f"could not log in as {email}"
    principal = auth.resolve(issued.token)
    assert principal is not None
    return issued.token, principal


def deactivate(session: Session, user_id: uuid.UUID) -> None:
    session.execute(
        update(UserRow).where(UserRow.id == user_id).values(is_active=False)
    )


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
