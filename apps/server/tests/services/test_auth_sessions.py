"""Logging in and out, and resolving tokens to principals."""

from collections.abc import Callable
from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from kb_server.domain.errors import NotAllowed
from kb_server.domain.users import User
from kb_server.services.auth import AuthService
from kb_server.storage.repositories.credentials import CredentialRepository
from tests.support import PASSWORD, FakeClock, deactivate, sign_in


def test_authenticate_returns_the_user_for_the_right_password(
    auth: AuthService, make_user: Callable[..., User]
) -> None:
    user = make_user("an@example.com")

    assert auth.authenticate("an@example.com", PASSWORD) == user


def test_authenticate_ignores_the_case_of_the_email(
    auth: AuthService, make_user: Callable[..., User]
) -> None:
    make_user("an@example.com")

    assert auth.authenticate("AN@Example.com", PASSWORD) is not None


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("an@example.com", "wrong password"),
        ("binh@example.com", PASSWORD),  # no such user
        ("binh@example.com", "not a real password"),  # the dummy hash's password
    ],
)
def test_authenticate_refuses_wrong_details(
    auth: AuthService, make_user: Callable[..., User], email: str, password: str
) -> None:
    make_user("an@example.com")

    assert auth.authenticate(email, password) is None


def test_an_inactive_user_cannot_log_in(
    auth: AuthService, session: Session, make_user: Callable[..., User]
) -> None:
    user = make_user("an@example.com")
    deactivate(session, user.id)

    assert auth.login("an@example.com", PASSWORD) is None


def test_login_issues_a_session_that_resolves_to_the_user(
    auth: AuthService, make_user: Callable[..., User], clock: FakeClock
) -> None:
    user = make_user("an@example.com")

    issued = auth.login("an@example.com", PASSWORD)

    assert issued is not None
    assert issued.token.startswith("kb_s_")
    assert issued.credential.kind == "session"
    assert issued.credential.expires_at == clock.now + timedelta(hours=12)
    principal = auth.resolve(issued.token)
    assert principal is not None
    assert principal.user == user
    assert principal.credential_id == issued.credential.id
    assert principal.credential_kind == "session"
    assert not principal.acting_for_others


def test_a_session_works_until_it_expires(
    auth: AuthService, make_user: Callable[..., User], clock: FakeClock
) -> None:
    make_user("an@example.com")
    token, _ = sign_in(auth, "an@example.com")

    clock.advance(timedelta(hours=12) - timedelta(seconds=1))
    assert auth.resolve(token) is not None

    clock.advance(timedelta(seconds=1))
    assert auth.resolve(token) is None


@pytest.mark.parametrize("token", ["", "not a token", "kb_s_unknown", "kb_k_unknown"])
def test_resolve_refuses_unknown_tokens(
    auth: AuthService, make_user: Callable[..., User], token: str
) -> None:
    make_user("an@example.com")
    sign_in(auth, "an@example.com")

    assert auth.resolve(token) is None


def test_a_session_stops_working_when_its_user_is_deactivated(
    auth: AuthService, session: Session, make_user: Callable[..., User]
) -> None:
    user = make_user("an@example.com")
    token, _ = sign_in(auth, "an@example.com")

    deactivate(session, user.id)

    assert auth.resolve(token) is None


def test_resolve_records_use_at_most_once_a_minute(
    auth: AuthService,
    credentials: CredentialRepository,
    make_user: Callable[..., User],
    clock: FakeClock,
) -> None:
    user = make_user("an@example.com")
    token, _ = sign_in(auth, "an@example.com")  # signing in resolves once
    first_use = clock.now

    def last_used() -> datetime | None:
        (session,) = credentials.list_for_user(user.id, kind="session")
        return session.last_used_at

    assert last_used() == first_use

    clock.advance(timedelta(seconds=30))
    auth.resolve(token)
    assert last_used() == first_use

    clock.advance(timedelta(seconds=31))
    auth.resolve(token)
    assert last_used() == clock.now


def test_logout_ends_the_session(
    auth: AuthService, make_user: Callable[..., User]
) -> None:
    make_user("an@example.com")
    token, principal = sign_in(auth, "an@example.com")

    auth.logout(principal)

    assert auth.resolve(token) is None


def test_logout_leaves_other_sessions_alone(
    auth: AuthService, make_user: Callable[..., User]
) -> None:
    make_user("an@example.com")
    _, laptop = sign_in(auth, "an@example.com")
    phone_token, _ = sign_in(auth, "an@example.com")

    auth.logout(laptop)

    assert auth.resolve(phone_token) is not None


def test_an_api_key_cannot_log_out(
    auth: AuthService, make_user: Callable[..., User]
) -> None:
    make_user("an@example.com")
    _, session = sign_in(auth, "an@example.com")
    key = auth.create_key(session, name="script")
    key_principal = auth.resolve(key.token)
    assert key_principal is not None

    with pytest.raises(NotAllowed):
        auth.logout(key_principal)
