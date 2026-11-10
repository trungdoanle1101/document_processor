"""POST /api/v1/auth/login and /logout."""

from collections.abc import Callable
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from kb_server.domain.users import User
from kb_server.services.auth import AuthService
from kb_server.storage.repositories.credentials import CredentialRepository
from tests.support import PASSWORD, FakeClock, bearer, deactivate, sign_in

LOGIN = "/api/v1/auth/login"
LOGOUT = "/api/v1/auth/logout"


@pytest.fixture
def an(make_user: Callable[..., User]) -> User:
    return make_user("an@example.com")


def log_in(client: TestClient, email: str = "an@example.com") -> str:
    response = client.post(LOGIN, json={"email": email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    token: str = response.json()["token"]
    return token


# logging in


def test_login_returns_a_session_token(
    client: TestClient, an: User, clock: FakeClock
) -> None:
    response = client.post(
        LOGIN, json={"email": "an@example.com", "password": PASSWORD}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token"].startswith("kb_s_")
    assert body["token_type"] == "bearer"
    assert datetime.fromisoformat(body["expires_at"]) == clock.now + timedelta(hours=12)


def test_the_token_from_login_works(client: TestClient, an: User) -> None:
    token = log_in(client)

    # logging out needs a usable session, so it's the simplest proof for now
    assert client.get("/api/v1/me", headers=bearer(token)).status_code == 200


def test_login_ignores_the_case_of_the_email(client: TestClient, an: User) -> None:
    response = client.post(
        LOGIN, json={"email": "AN@Example.com", "password": PASSWORD}
    )

    assert response.status_code == 200


def test_each_login_is_a_separate_session(client: TestClient, an: User) -> None:
    first = log_in(client)
    second = log_in(client)

    assert first != second
    client.post(LOGOUT, headers=bearer(first))
    assert client.post(LOGOUT, headers=bearer(second)).status_code == 204


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("an@example.com", "wrong password"),
        ("nobody@example.com", PASSWORD),
    ],
    ids=["wrong password", "unknown email"],
)
def test_a_failed_login_is_401_without_saying_why(
    client: TestClient, an: User, email: str, password: str
) -> None:
    response = client.post(LOGIN, json={"email": email, "password": password})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json() == {
        "error": {
            "code": "login_failed",
            "message": "wrong email or password",
            "details": None,
        }
    }


def test_an_inactive_user_gets_the_same_401(
    client: TestClient, session: Session, an: User
) -> None:
    deactivate(session, an.id)

    response = client.post(
        LOGIN, json={"email": "an@example.com", "password": PASSWORD}
    )

    assert response.status_code == 401
    assert response.json()["error"]["message"] == "wrong email or password"


@pytest.mark.parametrize(
    "body",
    [
        {"email": "an@example.com"},
        {"password": PASSWORD},
        {"email": "not an email", "password": PASSWORD},
        {"email": "an@example.com", "password": PASSWORD, "remember": True},
    ],
    ids=["no password", "no email", "malformed email", "unknown field"],
)
def test_a_malformed_login_is_422(
    client: TestClient, an: User, body: dict[str, object]
) -> None:
    response = client.post(LOGIN, json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_a_failed_login_issues_no_session(
    client: TestClient, credentials: CredentialRepository, an: User
) -> None:
    client.post(LOGIN, json={"email": "an@example.com", "password": "wrong password"})

    assert credentials.list_for_user(an.id, kind="session") == []


# logging out


def test_logout_ends_the_session(client: TestClient, an: User) -> None:
    token = log_in(client)

    response = client.post(LOGOUT, headers=bearer(token))

    assert response.status_code == 204
    assert response.content == b""
    # the token no longer works, so a second logout is refused
    assert client.post(LOGOUT, headers=bearer(token)).status_code == 401


def test_logout_leaves_other_sessions_alone(client: TestClient, an: User) -> None:
    laptop = log_in(client)
    phone = log_in(client)

    client.post(LOGOUT, headers=bearer(laptop))

    assert client.post(LOGOUT, headers=bearer(phone)).status_code == 204


def test_logout_without_a_token_is_401(client: TestClient) -> None:
    response = client.post(LOGOUT)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "not_authenticated"


def test_an_api_key_cannot_log_out(
    client: TestClient, auth: AuthService, an: User
) -> None:
    _, session = sign_in(auth, "an@example.com")
    key = auth.create_key(session, name="script")

    response = client.post(LOGOUT, headers=bearer(key.token))

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "not_allowed"
    # the key still works afterwards
    assert auth.resolve(key.token) is not None


def test_an_expired_session_cannot_log_out(
    client: TestClient, an: User, clock: FakeClock
) -> None:
    token = log_in(client)

    clock.advance(timedelta(hours=12))

    assert client.post(LOGOUT, headers=bearer(token)).status_code == 401
