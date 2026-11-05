"""/api/v1/keys (your own API keys) and /api/v1/admin/keys (front-end keys, any user's keys)."""

import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kb_server.domain.users import User
from kb_server.services.auth import AuthService
from tests.support import FakeClock, bearer, sign_in

KEYS = "/api/v1/keys"
ADMIN_KEYS = "/api/v1/admin/keys"
ME = "/api/v1/me"
KEY_FIELDS = {
    "id",
    "name",
    "prefix",
    "act_for_users",
    "created_at",
    "last_used_at",
    "expires_at",
    "revoked_at",
}


@pytest.fixture
def an(make_user: Callable[..., User]) -> User:
    return make_user("an@example.com")


@pytest.fixture
def an_headers(auth: AuthService, an: User) -> dict[str, str]:
    return bearer(sign_in(auth, "an@example.com")[0])


@pytest.fixture
def admin_headers(auth: AuthService, make_user: Callable[..., User]) -> dict[str, str]:
    make_user("admin@example.com", role="admin")
    return bearer(sign_in(auth, "admin@example.com")[0])


@pytest.fixture
def service_user(make_user: Callable[..., User]) -> User:
    return make_user("openwebui@example.com")


def create_key(
    client: TestClient, headers: dict[str, str], **body: object
) -> dict[str, Any]:
    response = client.post(KEYS, headers=headers, json={"name": "script", **body})
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


def create_front_end_key(
    client: TestClient, headers: dict[str, str], service_user: User
) -> dict[str, Any]:
    response = client.post(
        ADMIN_KEYS,
        headers=headers,
        json={"service_user_id": str(service_user.id), "name": "open webui"},
    )
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


# creating your own keys


def test_creating_a_key_returns_its_token_once(
    client: TestClient, an_headers: dict[str, str]
) -> None:
    response = client.post(KEYS, headers=an_headers, json={"name": "  script  "})

    assert response.status_code == 201
    body = response.json()
    assert set(body) == KEY_FIELDS | {"token"}
    assert body["token"].startswith("kb_k_")
    assert body["token"].startswith(body["prefix"])
    assert body["name"] == "script"
    assert body["act_for_users"] is False
    assert body["expires_at"] is None
    assert body["last_used_at"] is None
    assert body["revoked_at"] is None


def test_a_new_key_works_as_its_owner(
    client: TestClient, an_headers: dict[str, str]
) -> None:
    key = create_key(client, an_headers)

    response = client.get(ME, headers=bearer(key["token"]))

    assert response.json()["email"] == "an@example.com"


def test_a_key_can_expire(
    client: TestClient, an_headers: dict[str, str], clock: FakeClock
) -> None:
    expires_at = clock.now + timedelta(days=30)

    key = create_key(client, an_headers, expires_at=expires_at.isoformat())

    assert datetime.fromisoformat(key["expires_at"]) == expires_at
    clock.advance(timedelta(days=30))
    assert client.get(ME, headers=bearer(key["token"])).status_code == 401


def test_an_expiry_in_the_past_is_422(
    client: TestClient, an_headers: dict[str, str], clock: FakeClock
) -> None:
    response = client.post(
        KEYS,
        headers=an_headers,
        json={"name": "script", "expires_at": clock.now.isoformat()},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "body",
    [
        {"name": "script", "expires_at": "2027-01-01T00:00:00"},
        {"name": "   "},
        {},
        {"name": "script", "act_for_users": True},
    ],
    ids=["expiry without a timezone", "blank name", "no name", "unknown field"],
)
def test_an_invalid_key_request_is_422(
    client: TestClient, an_headers: dict[str, str], body: dict[str, object]
) -> None:
    response = client.post(KEYS, headers=an_headers, json=body)

    assert response.status_code == 422


def test_creating_a_key_needs_a_login_session(
    client: TestClient, an_headers: dict[str, str]
) -> None:
    key = create_key(client, an_headers)

    response = client.post(KEYS, headers=bearer(key["token"]), json={"name": "another"})

    assert response.status_code == 403


def test_creating_a_key_needs_a_token(client: TestClient) -> None:
    assert client.post(KEYS, json={"name": "script"}).status_code == 401


# listing and revoking your own keys


def test_the_list_shows_your_keys_without_their_tokens(
    client: TestClient,
    auth: AuthService,
    make_user: Callable[..., User],
    an_headers: dict[str, str],
) -> None:
    key = create_key(client, an_headers)
    make_user("binh@example.com")
    create_key(client, bearer(sign_in(auth, "binh@example.com")[0]), name="binh's")

    response = client.get(KEYS, headers=an_headers)

    assert response.status_code == 200
    (listed,) = response.json()  # not the login session, not binh's key
    assert listed["id"] == key["id"]
    assert set(listed) == KEY_FIELDS
    assert key["token"] not in response.text


def test_the_list_shows_when_a_key_was_last_used(
    client: TestClient, an_headers: dict[str, str], clock: FakeClock
) -> None:
    key = create_key(client, an_headers)
    client.get(ME, headers=bearer(key["token"]))

    (listed,) = client.get(KEYS, headers=an_headers).json()

    assert datetime.fromisoformat(listed["last_used_at"]) == clock.now


def test_a_revoked_key_stops_working_and_shows_as_revoked(
    client: TestClient, an_headers: dict[str, str]
) -> None:
    key = create_key(client, an_headers)

    response = client.delete(f"{KEYS}/{key['id']}", headers=an_headers)

    assert response.status_code == 204
    assert client.get(ME, headers=bearer(key["token"])).status_code == 401
    (listed,) = client.get(KEYS, headers=an_headers).json()
    assert listed["revoked_at"] is not None


def test_a_key_can_revoke_itself(
    client: TestClient, an_headers: dict[str, str]
) -> None:
    key = create_key(client, an_headers)

    response = client.delete(f"{KEYS}/{key['id']}", headers=bearer(key["token"]))

    assert response.status_code == 204
    assert client.get(ME, headers=bearer(key["token"])).status_code == 401


def test_revoking_someone_elses_key_is_404(
    client: TestClient,
    auth: AuthService,
    make_user: Callable[..., User],
    an_headers: dict[str, str],
) -> None:
    key = create_key(client, an_headers)
    make_user("binh@example.com")
    binh_headers = bearer(sign_in(auth, "binh@example.com")[0])

    response = client.delete(f"{KEYS}/{key['id']}", headers=binh_headers)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "credential_not_found"
    assert client.get(ME, headers=bearer(key["token"])).status_code == 200


def test_revoking_an_unknown_or_revoked_key_is_404(
    client: TestClient, an_headers: dict[str, str]
) -> None:
    key = create_key(client, an_headers)
    client.delete(f"{KEYS}/{key['id']}", headers=an_headers)

    assert client.delete(f"{KEYS}/{key['id']}", headers=an_headers).status_code == 404
    assert (
        client.delete(f"{KEYS}/{uuid.uuid4()}", headers=an_headers).status_code == 404
    )


def test_a_malformed_key_id_is_422(
    client: TestClient, an_headers: dict[str, str]
) -> None:
    assert client.delete(f"{KEYS}/not-a-uuid", headers=an_headers).status_code == 422


# front-end keys


def test_an_admin_creates_a_front_end_key_for_the_service_user(
    client: TestClient, admin_headers: dict[str, str], service_user: User
) -> None:
    key = create_front_end_key(client, admin_headers, service_user)

    assert set(key) == KEY_FIELDS | {"token"}
    assert key["act_for_users"] is True
    assert key["token"].startswith("kb_k_")


def test_a_front_end_key_acts_for_the_user_it_names(
    client: TestClient, an: User, admin_headers: dict[str, str], service_user: User
) -> None:
    key = create_front_end_key(client, admin_headers, service_user)

    response = client.get(
        ME, headers={**bearer(key["token"]), "x-act-for-user": "an@example.com"}
    )

    assert response.json()["email"] == "an@example.com"


def test_a_front_end_key_needs_a_known_service_user(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.post(
        ADMIN_KEYS,
        headers=admin_headers,
        json={"service_user_id": str(uuid.uuid4()), "name": "open webui"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_only_an_admin_creates_front_end_keys(
    client: TestClient, an_headers: dict[str, str], service_user: User
) -> None:
    response = client.post(
        ADMIN_KEYS,
        headers=an_headers,
        json={"service_user_id": str(service_user.id), "name": "open webui"},
    )

    assert response.status_code == 403


def test_a_front_end_key_cannot_create_another_even_for_an_admin(
    client: TestClient, admin_headers: dict[str, str], service_user: User
) -> None:
    key = create_front_end_key(client, admin_headers, service_user)
    as_admin = {**bearer(key["token"]), "x-act-for-user": "admin@example.com"}

    response = client.post(
        ADMIN_KEYS,
        headers=as_admin,
        json={"service_user_id": str(service_user.id), "name": "sneaky"},
    )

    assert response.status_code == 403


def test_a_front_end_key_cannot_manage_the_keys_of_the_user_it_acts_for(
    client: TestClient,
    an: User,
    an_headers: dict[str, str],
    admin_headers: dict[str, str],
    service_user: User,
) -> None:
    own_key = create_key(client, an_headers)
    front_end = create_front_end_key(client, admin_headers, service_user)
    as_an = {**bearer(front_end["token"]), "x-act-for-user": "an@example.com"}

    assert client.post(KEYS, headers=as_an, json={"name": "sneaky"}).status_code == 403
    assert client.get(KEYS, headers=as_an).status_code == 403
    assert client.delete(f"{KEYS}/{own_key['id']}", headers=as_an).status_code == 403


# admins managing other users' keys


def test_an_admin_lists_a_users_keys(
    client: TestClient,
    an: User,
    an_headers: dict[str, str],
    admin_headers: dict[str, str],
) -> None:
    key = create_key(client, an_headers)

    response = client.get(
        ADMIN_KEYS, headers=admin_headers, params={"user_id": str(an.id)}
    )

    assert response.status_code == 200
    assert [listed["id"] for listed in response.json()] == [key["id"]]
    assert key["token"] not in response.text


def test_an_admin_sees_the_service_users_front_end_keys(
    client: TestClient, admin_headers: dict[str, str], service_user: User
) -> None:
    key = create_front_end_key(client, admin_headers, service_user)

    response = client.get(
        ADMIN_KEYS, headers=admin_headers, params={"user_id": str(service_user.id)}
    )

    (listed,) = response.json()
    assert listed["id"] == key["id"]
    assert listed["act_for_users"] is True


def test_listing_the_keys_of_an_unknown_user_is_404(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.get(
        ADMIN_KEYS, headers=admin_headers, params={"user_id": str(uuid.uuid4())}
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "user_not_found"


def test_listing_keys_as_admin_needs_a_user_id(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.get(ADMIN_KEYS, headers=admin_headers)

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "user_id"


def test_an_admin_revokes_a_front_end_key(
    client: TestClient,
    an: User,
    admin_headers: dict[str, str],
    service_user: User,
) -> None:
    key = create_front_end_key(client, admin_headers, service_user)
    as_an = {**bearer(key["token"]), "x-act-for-user": "an@example.com"}

    response = client.delete(f"{ADMIN_KEYS}/{key['id']}", headers=admin_headers)

    assert response.status_code == 204
    assert client.get(ME, headers=as_an).status_code == 401
    again = client.delete(f"{ADMIN_KEYS}/{key['id']}", headers=admin_headers)
    assert again.status_code == 404


def test_an_admin_revokes_another_users_key(
    client: TestClient, an_headers: dict[str, str], admin_headers: dict[str, str]
) -> None:
    key = create_key(client, an_headers)

    response = client.delete(f"{ADMIN_KEYS}/{key['id']}", headers=admin_headers)

    assert response.status_code == 204
    assert client.get(ME, headers=bearer(key["token"])).status_code == 401


def test_an_admin_cannot_end_a_login_session_through_the_key_route(
    client: TestClient,
    auth: AuthService,
    an: User,
    an_headers: dict[str, str],
    admin_headers: dict[str, str],
) -> None:
    session = auth.resolve(an_headers["Authorization"].removeprefix("Bearer "))
    assert session is not None

    response = client.delete(
        f"{ADMIN_KEYS}/{session.credential_id}", headers=admin_headers
    )

    assert response.status_code == 404
    assert client.get(ME, headers=an_headers).status_code == 200


def test_only_an_admin_uses_the_admin_key_routes(
    client: TestClient, an: User, an_headers: dict[str, str]
) -> None:
    key = create_key(client, an_headers)

    listing = client.get(ADMIN_KEYS, headers=an_headers, params={"user_id": str(an.id)})
    revoking = client.delete(f"{ADMIN_KEYS}/{key['id']}", headers=an_headers)

    assert listing.status_code == 403
    assert revoking.status_code == 403
    assert client.get(ME, headers=bearer(key["token"])).status_code == 200
