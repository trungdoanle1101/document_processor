"""/api/v1/users (admin only) and /api/v1/me."""

import uuid
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from kb_server.domain.users import User
from kb_server.services.auth import AuthService
from tests.support import PASSWORD, bearer, sign_in

USERS = "/api/v1/users"
ME = "/api/v1/me"
LOGIN = "/api/v1/auth/login"
NEW_PASSWORD = "another long password"


@pytest.fixture
def admin_headers(auth: AuthService, make_user: Callable[..., User]) -> dict[str, str]:
    make_user("admin@example.com", role="admin")
    return bearer(sign_in(auth, "admin@example.com")[0])


@pytest.fixture
def an(make_user: Callable[..., User]) -> User:
    return make_user("an@example.com")


@pytest.fixture
def an_headers(auth: AuthService, an: User) -> dict[str, str]:
    return bearer(sign_in(auth, "an@example.com")[0])


@pytest.fixture
def front_end_headers(
    auth: AuthService, make_user: Callable[..., User], admin_headers: dict[str, str]
) -> Callable[[str], dict[str, str]]:
    """front_end_headers("an@example.com") -> headers of Open WebUI acting for that user."""
    service_user = make_user("openwebui@example.com")
    _, admin = sign_in(auth, "admin@example.com")
    key = auth.create_front_end_key(
        admin, service_user_id=service_user.id, name="open webui"
    )
    return lambda email: {**bearer(key.token), "x-act-for-user": email}


def new_user(**overrides: object) -> dict[str, object]:
    return {
        "email": "binh@example.com",
        "display_name": "Binh",
        "password": PASSWORD,
        "role": "reader",
        **overrides,
    }


def error_code(response_json: dict[str, dict[str, str]]) -> str:
    return response_json["error"]["code"]


# creating users


def test_an_admin_creates_a_user(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.post(
        USERS,
        headers=admin_headers,
        json=new_user(
            email="Binh@Example.com", display_name="  Binh  ", role="manager"
        ),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "binh@example.com"
    assert body["display_name"] == "Binh"
    assert body["role"] == "manager"
    assert body["is_active"] is True
    uuid.UUID(body["id"])
    assert set(body) == {
        "id",
        "email",
        "display_name",
        "role",
        "is_active",
        "created_at",
    }


def test_a_created_user_can_log_in(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    client.post(USERS, headers=admin_headers, json=new_user())

    response = client.post(
        LOGIN, json={"email": "binh@example.com", "password": PASSWORD}
    )

    assert response.status_code == 200


def test_an_admin_can_create_users_with_their_own_api_key(
    client: TestClient, auth: AuthService, admin_headers: dict[str, str]
) -> None:
    _, admin = sign_in(auth, "admin@example.com")
    key = auth.create_key(admin, name="provisioning")

    response = client.post(USERS, headers=bearer(key.token), json=new_user())

    assert response.status_code == 201


def test_creating_a_user_needs_a_token(client: TestClient) -> None:
    response = client.post(USERS, json=new_user())

    assert response.status_code == 401


def test_a_reader_cannot_create_users(
    client: TestClient, an_headers: dict[str, str]
) -> None:
    response = client.post(USERS, headers=an_headers, json=new_user())

    assert response.status_code == 403
    assert error_code(response.json()) == "not_allowed"


def test_a_front_end_key_cannot_create_users_even_for_an_admin(
    client: TestClient, front_end_headers: Callable[[str], dict[str, str]]
) -> None:
    response = client.post(
        USERS, headers=front_end_headers("admin@example.com"), json=new_user()
    )

    assert response.status_code == 403


def test_a_taken_email_is_409_whatever_its_case(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    client.post(USERS, headers=admin_headers, json=new_user())

    response = client.post(
        USERS, headers=admin_headers, json=new_user(email="BINH@example.com")
    )

    assert response.status_code == 409
    assert error_code(response.json()) == "email_taken"


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"password": "too short"}, "password"),
        ({"email": "not an email"}, "email"),
        ({"role": "owner"}, "role"),
        ({"display_name": "   "}, "display_name"),
    ],
    ids=["short password", "malformed email", "unknown role", "blank name"],
)
def test_an_invalid_new_user_is_422_naming_the_field(
    client: TestClient,
    admin_headers: dict[str, str],
    overrides: dict[str, object],
    field: str,
) -> None:
    response = client.post(USERS, headers=admin_headers, json=new_user(**overrides))

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert [detail["field"] for detail in error["details"]] == [field]


def test_unknown_fields_are_refused(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.post(USERS, headers=admin_headers, json=new_user(is_active=False))

    assert response.status_code == 422


# reading users


def test_an_admin_lists_users_by_email(
    client: TestClient,
    make_user: Callable[..., User],
    admin_headers: dict[str, str],
) -> None:
    make_user("chi@example.com")
    make_user("binh@example.com")

    response = client.get(USERS, headers=admin_headers)

    assert response.status_code == 200
    assert [user["email"] for user in response.json()] == [
        "admin@example.com",
        "binh@example.com",
        "chi@example.com",
    ]


def test_an_admin_gets_a_user_by_id(
    client: TestClient, an: User, admin_headers: dict[str, str]
) -> None:
    response = client.get(f"{USERS}/{an.id}", headers=admin_headers)

    assert response.status_code == 200
    assert response.json()["email"] == "an@example.com"


def test_an_unknown_user_is_404(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.get(f"{USERS}/{uuid.uuid4()}", headers=admin_headers)

    assert response.status_code == 404
    assert error_code(response.json()) == "user_not_found"


def test_a_malformed_id_is_422(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.get(f"{USERS}/not-a-uuid", headers=admin_headers)

    assert response.status_code == 422


def test_a_reader_cannot_read_users(
    client: TestClient, an: User, an_headers: dict[str, str]
) -> None:
    assert client.get(USERS, headers=an_headers).status_code == 403
    assert client.get(f"{USERS}/{an.id}", headers=an_headers).status_code == 403


# updating users


def test_an_update_changes_only_the_given_fields_and_is_saved(
    client: TestClient, an: User, admin_headers: dict[str, str]
) -> None:
    response = client.patch(
        f"{USERS}/{an.id}", headers=admin_headers, json={"role": "manager"}
    )

    assert response.status_code == 200
    assert response.json()["role"] == "manager"
    assert response.json()["display_name"] == an.display_name
    saved = client.get(f"{USERS}/{an.id}", headers=admin_headers).json()
    assert saved == response.json()


def test_an_empty_update_changes_nothing(
    client: TestClient, an: User, admin_headers: dict[str, str]
) -> None:
    before = client.get(f"{USERS}/{an.id}", headers=admin_headers).json()

    response = client.patch(f"{USERS}/{an.id}", headers=admin_headers, json={})

    assert response.status_code == 200
    assert response.json() == before


def test_a_deactivated_user_is_signed_out_and_cannot_log_in(
    client: TestClient,
    an: User,
    an_headers: dict[str, str],
    admin_headers: dict[str, str],
) -> None:
    response = client.patch(
        f"{USERS}/{an.id}", headers=admin_headers, json={"is_active": False}
    )

    assert response.status_code == 200
    assert response.json()["is_active"] is False
    assert client.get(ME, headers=an_headers).status_code == 401
    login = client.post(LOGIN, json={"email": "an@example.com", "password": PASSWORD})
    assert login.status_code == 401


def test_an_admin_cannot_demote_or_deactivate_themselves(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    me = client.get(ME, headers=admin_headers).json()

    for change in ({"role": "reader"}, {"is_active": False}):
        response = client.patch(
            f"{USERS}/{me['id']}", headers=admin_headers, json=change
        )
        assert response.status_code == 403

    assert client.get(ME, headers=admin_headers).json() == me


def test_an_admin_may_send_back_their_own_unchanged_user(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    me = client.get(ME, headers=admin_headers).json()

    response = client.patch(
        f"{USERS}/{me['id']}",
        headers=admin_headers,
        json={"display_name": "Boss", "role": "admin", "is_active": True},
    )

    assert response.status_code == 200
    assert response.json()["display_name"] == "Boss"


def test_updating_an_unknown_user_is_404(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.patch(
        f"{USERS}/{uuid.uuid4()}",
        headers=admin_headers,
        json={"display_name": "Nobody"},
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    "body",
    [{"email": "new@example.com"}, {"role": "owner"}, {"display_name": ""}],
    ids=["email can't change", "unknown role", "blank name"],
)
def test_an_invalid_update_is_422(
    client: TestClient, an: User, admin_headers: dict[str, str], body: dict[str, object]
) -> None:
    response = client.patch(f"{USERS}/{an.id}", headers=admin_headers, json=body)

    assert response.status_code == 422


def test_a_reader_cannot_update_users_or_promote_themselves(
    client: TestClient, an: User, an_headers: dict[str, str]
) -> None:
    response = client.patch(
        f"{USERS}/{an.id}", headers=an_headers, json={"role": "admin"}
    )

    assert response.status_code == 403


# /me


def test_me_shows_the_signed_in_user(
    client: TestClient, an: User, an_headers: dict[str, str]
) -> None:
    response = client.get(ME, headers=an_headers)

    assert response.status_code == 200
    assert response.json()["id"] == str(an.id)
    assert response.json()["email"] == "an@example.com"
    assert "password_hash" not in response.json()


def test_me_works_with_an_api_key(
    client: TestClient, auth: AuthService, an: User
) -> None:
    _, session = sign_in(auth, "an@example.com")
    key = auth.create_key(session, name="script")

    response = client.get(ME, headers=bearer(key.token))

    assert response.json()["email"] == "an@example.com"


def test_me_through_a_front_end_shows_the_user_it_acts_for(
    client: TestClient,
    an: User,
    front_end_headers: Callable[[str], dict[str, str]],
) -> None:
    response = client.get(ME, headers=front_end_headers("an@example.com"))

    assert response.status_code == 200
    assert response.json()["email"] == "an@example.com"


def test_me_needs_a_token(client: TestClient) -> None:
    assert client.get(ME).status_code == 401


# changing your password


def change_password(
    client: TestClient, headers: dict[str, str], current: str = PASSWORD
) -> int:
    response = client.post(
        f"{ME}/password",
        headers=headers,
        json={"current_password": current, "new_password": NEW_PASSWORD},
    )
    return int(response.status_code)


def log_in_status(client: TestClient, password: str) -> int:
    response = client.post(
        LOGIN, json={"email": "an@example.com", "password": password}
    )
    return int(response.status_code)


def test_changing_the_password_replaces_the_old_one(
    client: TestClient, an_headers: dict[str, str]
) -> None:
    assert change_password(client, an_headers) == 204

    assert log_in_status(client, PASSWORD) == 401
    assert log_in_status(client, NEW_PASSWORD) == 200


def test_changing_the_password_signs_out_only_the_other_sessions(
    client: TestClient, auth: AuthService, an: User
) -> None:
    phone = bearer(sign_in(auth, "an@example.com")[0])
    laptop = bearer(sign_in(auth, "an@example.com")[0])

    change_password(client, laptop)

    assert client.get(ME, headers=laptop).status_code == 200
    assert client.get(ME, headers=phone).status_code == 401


def test_a_wrong_current_password_is_422(
    client: TestClient, an_headers: dict[str, str]
) -> None:
    assert change_password(client, an_headers, current="wrong password") == 422

    assert log_in_status(client, PASSWORD) == 200


def test_a_short_new_password_is_422(
    client: TestClient, an_headers: dict[str, str]
) -> None:
    response = client.post(
        f"{ME}/password",
        headers=an_headers,
        json={"current_password": PASSWORD, "new_password": "short"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "new_password"


def test_an_api_key_cannot_change_the_password(
    client: TestClient, auth: AuthService, an: User
) -> None:
    _, session = sign_in(auth, "an@example.com")
    key = auth.create_key(session, name="script")

    assert change_password(client, bearer(key.token)) == 403


def test_a_front_end_key_cannot_change_the_password(
    client: TestClient, an: User, front_end_headers: Callable[[str], dict[str, str]]
) -> None:
    assert change_password(client, front_end_headers("an@example.com")) == 403
