"""/api/v1/groups and their members (admin only)."""

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kb_server.domain.groups import EVERYONE
from kb_server.domain.users import User
from kb_server.services.auth import AuthService
from tests.support import PASSWORD, bearer, sign_in

GROUPS = "/api/v1/groups"
USERS = "/api/v1/users"


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


def create_group(
    client: TestClient, headers: dict[str, str], name: str = "hr"
) -> dict[str, Any]:
    response = client.post(GROUPS, headers=headers, json={"name": name})
    assert response.status_code == 201, response.text
    group: dict[str, Any] = response.json()
    return group


def group_named(
    client: TestClient, headers: dict[str, str], name: str
) -> dict[str, Any]:
    (group,) = [
        g for g in client.get(GROUPS, headers=headers).json() if g["name"] == name
    ]
    found: dict[str, Any] = group
    return found


def member_emails(
    client: TestClient, headers: dict[str, str], group_id: str
) -> list[str]:
    response = client.get(f"{GROUPS}/{group_id}/members", headers=headers)
    assert response.status_code == 200, response.text
    return [user["email"] for user in response.json()]


# groups


def test_an_admin_creates_a_group(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.post(GROUPS, headers=admin_headers, json={"name": "  HR "})

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"id", "name", "created_at"}
    assert body["name"] == "hr"
    uuid.UUID(body["id"])


def test_groups_are_listed_by_name_with_everyone(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    create_group(client, admin_headers, "it")
    create_group(client, admin_headers, "hr")

    response = client.get(GROUPS, headers=admin_headers)

    assert response.status_code == 200
    assert [group["name"] for group in response.json()] == [EVERYONE, "hr", "it"]


def test_a_taken_group_name_is_409_whatever_its_case(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    create_group(client, admin_headers, "hr")

    response = client.post(GROUPS, headers=admin_headers, json={"name": "HR"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "group_name_taken"


@pytest.mark.parametrize("name", ["team1", "it-ops", "data_lake", "R2D2"])
def test_valid_group_names(
    client: TestClient, admin_headers: dict[str, str], name: str
) -> None:
    group = create_group(client, admin_headers, name)

    assert group["name"] == name.lower()


@pytest.mark.parametrize(
    "name",
    ["", "   ", "it ops", "-hr", "_hr", "hr!", "nhân-sự", "x" * 51],
    ids=[
        "empty",
        "blank",
        "space",
        "leading dash",
        "leading underscore",
        "punctuation",
        "accents",
        "too long",
    ],
)
def test_invalid_group_names_are_422(
    client: TestClient, admin_headers: dict[str, str], name: str
) -> None:
    response = client.post(GROUPS, headers=admin_headers, json={"name": name})

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "name"


def test_unknown_fields_are_refused(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.post(
        GROUPS, headers=admin_headers, json={"name": "hr", "description": "people"}
    )

    assert response.status_code == 422


def test_an_admin_deletes_a_group(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    group = create_group(client, admin_headers)

    response = client.delete(f"{GROUPS}/{group['id']}", headers=admin_headers)

    assert response.status_code == 204
    names = [g["name"] for g in client.get(GROUPS, headers=admin_headers).json()]
    assert names == [EVERYONE]
    again = client.delete(f"{GROUPS}/{group['id']}", headers=admin_headers)
    assert again.status_code == 404


def test_everyone_cannot_be_deleted(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    everyone = group_named(client, admin_headers, EVERYONE)

    response = client.delete(f"{GROUPS}/{everyone['id']}", headers=admin_headers)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "not_allowed"


def test_an_unknown_group_is_404(
    client: TestClient, an: User, admin_headers: dict[str, str]
) -> None:
    unknown = uuid.uuid4()

    for method, path in [
        ("DELETE", f"{GROUPS}/{unknown}"),
        ("GET", f"{GROUPS}/{unknown}/members"),
        ("PUT", f"{GROUPS}/{unknown}/members/{an.id}"),
        ("DELETE", f"{GROUPS}/{unknown}/members/{an.id}"),
    ]:
        response = client.request(method, path, headers=admin_headers)
        assert response.status_code == 404, (method, path)
        assert response.json()["error"]["code"] == "group_not_found"


def test_a_malformed_group_id_is_422(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.get(f"{GROUPS}/not-a-uuid/members", headers=admin_headers)

    assert response.status_code == 422


# members


def test_an_admin_adds_a_member(
    client: TestClient, an: User, admin_headers: dict[str, str]
) -> None:
    group = create_group(client, admin_headers)

    response = client.put(
        f"{GROUPS}/{group['id']}/members/{an.id}", headers=admin_headers
    )

    assert response.status_code == 204
    assert member_emails(client, admin_headers, group["id"]) == ["an@example.com"]


def test_the_members_list_shows_users(
    client: TestClient, an: User, admin_headers: dict[str, str]
) -> None:
    group = create_group(client, admin_headers)
    client.put(f"{GROUPS}/{group['id']}/members/{an.id}", headers=admin_headers)

    (member,) = client.get(
        f"{GROUPS}/{group['id']}/members", headers=admin_headers
    ).json()

    assert set(member) == {
        "id",
        "email",
        "display_name",
        "role",
        "is_active",
        "created_at",
    }


def test_adding_a_member_twice_is_204_both_times(
    client: TestClient, an: User, admin_headers: dict[str, str]
) -> None:
    group = create_group(client, admin_headers)
    path = f"{GROUPS}/{group['id']}/members/{an.id}"

    assert client.put(path, headers=admin_headers).status_code == 204
    assert client.put(path, headers=admin_headers).status_code == 204
    assert member_emails(client, admin_headers, group["id"]) == ["an@example.com"]


def test_removing_a_member_twice_is_204_both_times(
    client: TestClient, an: User, admin_headers: dict[str, str]
) -> None:
    group = create_group(client, admin_headers)
    path = f"{GROUPS}/{group['id']}/members/{an.id}"
    client.put(path, headers=admin_headers)

    assert client.delete(path, headers=admin_headers).status_code == 204
    assert client.delete(path, headers=admin_headers).status_code == 204
    assert member_emails(client, admin_headers, group["id"]) == []


def test_adding_an_unknown_user_is_404(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    group = create_group(client, admin_headers)

    response = client.put(
        f"{GROUPS}/{group['id']}/members/{uuid.uuid4()}", headers=admin_headers
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "user_not_found"


def test_a_user_created_through_the_api_joins_everyone(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    client.post(
        USERS,
        headers=admin_headers,
        json={
            "email": "binh@example.com",
            "display_name": "Binh",
            "password": PASSWORD,
            "role": "reader",
        },
    )
    everyone = group_named(client, admin_headers, EVERYONE)

    assert "binh@example.com" in member_emails(client, admin_headers, everyone["id"])


def test_no_one_can_be_removed_from_everyone(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    created = client.post(
        USERS,
        headers=admin_headers,
        json={
            "email": "binh@example.com",
            "display_name": "Binh",
            "password": PASSWORD,
            "role": "reader",
        },
    ).json()
    everyone = group_named(client, admin_headers, EVERYONE)

    response = client.delete(
        f"{GROUPS}/{everyone['id']}/members/{created['id']}", headers=admin_headers
    )

    assert response.status_code == 403
    assert "binh@example.com" in member_emails(client, admin_headers, everyone["id"])


def test_deleting_a_group_removes_it_but_not_its_members(
    client: TestClient, an: User, admin_headers: dict[str, str]
) -> None:
    group = create_group(client, admin_headers)
    client.put(f"{GROUPS}/{group['id']}/members/{an.id}", headers=admin_headers)

    client.delete(f"{GROUPS}/{group['id']}", headers=admin_headers)

    assert client.get(f"{USERS}/{an.id}", headers=admin_headers).status_code == 200


# who may


def test_group_routes_need_a_token(client: TestClient) -> None:
    assert client.get(GROUPS).status_code == 401
    assert client.post(GROUPS, json={"name": "hr"}).status_code == 401


def test_a_reader_cannot_use_group_routes(
    client: TestClient,
    an: User,
    an_headers: dict[str, str],
    admin_headers: dict[str, str],
) -> None:
    group = create_group(client, admin_headers)
    member = f"{GROUPS}/{group['id']}/members/{an.id}"

    for method, path, body in [
        ("POST", GROUPS, {"name": "it"}),
        ("GET", GROUPS, None),
        ("DELETE", f"{GROUPS}/{group['id']}", None),
        ("GET", f"{GROUPS}/{group['id']}/members", None),
        ("PUT", member, None),
        ("DELETE", member, None),
    ]:
        response = client.request(method, path, headers=an_headers, json=body)
        assert response.status_code == 403, (method, path)

    # nothing changed: the reader didn't add themselves
    assert member_emails(client, admin_headers, group["id"]) == []


def test_a_front_end_acting_for_an_admin_cannot_use_group_routes(
    client: TestClient,
    auth: AuthService,
    make_user: Callable[..., User],
    admin_headers: dict[str, str],
) -> None:
    service_user = make_user("openwebui@example.com")
    _, admin = sign_in(auth, "admin@example.com")
    key = auth.create_front_end_key(admin, service_user_id=service_user.id, name="owui")
    as_admin = {**bearer(key.token), "x-act-for-user": "admin@example.com"}

    assert client.get(GROUPS, headers=as_admin).status_code == 403
    assert client.post(GROUPS, headers=as_admin, json={"name": "it"}).status_code == 403
