"""/api/v1/collections, their default groups and their managers."""

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kb_server.domain.groups import Group
from kb_server.domain.users import User, UserRole
from kb_server.services.auth import AuthService
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.users import UserRepository
from tests.support import bearer, sign_in

COLLECTIONS = "/api/v1/collections"
GROUPS = "/api/v1/groups"
COLLECTION_FIELDS = {"id", "slug", "title", "description", "created_at", "groups"}


@pytest.fixture
def headers_for(
    auth: AuthService, make_user: Callable[..., User]
) -> Callable[[str, UserRole], dict[str, str]]:
    """headers_for("an@example.com", "reader") -> a new user's login headers."""

    def make(email: str, role: UserRole) -> dict[str, str]:
        make_user(email, role=role)
        return bearer(sign_in(auth, email)[0])

    return make


@pytest.fixture
def admin_headers(
    headers_for: Callable[[str, UserRole], dict[str, str]],
) -> dict[str, str]:
    return headers_for("admin@example.com", "admin")


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def it(groups: GroupRepository) -> Group:
    return groups.add("it")


def new_collection(**overrides: object) -> dict[str, object]:
    return {"slug": "hr-docs", "title": "HR documents", "group_ids": [], **overrides}


def create(
    client: TestClient, headers: dict[str, str], **overrides: object
) -> dict[str, Any]:
    response = client.post(
        COLLECTIONS, headers=headers, json=new_collection(**overrides)
    )
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


@pytest.fixture
def hr_docs(
    client: TestClient, admin_headers: dict[str, str], hr: Group
) -> dict[str, Any]:
    return create(client, admin_headers, group_ids=[str(hr.id)])


def user_id(users: UserRepository, email: str) -> uuid.UUID:
    user = users.get_by_email(email)
    assert user is not None
    return user.id


def error_code(response: Any) -> str:
    code: str = response.json()["error"]["code"]
    return code


# creating


def test_an_admin_creates_a_collection(
    client: TestClient, admin_headers: dict[str, str], hr: Group, it: Group
) -> None:
    response = client.post(
        COLLECTIONS,
        headers=admin_headers,
        json={
            "slug": " HR-Docs ",
            "title": "  Quy định nhân sự ",
            "description": "Nội quy công ty",
            "group_ids": [str(it.id), str(hr.id)],
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert set(body) == COLLECTION_FIELDS
    assert body["slug"] == "hr-docs"
    assert body["title"] == "Quy định nhân sự"
    assert body["description"] == "Nội quy công ty"
    assert [group["name"] for group in body["groups"]] == ["hr", "it"]


def test_the_description_is_optional(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    assert create(client, admin_headers)["description"] == ""


def test_a_long_title_fits(client: TestClient, admin_headers: dict[str, str]) -> None:
    assert len(create(client, admin_headers, title="T" * 200)["title"]) == 200


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"slug": "x", "title": "X"}, "group_ids"),
        (new_collection(slug="hr docs"), "slug"),
        (new_collection(slug="-hr"), "slug"),
        (new_collection(title="   "), "title"),
        (new_collection(title="T" * 201), "title"),
        (new_collection(group_ids=["not-a-uuid"]), "group_ids.0"),
    ],
    ids=[
        "no group_ids",
        "space in slug",
        "leading dash",
        "blank title",
        "title too long",
        "malformed group id",
    ],
)
def test_an_invalid_collection_is_422(
    client: TestClient,
    admin_headers: dict[str, str],
    body: dict[str, object],
    field: str,
) -> None:
    response = client.post(COLLECTIONS, headers=admin_headers, json=body)

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == field


def test_unknown_group_ids_are_422_and_create_nothing(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.post(
        COLLECTIONS,
        headers=admin_headers,
        json=new_collection(group_ids=[str(uuid.uuid4())]),
    )

    assert response.status_code == 422
    assert error_code(response) == "invalid_request"
    assert client.get(COLLECTIONS, headers=admin_headers).json() == []


def test_a_taken_slug_is_409(
    client: TestClient, admin_headers: dict[str, str], hr_docs: dict[str, Any]
) -> None:
    response = client.post(
        COLLECTIONS, headers=admin_headers, json=new_collection(slug="HR-Docs")
    )

    assert response.status_code == 409
    assert error_code(response) == "collection_slug_taken"


def test_only_an_admin_creates_collections(
    client: TestClient, headers_for: Callable[[str, UserRole], dict[str, str]]
) -> None:
    manager = headers_for("manager@example.com", "manager")

    assert (
        client.post(COLLECTIONS, headers=manager, json=new_collection()).status_code
        == 403
    )
    assert client.post(COLLECTIONS, json=new_collection()).status_code == 401


# seeing


def test_admins_and_managers_see_every_collection(
    client: TestClient,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    admin_headers: dict[str, str],
    hr_docs: dict[str, Any],
) -> None:
    create(client, admin_headers, slug="private")
    manager = headers_for("manager@example.com", "manager")

    for headers in (admin_headers, manager):
        slugs = [c["slug"] for c in client.get(COLLECTIONS, headers=headers).json()]
        assert slugs == ["hr-docs", "private"]


def test_a_reader_sees_a_collection_once_they_share_a_group(
    client: TestClient,
    groups: GroupRepository,
    users: UserRepository,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    hr: Group,
    hr_docs: dict[str, Any],
) -> None:
    reader = headers_for("an@example.com", "reader")
    one = f"{COLLECTIONS}/{hr_docs['id']}"
    assert client.get(COLLECTIONS, headers=reader).json() == []
    hidden = client.get(one, headers=reader)
    assert hidden.status_code == 404
    assert error_code(hidden) == "collection_not_found"

    groups.add_member(user_id=user_id(users, "an@example.com"), group_id=hr.id)

    assert [c["slug"] for c in client.get(COLLECTIONS, headers=reader).json()] == [
        "hr-docs"
    ]
    assert client.get(one, headers=reader).json() == hr_docs


def test_an_unknown_collection_is_404(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    response = client.get(f"{COLLECTIONS}/{uuid.uuid4()}", headers=admin_headers)

    assert response.status_code == 404


# updating and deleting


def test_a_patch_changes_only_the_given_fields(
    client: TestClient, admin_headers: dict[str, str], hr_docs: dict[str, Any]
) -> None:
    one = f"{COLLECTIONS}/{hr_docs['id']}"

    response = client.patch(one, headers=admin_headers, json={"title": "Nhân sự"})

    assert response.status_code == 200
    assert response.json() == {**hr_docs, "title": "Nhân sự"}
    assert client.get(one, headers=admin_headers).json() == response.json()


def test_a_patch_to_a_taken_slug_is_409_and_changes_nothing(
    client: TestClient, admin_headers: dict[str, str], hr_docs: dict[str, Any]
) -> None:
    create(client, admin_headers, slug="it-docs")
    one = f"{COLLECTIONS}/{hr_docs['id']}"

    response = client.patch(one, headers=admin_headers, json={"slug": "IT-docs"})

    assert response.status_code == 409
    assert client.get(one, headers=admin_headers).json()["slug"] == "hr-docs"


@pytest.mark.parametrize(
    "body",
    [{"group_ids": []}, {"slug": "a b"}, {"title": ""}],
    ids=["groups have their own route", "bad slug", "blank title"],
)
def test_an_invalid_patch_is_422(
    client: TestClient,
    admin_headers: dict[str, str],
    hr_docs: dict[str, Any],
    body: dict[str, object],
) -> None:
    response = client.patch(
        f"{COLLECTIONS}/{hr_docs['id']}", headers=admin_headers, json=body
    )

    assert response.status_code == 422


def test_an_admin_deletes_a_collection(
    client: TestClient, admin_headers: dict[str, str], hr_docs: dict[str, Any]
) -> None:
    one = f"{COLLECTIONS}/{hr_docs['id']}"

    response = client.delete(one, headers=admin_headers)

    assert response.status_code == 204
    assert response.content == b""
    assert client.get(one, headers=admin_headers).status_code == 404
    assert client.delete(one, headers=admin_headers).status_code == 404


def test_only_an_admin_updates_or_deletes(
    client: TestClient,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    hr_docs: dict[str, Any],
) -> None:
    manager = headers_for("manager@example.com", "manager")
    one = f"{COLLECTIONS}/{hr_docs['id']}"

    assert client.patch(one, headers=manager, json={"title": "Mine"}).status_code == 403
    assert client.delete(one, headers=manager).status_code == 403


# default groups


def test_an_admin_replaces_the_default_groups(
    client: TestClient,
    admin_headers: dict[str, str],
    it: Group,
    hr_docs: dict[str, Any],
) -> None:
    response = client.put(
        f"{COLLECTIONS}/{hr_docs['id']}/groups",
        headers=admin_headers,
        json={"group_ids": [str(it.id), str(it.id)]},
    )

    assert response.status_code == 200
    assert [group["name"] for group in response.json()["groups"]] == ["it"]


def test_a_manager_of_the_collection_sets_its_groups(
    client: TestClient,
    users: UserRepository,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    admin_headers: dict[str, str],
    hr_docs: dict[str, Any],
) -> None:
    manager = headers_for("manager@example.com", "manager")
    groups_route = f"{COLLECTIONS}/{hr_docs['id']}/groups"
    assert (
        client.put(groups_route, headers=manager, json={"group_ids": []}).status_code
        == 403
    )

    manager_id = user_id(users, "manager@example.com")
    client.put(
        f"{COLLECTIONS}/{hr_docs['id']}/managers/{manager_id}", headers=admin_headers
    )

    response = client.put(groups_route, headers=manager, json={"group_ids": []})
    assert response.status_code == 200
    assert response.json()["groups"] == []


def test_set_groups_refuses_unknown_groups_and_unknown_fields(
    client: TestClient, admin_headers: dict[str, str], hr_docs: dict[str, Any]
) -> None:
    groups_route = f"{COLLECTIONS}/{hr_docs['id']}/groups"

    unknown = client.put(
        groups_route, headers=admin_headers, json={"group_ids": [str(uuid.uuid4())]}
    )
    extra = client.put(
        groups_route, headers=admin_headers, json={"group_ids": [], "replace": True}
    )

    assert unknown.status_code == 422
    assert extra.status_code == 422
    assert (
        client.get(f"{COLLECTIONS}/{hr_docs['id']}", headers=admin_headers).json()
        == hr_docs
    )


def test_set_groups_on_a_hidden_collection_is_404(
    client: TestClient,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    hr_docs: dict[str, Any],
) -> None:
    reader = headers_for("an@example.com", "reader")

    response = client.put(
        f"{COLLECTIONS}/{hr_docs['id']}/groups", headers=reader, json={"group_ids": []}
    )

    assert response.status_code == 404


def test_a_group_used_by_a_collection_cannot_be_deleted(
    client: TestClient,
    admin_headers: dict[str, str],
    hr: Group,
    hr_docs: dict[str, Any],
) -> None:
    response = client.delete(f"{GROUPS}/{hr.id}", headers=admin_headers)

    assert response.status_code == 409
    assert error_code(response) == "group_in_use"

    client.put(
        f"{COLLECTIONS}/{hr_docs['id']}/groups",
        headers=admin_headers,
        json={"group_ids": []},
    )
    assert client.delete(f"{GROUPS}/{hr.id}", headers=admin_headers).status_code == 204


# managers


def test_an_admin_adds_and_removes_managers(
    client: TestClient,
    make_user: Callable[..., User],
    admin_headers: dict[str, str],
    hr_docs: dict[str, Any],
) -> None:
    binh = make_user("binh@example.com", role="manager")
    chi = make_user("chi@example.com", role="admin")
    managers = f"{COLLECTIONS}/{hr_docs['id']}/managers"

    for user in (chi, binh, binh):
        response = client.put(f"{managers}/{user.id}", headers=admin_headers)
        assert response.status_code == 204
        assert response.content == b""
    listed = client.get(managers, headers=admin_headers).json()
    assert [user["email"] for user in listed] == ["binh@example.com", "chi@example.com"]

    for _ in range(2):
        assert (
            client.delete(f"{managers}/{binh.id}", headers=admin_headers).status_code
            == 204
        )
    listed = client.get(managers, headers=admin_headers).json()
    assert [user["email"] for user in listed] == ["chi@example.com"]


def test_a_reader_cannot_be_made_a_manager(
    client: TestClient,
    make_user: Callable[..., User],
    admin_headers: dict[str, str],
    hr_docs: dict[str, Any],
) -> None:
    an = make_user("an@example.com")

    response = client.put(
        f"{COLLECTIONS}/{hr_docs['id']}/managers/{an.id}", headers=admin_headers
    )

    assert response.status_code == 422


def test_manager_routes_report_unknown_collections_and_users(
    client: TestClient,
    make_user: Callable[..., User],
    admin_headers: dict[str, str],
    hr_docs: dict[str, Any],
) -> None:
    binh = make_user("binh@example.com", role="manager")
    unknown = uuid.uuid4()

    for method in ("PUT", "DELETE"):
        no_collection = client.request(
            method, f"{COLLECTIONS}/{unknown}/managers/{binh.id}", headers=admin_headers
        )
        no_user = client.request(
            method,
            f"{COLLECTIONS}/{hr_docs['id']}/managers/{unknown}",
            headers=admin_headers,
        )
        assert (no_collection.status_code, error_code(no_collection)) == (
            404,
            "collection_not_found",
        )
        assert (no_user.status_code, error_code(no_user)) == (404, "user_not_found")


def test_who_sees_and_chooses_managers(
    client: TestClient,
    groups: GroupRepository,
    users: UserRepository,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    hr: Group,
    hr_docs: dict[str, Any],
) -> None:
    manager = headers_for("manager@example.com", "manager")
    reader = headers_for("an@example.com", "reader")
    groups.add_member(user_id=user_id(users, "an@example.com"), group_id=hr.id)
    managers = f"{COLLECTIONS}/{hr_docs['id']}/managers"
    manager_id = user_id(users, "manager@example.com")

    assert client.get(managers, headers=manager).status_code == 200
    assert client.get(managers, headers=reader).status_code == 403
    assert client.put(f"{managers}/{manager_id}", headers=manager).status_code == 403
    assert client.delete(f"{managers}/{manager_id}", headers=manager).status_code == 403


def test_a_front_end_acting_for_an_admin_cannot_change_collections(
    client: TestClient,
    auth: AuthService,
    make_user: Callable[..., User],
    hr_docs: dict[str, Any],
) -> None:
    service_user = make_user("openwebui@example.com")
    _, admin = sign_in(auth, "admin@example.com")
    key = auth.create_front_end_key(admin, service_user_id=service_user.id, name="owui")
    as_admin = {**bearer(key.token), "x-act-for-user": "admin@example.com"}
    one = f"{COLLECTIONS}/{hr_docs['id']}"

    assert client.get(one, headers=as_admin).status_code == 200  # seeing is fine
    assert (
        client.post(
            COLLECTIONS, headers=as_admin, json=new_collection(slug="x")
        ).status_code
        == 403
    )
    assert client.patch(one, headers=as_admin, json={"title": "x"}).status_code == 403
    assert (
        client.put(
            f"{one}/groups", headers=as_admin, json={"group_ids": []}
        ).status_code
        == 403
    )
