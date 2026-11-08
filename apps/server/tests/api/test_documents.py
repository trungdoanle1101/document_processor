"""POST /api/v1/documents (multipart upload) and GET /api/v1/documents/{id}."""

import io
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kb_server.domain.collections import Collection
from kb_server.domain.groups import Group
from kb_server.domain.users import User, UserRole
from kb_server.services.auth import AuthService
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.users import UserRepository
from tests.support import MAX_UPLOAD_MB, bearer, sign_in

DOCUMENTS = "/api/v1/documents"
CONTENT = "Điều 1. Phạm vi điều chỉnh\n".encode()
DOCUMENT_FIELDS = {
    "id",
    "collection_id",
    "title",
    "status",
    "current_version",
    "created_at",
    "updated_at",
    "groups",
    "file",
    "job",
}


@pytest.fixture
def headers_for(
    auth: AuthService, make_user: Callable[..., User]
) -> Callable[[str, UserRole], dict[str, str]]:
    def make(email: str, role: UserRole) -> dict[str, str]:
        make_user(email, role=role)
        return bearer(sign_in(auth, email)[0])

    return make


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def hr_docs(collections: CollectionRepository, hr: Group) -> Collection:
    collection = collections.add(slug="hr-docs", title="HR documents", description="")
    collections.set_groups(collection.id, [hr.id])
    return collection


@pytest.fixture
def admin_headers(
    headers_for: Callable[[str, UserRole], dict[str, str]],
) -> dict[str, str]:
    return headers_for("admin@example.com", "admin")


def post(
    client: TestClient,
    headers: dict[str, str],
    fields: dict[str, Any],
    *,
    content: bytes = CONTENT,
    filename: str = "quy-che.txt",
) -> Any:
    return client.post(
        DOCUMENTS,
        headers=headers,
        data=fields,
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
    )


def upload(
    client: TestClient, headers: dict[str, str], collection: Collection, **fields: Any
) -> dict[str, Any]:
    content = fields.pop("content", CONTENT)
    response = post(
        client,
        headers,
        {"collection_id": str(collection.id), **fields},
        content=content,
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def error_code(response: Any) -> str:
    code: str = response.json()["error"]["code"]
    return code


# uploading


def test_an_upload_returns_the_new_document(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    response = post(client, admin_headers, {"collection_id": str(hr_docs.id)})

    assert response.status_code == 201
    body = response.json()
    assert set(body) == DOCUMENT_FIELDS
    assert body["collection_id"] == str(hr_docs.id)
    assert body["title"] == "quy-che"
    assert body["status"] == "active"
    assert body["current_version"] == 1
    assert [group["name"] for group in body["groups"]] == ["hr"]
    assert body["file"]["filename"] == "quy-che.txt"
    assert body["file"]["size"] == len(CONTENT)
    assert body["file"]["media_type"] == "text/plain"
    assert set(body["job"]) == {"job_id", "status", "error"}
    assert body["job"]["status"] == "queued"
    assert body["job"]["error"] is None


def test_a_title_can_be_given(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    body = upload(client, admin_headers, hr_docs, title="  Quy chế 2026 ")

    assert body["title"] == "Quy chế 2026"


def test_group_ids_are_sent_as_repeated_fields(
    client: TestClient,
    groups: GroupRepository,
    admin_headers: dict[str, str],
    hr: Group,
    hr_docs: Collection,
) -> None:
    it = groups.add("it")

    body = upload(client, admin_headers, hr_docs, group_ids=[str(hr.id), str(it.id)])

    assert sorted(group["name"] for group in body["groups"]) == ["hr", "it"]


@pytest.mark.parametrize("value", ["true", "1", "on"])
def test_private_is_a_form_flag(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection, value: str
) -> None:
    assert upload(client, admin_headers, hr_docs, private=value)["groups"] == []


# refused uploads


def test_upload_needs_a_token(client: TestClient, hr_docs: Collection) -> None:
    assert post(client, {}, {"collection_id": str(hr_docs.id)}).status_code == 401


@pytest.mark.parametrize(
    ("extra", "field"),
    [
        ({"title": "T" * 501}, "title"),
        ({"group_ids": ["not-a-uuid"]}, "group_ids.0"),
    ],
    ids=["title too long", "malformed group id"],
)
def test_a_malformed_field_is_422(
    client: TestClient,
    admin_headers: dict[str, str],
    hr_docs: Collection,
    extra: dict[str, Any],
    field: str,
) -> None:
    response = post(client, admin_headers, {"collection_id": str(hr_docs.id), **extra})

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == field


@pytest.mark.parametrize(
    "form", [{}, {"collection_id": "not-a-uuid"}], ids=["missing", "malformed"]
)
def test_the_collection_id_is_required_and_checked(
    client: TestClient, admin_headers: dict[str, str], form: dict[str, Any]
) -> None:
    response = post(client, admin_headers, form)

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "collection_id"


def test_no_file_is_422(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    response = client.post(
        DOCUMENTS, headers=admin_headers, data={"collection_id": str(hr_docs.id)}
    )

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "file"


def test_private_and_group_ids_together_are_422(
    client: TestClient, admin_headers: dict[str, str], hr: Group, hr_docs: Collection
) -> None:
    response = post(
        client,
        admin_headers,
        {
            "collection_id": str(hr_docs.id),
            "group_ids": [str(hr.id)],
            "private": "true",
        },
    )

    assert response.status_code == 422
    assert error_code(response) == "invalid_request"


def test_unknown_group_ids_are_422(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    response = post(
        client,
        admin_headers,
        {"collection_id": str(hr_docs.id), "group_ids": [str(uuid.uuid4())]},
    )

    assert response.status_code == 422


def test_the_same_file_again_is_409_naming_the_document(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    first = upload(client, admin_headers, hr_docs)

    response = post(
        client, admin_headers, {"collection_id": str(hr_docs.id)}, filename="other.txt"
    )

    assert response.status_code == 409
    assert error_code(response) == "duplicate_file"
    assert response.json()["error"]["details"] == {"document_id": first["id"]}


def test_a_file_over_the_limit_is_413(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    too_big = b"x" * (MAX_UPLOAD_MB * 1024 * 1024 + 1)

    response = post(
        client, admin_headers, {"collection_id": str(hr_docs.id)}, content=too_big
    )

    assert response.status_code == 413
    assert error_code(response) == "file_too_large"


def test_an_unsupported_file_is_415(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    response = post(
        client,
        admin_headers,
        {"collection_id": str(hr_docs.id)},
        content=b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n",
        filename="quy-che.pdf",
    )

    assert response.status_code == 415
    assert error_code(response) == "unsupported_file_type"
    assert ".docx" in response.json()["error"]["message"]


def test_an_empty_file_is_422(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    response = post(
        client, admin_headers, {"collection_id": str(hr_docs.id)}, content=b""
    )

    assert response.status_code == 422


def test_who_may_upload(
    client: TestClient,
    groups: GroupRepository,
    users: UserRepository,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    hr: Group,
    hr_docs: Collection,
) -> None:
    outsider = headers_for("binh@example.com", "reader")
    hr_reader = headers_for("an@example.com", "reader")
    an = users.get_by_email("an@example.com")
    assert an is not None
    groups.add_member(user_id=an.id, group_id=hr.id)
    other_manager = headers_for("other-manager@example.com", "manager")
    fields = {"collection_id": str(hr_docs.id)}

    hidden = post(client, outsider, fields)
    assert (hidden.status_code, error_code(hidden)) == (404, "collection_not_found")
    assert post(client, hr_reader, fields).status_code == 403
    assert post(client, other_manager, fields).status_code == 403


# reading


def test_get_returns_what_the_upload_returned(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    uploaded = upload(client, admin_headers, hr_docs)

    response = client.get(f"{DOCUMENTS}/{uploaded['id']}", headers=admin_headers)

    assert response.status_code == 200
    assert response.json() == uploaded


def test_a_document_you_cannot_read_is_404(
    client: TestClient,
    headers_for: Callable[[str, UserRole], dict[str, str]],
    admin_headers: dict[str, str],
    hr_docs: Collection,
) -> None:
    uploaded = upload(client, admin_headers, hr_docs)
    outsider = headers_for("binh@example.com", "reader")

    response = client.get(f"{DOCUMENTS}/{uploaded['id']}", headers=outsider)

    assert response.status_code == 404
    assert error_code(response) == "document_not_found"


def test_an_unknown_or_malformed_document_id(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    assert (
        client.get(f"{DOCUMENTS}/{uuid.uuid4()}", headers=admin_headers).status_code
        == 404
    )
    assert (
        client.get(f"{DOCUMENTS}/not-a-uuid", headers=admin_headers).status_code == 422
    )


def test_reading_needs_a_token(
    client: TestClient, admin_headers: dict[str, str], hr_docs: Collection
) -> None:
    uploaded = upload(client, admin_headers, hr_docs)

    assert client.get(f"{DOCUMENTS}/{uploaded['id']}").status_code == 401
