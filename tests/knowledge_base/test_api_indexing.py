"""Adding and removing documents through a running server, guarded by the admin token."""

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from knowledge_base.api import create_api
from knowledge_base.settings import parse_settings
from tests.knowledge_base.helpers import (
    DATA_REGULATION,
    Fakes,
    chunks_of,
    make_fakes,
    write,
)

TOKEN = "a-long-enough-admin-token"
ADMIN = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def fakes() -> Fakes:
    return make_fakes()


@pytest.fixture
def client(fakes: Fakes) -> TestClient:
    return TestClient(
        create_api(fakes.services, parse_settings({"admin_token": TOKEN}))
    )


def test_adding_indexes_with_the_servers_models(
    client: TestClient, fakes: Fakes, tmp_path: Path
) -> None:
    path = write(tmp_path, "du-lieu.txt", DATA_REGULATION)
    body = {
        "path": str(path),
        "group": "it",
        "title": "Quy định dữ liệu",
        "attributes": {
            "kind": "quy_dinh",
            "issued_on": "2025-03-01",
            "year": 2025,
            "tags": ["a", "b"],
        },
    }
    response = client.post("/v1/documents", json=body, headers=ADMIN)

    assert response.status_code == 200
    added = response.json()
    assert (added["document_id"], added["action"], added["version"]) == (
        "du-lieu",
        "indexed",
        1,
    )
    assert added["chunk_count"] == len(chunks_of(fakes, "du-lieu")) > 0
    attributes = chunks_of(fakes, "du-lieu")[0].attributes
    assert (attributes["issued_on"], attributes["year"], attributes["tags"]) == (
        date(2025, 3, 1),
        2025,
        ("a", "b"),
    )


def test_adding_again_reports_what_changed(client: TestClient, tmp_path: Path) -> None:
    body = {"path": str(write(tmp_path, "a.txt", DATA_REGULATION)), "group": "it"}
    client.post("/v1/documents", json=body, headers=ADMIN)
    assert (
        client.post("/v1/documents", json=body, headers=ADMIN).json()["action"]
        == "unchanged"
    )
    moved = client.post(
        "/v1/documents", json={**body, "group": "all"}, headers=ADMIN
    ).json()
    assert (moved["action"], moved["version"]) == ("attributes updated", 2)


def test_a_new_document_is_searchable_at_once(
    client: TestClient, tmp_path: Path
) -> None:
    client.post(
        "/v1/documents",
        json={
            "path": str(write(tmp_path, "a.txt", DATA_REGULATION)),
            "group": "public",
        },
        headers=ADMIN,
    )
    hits = client.post(
        "/v1/search", json={"question": "Staging lưu trữ bao lâu?"}
    ).json()["hits"]
    assert {h["document_id"] for h in hits} == {"a"}


def test_removing_takes_the_document_out(
    client: TestClient, fakes: Fakes, tmp_path: Path
) -> None:
    client.post(
        "/v1/documents",
        json={
            "path": str(write(tmp_path, "qd 2025.txt", DATA_REGULATION)),
            "group": "it",
        },
        headers=ADMIN,
    )
    response = client.delete("/v1/documents/qd%202025", headers=ADMIN)
    assert response.json() == {"document_id": "qd 2025", "removed": True}
    assert chunks_of(fakes, "qd 2025") == []
    assert client.delete("/v1/documents/qd%202025", headers=ADMIN).status_code == 404


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token-of-some-length"},
        {"Authorization": TOKEN},
    ],
)
def test_writing_needs_the_admin_token(
    client: TestClient, tmp_path: Path, headers: dict[str, str]
) -> None:
    body = {"path": str(write(tmp_path, "a.txt", DATA_REGULATION)), "group": "it"}
    assert client.post("/v1/documents", json=body, headers=headers).status_code == 401
    assert client.delete("/v1/documents/a", headers=headers).status_code == 401


def test_writing_is_off_without_a_token_in_the_settings(
    fakes: Fakes, tmp_path: Path
) -> None:
    client = TestClient(create_api(fakes.services, parse_settings({})))
    body = {"path": str(write(tmp_path, "a.txt", DATA_REGULATION)), "group": "it"}
    response = client.post("/v1/documents", json=body, headers=ADMIN)
    assert response.status_code == 403
    assert "admin_token" in response.json()["detail"]


def test_a_missing_file_is_404(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/v1/documents",
        json={"path": str(tmp_path / "nope.txt"), "group": "it"},
        headers=ADMIN,
    )
    assert response.status_code == 404


def test_reserved_attributes_are_400(client: TestClient, tmp_path: Path) -> None:
    body = {
        "path": str(write(tmp_path, "a.txt", DATA_REGULATION)),
        "group": "it",
        "attributes": {"part": "x"},
    }
    response = client.post("/v1/documents", json=body, headers=ADMIN)
    assert (response.status_code, "set by the app" in response.json()["detail"]) == (
        400,
        True,
    )


def test_a_short_admin_token_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 16"):
        parse_settings({"admin_token": "short"})
