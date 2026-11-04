"""Every error comes back as {"error": {"code", "message", "details"}}.

The probe routes raise each error the handlers know about.
"""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from kb_server.app import create_app
from kb_server.domain.errors import (
    CredentialNotFound,
    EmailTaken,
    InvalidRequest,
    NotAllowed,
    NotAuthenticated,
)
from kb_server.wiring import Services

RAISED = {
    "credential-not-found": CredentialNotFound("key not found for the owner"),
    "email-taken": EmailTaken("an@example.com"),
    "invalid-request": InvalidRequest("expires_at must be in the future"),
    "not-allowed": NotAllowed("front-end keys can only be created by admins"),
    "not-authenticated": NotAuthenticated(),
}


class Item(BaseModel):
    name: str
    count: int


def probe_app(services: Services) -> FastAPI:
    app = create_app(services)

    @app.get("/probe/raise/{name}")
    def raise_error(name: str) -> None:
        raise RAISED[name]

    @app.post("/probe/items")
    def add_item(item: Item) -> Item:
        return item

    return app


@pytest.fixture
def client(services: Services) -> Iterator[TestClient]:
    with TestClient(probe_app(services)) as client:
        yield client


@pytest.mark.parametrize(
    ("name", "status", "code", "message"),
    [
        (
            "credential-not-found",
            404,
            "credential_not_found",
            "key not found for the owner",
        ),
        ("email-taken", 409, "email_taken", "a user with this email already exists"),
        ("invalid-request", 422, "invalid_request", "expires_at must be in the future"),
        (
            "not-allowed",
            403,
            "not_allowed",
            "front-end keys can only be created by admins",
        ),
        (
            "not-authenticated",
            401,
            "unauthenticated",
            "not signed in, or the token is invalid",
        ),
    ],
)
def test_a_domain_error_maps_to_its_status_and_code(
    client: TestClient, name: str, status: int, code: str, message: str
) -> None:
    response = client.get(f"/probe/raise/{name}")

    assert response.status_code == status
    assert response.json() == {
        "error": {"code": code, "message": message, "details": None}
    }


def test_an_invalid_body_lists_each_problem(client: TestClient) -> None:
    response = client.post("/probe/items", json={"count": "many"})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert {detail["field"] for detail in error["details"]} == {"name", "count"}


def test_an_unknown_path_is_404(client: TestClient) -> None:
    response = client.get("/no/such/path")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_a_wrong_method_is_405(client: TestClient) -> None:
    response = client.delete("/health")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"
