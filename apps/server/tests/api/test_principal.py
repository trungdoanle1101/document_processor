"""Who a request acts as: the bearer token, the act-for header and required roles.

No real route needs a principal yet, so these tests add small probe routes.
"""

from collections.abc import Callable, Iterator
from dataclasses import replace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from kb_server.api.deps import AdminPrincipal, CurrentPrincipal, require_role
from kb_server.app import create_app
from kb_server.domain.users import User
from kb_server.services.auth import AuthService
from kb_server.wiring import Services
from tests.support import bearer, sign_in

OPEN_WEBUI_HEADER = "x-openwebui-user-email"


def probe_app(services: Services) -> FastAPI:
    app = create_app(services)

    @app.get("/probe/me")
    def me(principal: CurrentPrincipal) -> dict[str, str | bool]:
        return {
            "email": principal.user.email,
            "kind": principal.credential_kind,
            "acting_for_others": principal.acting_for_others,
        }

    @app.get("/probe/admin")
    def admin_only(principal: AdminPrincipal) -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/probe/managers", dependencies=[Depends(require_role("manager", "admin"))]
    )
    def managers_only() -> dict[str, str]:
        return {"status": "ok"}

    return app


@pytest.fixture
def client(services: Services) -> Iterator[TestClient]:
    # both act-for headers configured, as for Open WebUI
    services = replace(
        services,
        settings=services.settings.model_copy(
            update={"act_for_headers": ("x-act-for-user", OPEN_WEBUI_HEADER)}
        ),
    )
    with TestClient(probe_app(services)) as client:
        yield client


@pytest.fixture
def an_token(auth: AuthService, make_user: Callable[..., User]) -> str:
    make_user("an@example.com")
    return sign_in(auth, "an@example.com")[0]


@pytest.fixture
def front_end_token(auth: AuthService, make_user: Callable[..., User]) -> str:
    make_user("admin@example.com", role="admin")
    service_user = make_user("openwebui@example.com")
    make_user("binh@example.com")
    _, admin = sign_in(auth, "admin@example.com")
    return auth.create_front_end_key(
        admin, service_user_id=service_user.id, name="open webui"
    ).token


def test_a_token_identifies_its_user(client: TestClient, an_token: str) -> None:
    response = client.get("/probe/me", headers=bearer(an_token))

    assert response.status_code == 200
    assert response.json() == {
        "email": "an@example.com",
        "kind": "session",
        "acting_for_others": False,
    }


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer kb_s_unknown"},
        {"Authorization": "Bearer not-a-token"},
        {"Authorization": "Basic YW46cGFzc3dvcmQ="},
    ],
    ids=["no token", "unknown token", "junk token", "not bearer"],
)
def test_no_usable_token_is_401_with_a_bearer_challenge(
    client: TestClient, headers: dict[str, str]
) -> None:
    response = client.get("/probe/me", headers=headers)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "not_authenticated"


def test_a_front_end_key_acts_for_the_user_in_either_header(
    client: TestClient, front_end_token: str
) -> None:
    for header in ("x-act-for-user", OPEN_WEBUI_HEADER):
        response = client.get(
            "/probe/me",
            headers={**bearer(front_end_token), header: "binh@example.com"},
        )

        assert response.status_code == 200
        assert response.json()["email"] == "binh@example.com"
        assert response.json()["acting_for_others"] is True


def test_the_same_user_in_both_headers_is_fine(
    client: TestClient, front_end_token: str
) -> None:
    response = client.get(
        "/probe/me",
        headers={
            **bearer(front_end_token),
            "x-act-for-user": "binh@example.com",
            OPEN_WEBUI_HEADER: "binh@example.com",
        },
    )

    assert response.status_code == 200


def test_two_different_users_in_the_headers_are_refused(
    client: TestClient, front_end_token: str
) -> None:
    response = client.get(
        "/probe/me",
        headers={
            **bearer(front_end_token),
            "x-act-for-user": "binh@example.com",
            OPEN_WEBUI_HEADER: "admin@example.com",
        },
    )

    assert response.status_code == 401


def test_a_front_end_key_without_a_user_is_refused(
    client: TestClient, front_end_token: str
) -> None:
    assert client.get("/probe/me", headers=bearer(front_end_token)).status_code == 401


def test_an_ordinary_token_naming_a_user_is_refused(
    client: TestClient, an_token: str, make_user: Callable[..., User]
) -> None:
    make_user("binh@example.com")

    response = client.get(
        "/probe/me", headers={**bearer(an_token), "x-act-for-user": "binh@example.com"}
    )

    assert response.status_code == 401


def test_an_unconfigured_header_is_ignored(
    services: Services, an_token: str, make_user: Callable[..., User]
) -> None:
    # the default settings: only x-act-for-user
    make_user("binh@example.com")
    with TestClient(probe_app(services)) as client:
        response = client.get(
            "/probe/me",
            headers={**bearer(an_token), OPEN_WEBUI_HEADER: "binh@example.com"},
        )

    assert response.status_code == 200
    assert response.json()["email"] == "an@example.com"


def test_a_route_for_admins_refuses_others(client: TestClient, an_token: str) -> None:
    response = client.get("/probe/admin", headers=bearer(an_token))

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "not_allowed"


def test_a_route_for_admins_lets_an_admin_in(
    client: TestClient, auth: AuthService, make_user: Callable[..., User]
) -> None:
    make_user("admin@example.com", role="admin")
    token, _ = sign_in(auth, "admin@example.com")

    assert client.get("/probe/admin", headers=bearer(token)).status_code == 200


def test_a_front_end_acting_for_an_admin_is_not_an_admin(
    client: TestClient, front_end_token: str
) -> None:
    response = client.get(
        "/probe/admin",
        headers={**bearer(front_end_token), "x-act-for-user": "admin@example.com"},
    )

    assert response.status_code == 403


def test_a_role_route_lets_in_the_listed_roles(
    client: TestClient, auth: AuthService, make_user: Callable[..., User]
) -> None:
    make_user("manager@example.com", role="manager")
    token, _ = sign_in(auth, "manager@example.com")

    assert client.get("/probe/managers", headers=bearer(token)).status_code == 200


def test_a_role_route_refuses_other_roles(client: TestClient, an_token: str) -> None:
    assert client.get("/probe/managers", headers=bearer(an_token)).status_code == 403


def test_a_front_end_gets_the_role_of_the_user_it_acts_for(
    client: TestClient, front_end_token: str
) -> None:
    # chatting through Open WebUI keeps your role, just not admin powers
    as_admin = {**bearer(front_end_token), "x-act-for-user": "admin@example.com"}
    as_reader = {**bearer(front_end_token), "x-act-for-user": "binh@example.com"}

    assert client.get("/probe/managers", headers=as_admin).status_code == 200
    assert client.get("/probe/managers", headers=as_reader).status_code == 403
