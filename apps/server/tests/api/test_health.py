import logging
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from kb_server.app import create_app
from kb_server.wiring import Services
from tests.support import RefusingStore

# nothing listens on port 1
DOWN_DATABASE_URL = "postgresql+psycopg://kb:kb@127.0.0.1:1/kb"


def test_health_needs_nothing(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_the_database_answers(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_not_ready_when_the_database_is_down(services: Services) -> None:
    down = create_engine(DOWN_DATABASE_URL)
    with TestClient(create_app(replace(services, engine=down))) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "database unreachable"}


def test_not_ready_when_the_store_cant_be_written(
    services: Services, caplog: pytest.LogCaptureFixture
) -> None:
    denied = PermissionError(13, "Permission denied", "/srv/kb/write_checks")
    refusing = RefusingStore(services.files, on_put=denied)

    with (
        caplog.at_level(logging.WARNING, logger="kb_server.api.health"),
        TestClient(create_app(replace(services, files=refusing))) as client,
    ):
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "storage not writable"}
    # the reason goes to the log, never to whoever polls /ready
    assert "Permission denied" not in response.text
    assert any("Permission denied" in record.getMessage() for record in caplog.records)


def test_the_database_is_reported_before_the_store(services: Services) -> None:
    denied = PermissionError(13, "Permission denied", "/srv/kb/write_checks")
    both_down = replace(
        services,
        engine=create_engine(DOWN_DATABASE_URL),
        files=RefusingStore(services.files, on_put=denied),
    )

    with TestClient(create_app(both_down)) as client:
        response = client.get("/ready")

    assert response.json() == {"status": "database unreachable"}
