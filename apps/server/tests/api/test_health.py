from dataclasses import replace

from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from kb_server.app import create_app
from kb_server.wiring import Services


def test_health_needs_nothing(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_the_database_answers(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_not_ready_when_the_database_is_down(services: Services) -> None:
    # nothing listens on port 1
    down = create_engine("postgresql+psycopg://kb:kb@127.0.0.1:1/kb")
    with TestClient(create_app(replace(services, engine=down))) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "database unreachable"}
