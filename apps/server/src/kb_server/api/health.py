from http import HTTPStatus

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from kb_server.api.deps import ServicesDep

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(services: ServicesDep) -> JSONResponse:
    try:
        with services.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError:
        return JSONResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            content={"status": "database unreachable"},
        )

    return JSONResponse(status_code=HTTPStatus.OK, content={"status": "ready"})
