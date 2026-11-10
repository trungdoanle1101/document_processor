import logging
from http import HTTPStatus

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from kb_server.api.deps import ServicesDep
from kb_server.domain.errors import StoreNotWritable
from kb_server.storage.files import check_store_writable

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(services: ServicesDep) -> JSONResponse:
    try:
        with services.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except OperationalError as error:
        logger.warning("not ready: %s", error)
        return JSONResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            content={"status": "database unreachable"},
        )

    try:
        check_store_writable(services.files)
    except StoreNotWritable as error:
        logger.warning("not ready: %s", error)
        return JSONResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            content={"status": "storage not writable"},
        )

    return JSONResponse(status_code=HTTPStatus.OK, content={"status": "ready"})
