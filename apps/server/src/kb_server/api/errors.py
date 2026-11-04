from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from kb_server.domain.errors import (
    CredentialNotFound,
    EmailTaken,
    InvalidRequest,
    NotAllowed,
    NotAuthenticated,
)

_CODES = {
    401: "unauthenticated",
    403: "not_allowed",
    404: "not_found",
    405: "method_not_allowed",
}


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(CredentialNotFound, _credential_not_found)
    app.add_exception_handler(EmailTaken, _email_taken)
    app.add_exception_handler(InvalidRequest, _invalid_request)
    app.add_exception_handler(NotAllowed, _not_allowed)
    app.add_exception_handler(NotAuthenticated, _not_authenticated)
    app.add_exception_handler(RequestValidationError, _request_validation_error)
    app.add_exception_handler(StarletteHTTPException, _starlette_http_exception)


def _email_taken(request: Request, error: Exception) -> JSONResponse:
    return _error(
        HTTPStatus.CONFLICT, "email_taken", "a user with this email already exists"
    )


def _credential_not_found(request: Request, error: Exception) -> JSONResponse:
    return _error(
        HTTPStatus.NOT_FOUND,
        "credential_not_found",
        str(error) or "credential not found",
    )


def _invalid_request(request: Request, error: Exception) -> JSONResponse:
    return _error(
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "invalid_request",
        str(error) or "invalid request",
    )


def _not_allowed(request: Request, error: Exception) -> JSONResponse:
    return _error(
        HTTPStatus.FORBIDDEN,
        "not_allowed",
        str(error) or "not allowed",
    )


def _not_authenticated(request: Request, error: Exception) -> JSONResponse:
    return _error(
        HTTPStatus.UNAUTHORIZED,
        "unauthenticated",
        "not signed in, or the token is invalid",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _request_validation_error(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, RequestValidationError)
    fields = [
        {
            "field": ".".join(str(part) for part in item["loc"][1:]),
            "problem": item["msg"],
        }
        for item in error.errors()
    ]
    return _error(
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "invalid_request",
        "the request is invalid",
        details=fields,
    )


def _starlette_http_exception(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, StarletteHTTPException)
    status = HTTPStatus(error.status_code)
    code = _CODES.get(status, status.name.lower())
    message = (
        error.detail
        if isinstance(error.detail, str) and error.detail
        else status.phrase
    )
    return _error(
        status,
        code,
        message,
        headers=dict(error.headers) if error.headers is not None else None,
    )


def _error(
    status: int,
    code: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
    details: list[dict[str, str]] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details}},
        headers=headers,
    )
