from datetime import datetime
from http import HTTPStatus
from typing import Literal

from fastapi import APIRouter, Response
from pydantic import BaseModel, ConfigDict, EmailStr

from kb_server.api.deps import AuthServiceDep, CurrentPrincipal
from kb_server.domain.errors import LoginFailed

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginIn(BaseModel):
    email: EmailStr
    password: str

    model_config = ConfigDict(extra="forbid")


class TokenOut(BaseModel):
    token: str
    expires_at: datetime
    token_type: Literal["bearer"] = "bearer"


@router.post("/login")
def login(body: LoginIn, auth: AuthServiceDep, response: Response) -> TokenOut:
    issued = auth.login(body.email, body.password)
    if issued is None:
        raise LoginFailed()
    assert issued.credential.expires_at is not None
    response.headers["Cache-Control"] = "no-store"
    return TokenOut(token=issued.token, expires_at=issued.credential.expires_at)


@router.post("/logout", status_code=HTTPStatus.NO_CONTENT)
def logout(principal: CurrentPrincipal, auth: AuthServiceDep) -> None:
    auth.logout(principal)
