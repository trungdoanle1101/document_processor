from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from http.client import FORBIDDEN
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from kb_server.domain.credentials import NotAuthenticated, Principal
from kb_server.domain.users import UserRole
from kb_server.services.auth import AuthService
from kb_server.storage.repositories.credentials import CredentialRepository
from kb_server.storage.repositories.users import UserRepository
from kb_server.wiring import Services


def get_services(request: Request) -> Services:
    return cast(Services, request.app.state.services)


ServicesDep = Annotated[Services, Depends(get_services)]


def get_session(services: ServicesDep) -> Iterator[Session]:
    with services.session_factory.begin() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def get_auth_service(session: SessionDep, services: ServicesDep) -> AuthService:
    return AuthService(
        users=UserRepository(session),
        credentials=CredentialRepository(session),
        passwords=services.passwords,
        session_ttl=timedelta(hours=services.settings.session_ttl_hours),
        now=lambda: datetime.now(UTC),
    )


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]

bearer = HTTPBearer(auto_error=False)


def get_principal(
    request: Request,
    auth: AuthServiceDep,
    services: ServicesDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    if credentials is None:
        raise NotAuthenticated()
    principal = auth.resolve(
        credentials.credentials,
        acting_for=_acting_for(request, services.settings.act_for_headers),
    )
    if principal is None:
        raise NotAuthenticated()
    return principal


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


def require_role(*roles: UserRole) -> Callable[[Principal], Principal]:
    def check(principal: CurrentPrincipal) -> Principal:
        if principal.user.role not in roles:
            raise HTTPException(status_code=FORBIDDEN, detail="Not allowed.")
        return principal

    return check


def _acting_for(request: Request, header_names: Sequence[str]) -> str | None:
    """The user a front end says it's acting for, from the first configured header present."""
    values = {request.headers[name] for name in header_names if name in request.headers}
    if len(values) > 1:
        raise NotAuthenticated()
    return values.pop() if values else None
