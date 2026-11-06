from collections.abc import Callable, Iterator, Sequence
from typing import Annotated, cast

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from kb_server.domain.credentials import Principal
from kb_server.domain.errors import NotAllowed, NotAuthenticated
from kb_server.domain.users import UserRole
from kb_server.services.auth import AuthService
from kb_server.services.collections import CollectionService
from kb_server.services.groups import GroupService
from kb_server.services.users import UserService
from kb_server.wiring import (
    Services,
    build_auth_service,
    build_collection_service,
    build_group_service,
    build_user_service,
)


def get_services(request: Request) -> Services:
    return cast(Services, request.app.state.services)


ServicesDep = Annotated[Services, Depends(get_services)]


def get_session(services: ServicesDep) -> Iterator[Session]:
    with services.session_factory.begin() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def get_auth_service(session: SessionDep, services: ServicesDep) -> AuthService:
    return build_auth_service(session, services)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


def get_user_service(session: SessionDep, services: ServicesDep) -> UserService:
    return build_user_service(session, services)


UserServiceDep = Annotated[UserService, Depends(get_user_service)]


def get_group_service(session: SessionDep) -> GroupService:
    return build_group_service(session)


GroupServiceDep = Annotated[GroupService, Depends(get_group_service)]


def get_collection_service(session: SessionDep) -> CollectionService:
    return build_collection_service(session)


CollectionServiceDep = Annotated[CollectionService, Depends(get_collection_service)]


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
            raise NotAllowed()
        return principal

    return check


def require_admin(principal: CurrentPrincipal) -> Principal:
    """An admin acting as themselves; a front end acting for an admin doesn't count."""
    if not principal.is_admin:
        raise NotAllowed()
    return principal


AdminPrincipal = Annotated[Principal, Depends(require_admin)]


def _acting_for(request: Request, header_names: Sequence[str]) -> str | None:
    """The user a front end says it's acting for, from the first configured header present."""
    values = {request.headers[name] for name in header_names if name in request.headers}
    if len(values) > 1:
        raise NotAuthenticated()
    return values.pop() if values else None
