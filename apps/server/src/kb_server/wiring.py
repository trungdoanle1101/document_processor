from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from kb_server.services.auth import AuthService
from kb_server.services.groups import GroupService
from kb_server.services.passwords import Passwords
from kb_server.services.users import UserService
from kb_server.settings import Settings
from kb_server.storage.repositories.credentials import CredentialRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.users import UserRepository


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, kw_only=True)
class Services:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    passwords: Passwords
    clock: Callable[[], datetime] = utc_now


def build_services(settings: Settings) -> Services:
    engine = create_engine(url=settings.database_url, pool_pre_ping=True)
    return Services(
        settings=settings,
        engine=engine,
        session_factory=sessionmaker(engine),
        passwords=Passwords(hasher=PasswordHasher()),
    )


def close_services(services: Services) -> None:
    services.engine.dispose()


def build_user_service(session: Session, services: Services) -> UserService:
    return UserService(
        users=UserRepository(session),
        groups=GroupRepository(session),
        credentials=CredentialRepository(session),
        passwords=services.passwords,
        now=services.clock,
    )


def build_auth_service(session: Session, services: Services) -> AuthService:
    return AuthService(
        users=UserRepository(session),
        credentials=CredentialRepository(session),
        passwords=services.passwords,
        session_ttl=timedelta(hours=services.settings.session_ttl_hours),
        now=services.clock,
    )


def build_group_service(session: Session) -> GroupService:
    return GroupService(groups=GroupRepository(session), users=UserRepository(session))
