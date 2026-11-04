from dataclasses import dataclass

from argon2 import PasswordHasher
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from kb_server.services.passwords import Passwords
from kb_server.settings import Settings


@dataclass(frozen=True, kw_only=True)
class Services:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    passwords: Passwords


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
