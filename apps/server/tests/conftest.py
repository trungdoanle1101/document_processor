"""Fixtures shared by the server tests.

One Postgres per test run, migrated once. Each test runs inside a transaction that is
rolled back afterwards, so tests never see each other's rows.
"""

import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from sqlalchemy import Connection, Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from kb_server.app import create_app
from kb_server.domain.users import User, UserRole
from kb_server.services.auth import AuthService
from kb_server.services.passwords import Passwords
from kb_server.services.users import UserService
from kb_server.settings import Settings
from kb_server.storage.repositories.credentials import CredentialRepository
from kb_server.storage.repositories.users import UserRepository
from kb_server.wiring import Services
from tests.support import PASSWORD, FakeClock

SERVER_DIR = Path(__file__).parents[1]


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """KB_TEST_DATABASE_URL if set (fast: reuses a running Postgres), else a throwaway container."""
    url = os.environ.get("KB_TEST_DATABASE_URL")
    if url:
        yield url
        return
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:17-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture(scope="session")
def engine(database_url: str) -> Iterator[Engine]:
    engine = create_engine(database_url)
    config = Config(str(SERVER_DIR / "alembic.ini"))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    yield engine
    engine.dispose()


@pytest.fixture
def connection(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as connection:
        transaction = connection.begin()
        yield connection
        transaction.rollback()


@pytest.fixture
def session_factory(connection: Connection) -> sessionmaker[Session]:
    # sessions join the test's transaction: their commits only release a savepoint
    return sessionmaker(bind=connection, join_transaction_mode="create_savepoint")


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as session:
        yield session


@pytest.fixture(scope="session")
def passwords() -> Passwords:
    # the cheapest argon2 settings: real hashing, but fast
    return Passwords(PasswordHasher(time_cost=1, memory_cost=8, parallelism=1))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(datetime(2026, 11, 5, 9, 0, tzinfo=UTC))


@pytest.fixture
def users(session: Session) -> UserRepository:
    return UserRepository(session)


@pytest.fixture
def credentials(session: Session) -> CredentialRepository:
    return CredentialRepository(session)


@pytest.fixture
def auth(
    users: UserRepository,
    credentials: CredentialRepository,
    passwords: Passwords,
    clock: FakeClock,
) -> AuthService:
    return AuthService(
        users=users,
        credentials=credentials,
        passwords=passwords,
        session_ttl=timedelta(hours=12),
        now=clock,
    )


@pytest.fixture
def user_service(
    users: UserRepository,
    credentials: CredentialRepository,
    passwords: Passwords,
    clock: FakeClock,
) -> UserService:
    return UserService(
        users=users, credentials=credentials, passwords=passwords, now=clock
    )


@pytest.fixture
def make_user(users: UserRepository, passwords: Passwords) -> Callable[..., User]:
    """make_user("an@example.com", role="admin") -> a saved user whose password is PASSWORD."""

    def make(email: str, *, role: UserRole = "reader") -> User:
        return users.add(
            email=email,
            display_name=email.split("@")[0],
            role=role,
            password_hash=passwords.hash(PASSWORD),
        )

    return make


@pytest.fixture
def services(
    engine: Engine,
    session_factory: sessionmaker[Session],
    passwords: Passwords,
    clock: FakeClock,
    database_url: str,
) -> Services:
    return Services(
        settings=Settings(_env_file=None, database_url=database_url),  # ignore .env
        engine=engine,
        session_factory=session_factory,
        passwords=passwords,
        clock=clock,
    )


@pytest.fixture
def client(services: Services) -> Iterator[TestClient]:
    with TestClient(create_app(services)) as client:
        yield client
