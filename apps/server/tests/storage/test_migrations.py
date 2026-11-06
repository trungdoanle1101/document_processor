import uuid
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Connection, inspect, text

from kb_server.storage import models  # noqa: F401  (fills Base.metadata)
from kb_server.storage.database import Base

ALEMBIC_INI = Path(__file__).parents[2] / "alembic.ini"


def test_the_migrations_match_the_models(connection: Connection) -> None:
    # fails when a model changed and no migration was generated for it
    differences = compare_metadata(
        MigrationContext.configure(connection), Base.metadata
    )

    assert differences == []


def test_the_migrations_can_be_undone_and_redone(connection: Connection) -> None:
    # inside the test's transaction: Postgres rolls the DDL back afterwards
    config = Config(str(ALEMBIC_INI))
    config.attributes["connection"] = connection

    command.downgrade(config, "base")
    assert "users" not in inspect(connection).get_table_names()

    command.upgrade(config, "head")
    assert "users" in inspect(connection).get_table_names()


def test_existing_users_join_everyone_when_groups_arrive(
    connection: Connection,
) -> None:
    # a database from before groups, with one user in it
    config = Config(str(ALEMBIC_INI))
    config.attributes["connection"] = connection
    command.downgrade(config, "f92ea875949a")
    user_id = uuid.uuid4()
    connection.execute(
        text(
            "INSERT INTO users (id, email, display_name, password_hash, role, is_active)"
            " VALUES (:id, 'an@example.com', 'An', 'hash', 'reader', true)"
        ),
        {"id": user_id},
    )

    command.upgrade(config, "head")

    members = connection.execute(
        text(
            "SELECT ug.user_id FROM user_groups ug"
            " JOIN groups g ON g.id = ug.group_id WHERE g.name = 'everyone'"
        )
    ).scalars()
    assert list(members) == [user_id]
