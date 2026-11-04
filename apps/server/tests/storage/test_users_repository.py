import uuid

import pytest

from kb_server.domain.errors import EmailTaken
from kb_server.storage.repositories.users import UserRepository


def add(users: UserRepository, email: str = "an@example.com") -> None:
    users.add(email=email, display_name="An", role="reader", password_hash="hash")


def test_a_new_user_is_saved_with_defaults(users: UserRepository) -> None:
    users.add(
        email=" An@Example.com ",
        display_name="An",
        role="manager",
        password_hash="hash",
    )

    user = users.get_by_email("an@example.com")
    assert user is not None
    assert user.email == "an@example.com"
    assert user.role == "manager"
    assert user.is_active
    assert user.created_at.tzinfo is not None


def test_get_finds_a_user_by_id(users: UserRepository) -> None:
    add(users)
    user = users.get_by_email("an@example.com")
    assert user is not None

    assert users.get(user.id) == user
    assert users.get(uuid.uuid4()) is None


def test_get_by_email_ignores_case(users: UserRepository) -> None:
    add(users)

    assert users.get_by_email("AN@example.com") is not None
    assert users.get_by_email("binh@example.com") is None


def test_an_email_is_taken_whatever_its_case(users: UserRepository) -> None:
    add(users)

    with pytest.raises(EmailTaken):
        add(users, "AN@Example.com")

    # the savepoint undid only the failed insert
    assert users.get_by_email("an@example.com") is not None


def test_get_credentials_by_email_returns_the_password_hash(
    users: UserRepository,
) -> None:
    add(users)

    found = users.get_credentials_by_email("an@example.com")

    assert found is not None
    user, password_hash = found
    assert user.email == "an@example.com"
    assert password_hash == "hash"
    assert users.get_credentials_by_email("binh@example.com") is None
