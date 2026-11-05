import uuid
from datetime import UTC, datetime

import pytest

from kb_server.domain.credentials import (
    Principal,
    hash_token,
    looks_like_token,
    new_token,
)
from kb_server.domain.users import User, UserRole, normalize_email


def test_a_session_token_and_a_key_are_marked_differently() -> None:
    assert new_token("session").token.startswith("kb_s_")
    assert new_token("api_key").token.startswith("kb_k_")


def test_a_new_token_keeps_only_its_hash_and_a_short_prefix() -> None:
    issued = new_token("api_key")

    assert issued.prefix == issued.token[:12]
    assert issued.token_hash == hash_token(issued.token)
    assert issued.token not in issued.token_hash


def test_every_new_token_is_different() -> None:
    assert new_token("session").token != new_token("session").token


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("kb_s_abc", True),
        ("kb_k_abc", True),
        ("sk-abc", False),
        ("", False),
    ],
)
def test_looks_like_token(token: str, expected: bool) -> None:
    assert looks_like_token(token) is expected


def test_only_an_api_key_can_act_for_others() -> None:
    user = User(
        id=uuid.uuid4(),
        email="an@example.com",
        display_name="An",
        role="reader",
        is_active=True,
        created_at=datetime(2026, 11, 5, tzinfo=UTC),
    )

    with pytest.raises(ValueError):
        Principal(
            user=user,
            credential_id=uuid.uuid4(),
            credential_kind="session",
            acting_for_others=True,
        )


def test_emails_are_compared_trimmed_and_in_lower_case() -> None:
    assert normalize_email("  An@Example.COM ") == "an@example.com"


def principal_for(role: UserRole, *, acting_for_others: bool) -> Principal:
    return Principal(
        user=User(
            id=uuid.uuid4(),
            email="someone@example.com",
            display_name="Someone",
            role=role,
            is_active=True,
            created_at=datetime(2026, 11, 5, tzinfo=UTC),
        ),
        credential_id=uuid.uuid4(),
        credential_kind="api_key",
        acting_for_others=acting_for_others,
    )


def test_an_admin_acting_as_themselves_is_an_admin() -> None:
    assert principal_for("admin", acting_for_others=False).is_admin


def test_a_front_end_acting_for_an_admin_is_not_an_admin() -> None:
    assert not principal_for("admin", acting_for_others=True).is_admin


@pytest.mark.parametrize("role", ["reader", "manager"])
def test_other_roles_are_not_admins(role: UserRole) -> None:
    assert not principal_for(role, acting_for_others=False).is_admin
