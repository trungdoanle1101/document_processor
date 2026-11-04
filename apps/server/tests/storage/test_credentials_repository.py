import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import update
from sqlalchemy.orm import Session

from kb_server.domain.credentials import Credential, CredentialKind
from kb_server.domain.users import User
from kb_server.storage.models import CredentialRow
from kb_server.storage.repositories.credentials import CredentialRepository
from tests.support import deactivate

NOW = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)


def add(
    credentials: CredentialRepository,
    user: User,
    *,
    token_hash: str | None = None,
    kind: CredentialKind = "api_key",
    expires_at: datetime | None = None,
) -> Credential:
    return credentials.add(
        user_id=user.id,
        kind=kind,
        name="test key",
        prefix="kb_k_test",
        token_hash=token_hash or uuid.uuid4().hex,
        act_for_users=False,
        expires_at=expires_at,
    )


def test_a_new_credential_is_saved_unused_and_not_revoked(
    credentials: CredentialRepository, make_user: Callable[..., User]
) -> None:
    user = make_user("an@example.com")

    credential = add(credentials, user)

    assert credential.user_id == user.id
    assert credential.last_used_at is None
    assert credential.revoked_at is None
    assert not credential.act_for_users


def test_find_usable_returns_the_credential_and_its_owner(
    credentials: CredentialRepository, make_user: Callable[..., User]
) -> None:
    user = make_user("an@example.com")
    credential = add(credentials, user, token_hash="h1")

    found = credentials.find_usable("h1", now=NOW)

    assert found == (credential, user)


def test_find_usable_ignores_an_unknown_hash(
    credentials: CredentialRepository, make_user: Callable[..., User]
) -> None:
    add(credentials, make_user("an@example.com"), token_hash="h1")

    assert credentials.find_usable("other", now=NOW) is None


def test_a_credential_is_usable_until_it_expires(
    credentials: CredentialRepository, make_user: Callable[..., User]
) -> None:
    add(credentials, make_user("an@example.com"), token_hash="h1", expires_at=NOW)

    assert credentials.find_usable("h1", now=NOW - timedelta(seconds=1)) is not None
    assert credentials.find_usable("h1", now=NOW) is None


def test_a_revoked_credential_is_not_usable(
    credentials: CredentialRepository, make_user: Callable[..., User]
) -> None:
    user = make_user("an@example.com")
    credential = add(credentials, user, token_hash="h1")

    credentials.revoke(credential.id, owner_id=user.id, now=NOW)

    assert credentials.find_usable("h1", now=NOW) is None


def test_an_inactive_users_credential_is_not_usable(
    session: Session,
    credentials: CredentialRepository,
    make_user: Callable[..., User],
) -> None:
    user = make_user("an@example.com")
    add(credentials, user, token_hash="h1")

    deactivate(session, user.id)

    assert credentials.find_usable("h1", now=NOW) is None


def test_touch_records_the_last_use(
    credentials: CredentialRepository, make_user: Callable[..., User]
) -> None:
    credential = add(credentials, make_user("an@example.com"), token_hash="h1")

    credentials.touch(credential.id, now=NOW)

    found = credentials.find_usable("h1", now=NOW)
    assert found is not None
    assert found[0].last_used_at == NOW


def test_only_the_owner_can_revoke_and_only_once(
    credentials: CredentialRepository, make_user: Callable[..., User]
) -> None:
    owner = make_user("an@example.com")
    other = make_user("binh@example.com")
    credential = add(credentials, owner)

    assert not credentials.revoke(credential.id, owner_id=other.id, now=NOW)
    assert credentials.revoke(credential.id, owner_id=owner.id, now=NOW)
    assert not credentials.revoke(credential.id, owner_id=owner.id, now=NOW)
    assert not credentials.revoke(uuid.uuid4(), owner_id=owner.id, now=NOW)


def test_revoke_all_revokes_only_what_is_still_live(
    credentials: CredentialRepository, make_user: Callable[..., User]
) -> None:
    user = make_user("an@example.com")
    other = make_user("binh@example.com")
    already = add(credentials, user)
    add(credentials, user)
    add(credentials, user, kind="session")
    add(credentials, other, token_hash="h-other")
    credentials.revoke(already.id, owner_id=user.id, now=NOW)

    assert credentials.revoke_all(user.id, now=NOW) == 2
    assert credentials.find_usable("h-other", now=NOW) is not None


def test_list_for_user_returns_their_credentials_of_one_kind_newest_first(
    session: Session,
    credentials: CredentialRepository,
    make_user: Callable[..., User],
) -> None:
    user = make_user("an@example.com")
    older = add(credentials, user)
    newer = add(credentials, user)
    add(credentials, user, kind="session")
    add(credentials, make_user("binh@example.com"))
    # created_at comes from the database's now(), which is the same for every row in
    # a transaction, and a test is one transaction: set it to tell the keys apart
    for credential, created_at in [(older, NOW - timedelta(days=1)), (newer, NOW)]:
        session.execute(
            update(CredentialRow)
            .where(CredentialRow.id == credential.id)
            .values(created_at=created_at)
        )

    listed = credentials.list_for_user(user.id, kind="api_key")

    assert [credential.id for credential in listed] == [newer.id, older.id]
