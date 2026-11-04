"""API keys and front-end keys: creating, using, listing and revoking them."""

import uuid
from collections.abc import Callable
from datetime import timedelta

import pytest
from sqlalchemy.orm import Session

from kb_server.domain.credentials import Principal
from kb_server.domain.errors import CredentialNotFound, InvalidRequest, NotAllowed
from kb_server.domain.users import User
from kb_server.services.auth import AuthService
from tests.support import FakeClock, deactivate, sign_in


@pytest.fixture
def an(auth: AuthService, make_user: Callable[..., User]) -> Principal:
    """A reader, logged in."""
    make_user("an@example.com")
    return sign_in(auth, "an@example.com")[1]


@pytest.fixture
def admin(auth: AuthService, make_user: Callable[..., User]) -> Principal:
    """An admin, logged in."""
    make_user("admin@example.com", role="admin")
    return sign_in(auth, "admin@example.com")[1]


@pytest.fixture
def service_user(make_user: Callable[..., User]) -> User:
    """The user a front end such as Open WebUI holds its key as."""
    return make_user("openwebui@example.com")


def key_principal(auth: AuthService, token: str) -> Principal:
    principal = auth.resolve(token)
    assert principal is not None
    return principal


# ordinary keys


def test_a_key_resolves_to_its_owner(auth: AuthService, an: Principal) -> None:
    issued = auth.create_key(an, name="script")

    assert issued.token.startswith("kb_k_")
    assert issued.credential.name == "script"
    assert issued.credential.expires_at is None
    principal = key_principal(auth, issued.token)
    assert principal.user == an.user
    assert principal.credential_kind == "api_key"
    assert not principal.acting_for_others


def test_a_key_works_until_it_expires(
    auth: AuthService, an: Principal, clock: FakeClock
) -> None:
    issued = auth.create_key(
        an, name="script", expires_at=clock.now + timedelta(days=1)
    )

    clock.advance(timedelta(days=1))

    assert auth.resolve(issued.token) is None


def test_a_key_cannot_expire_in_the_past(
    auth: AuthService, an: Principal, clock: FakeClock
) -> None:
    with pytest.raises(InvalidRequest):
        auth.create_key(an, name="script", expires_at=clock.now)


def test_a_key_cannot_create_keys(auth: AuthService, an: Principal) -> None:
    key = key_principal(auth, auth.create_key(an, name="script").token)

    with pytest.raises(NotAllowed):
        auth.create_key(key, name="another")


def test_a_key_stops_working_when_its_owner_is_deactivated(
    auth: AuthService, session: Session, an: Principal
) -> None:
    issued = auth.create_key(an, name="script")

    deactivate(session, an.user.id)

    assert auth.resolve(issued.token) is None


def test_an_ordinary_key_cannot_act_for_someone_else(
    auth: AuthService, make_user: Callable[..., User], an: Principal
) -> None:
    make_user("binh@example.com")
    issued = auth.create_key(an, name="script")

    assert auth.resolve(issued.token, acting_for="binh@example.com") is None


def test_a_session_cannot_act_for_someone_else(
    auth: AuthService, make_user: Callable[..., User]
) -> None:
    make_user("an@example.com")
    make_user("binh@example.com")
    token, _ = sign_in(auth, "an@example.com")

    assert auth.resolve(token, acting_for="binh@example.com") is None


# front-end keys


def test_a_front_end_key_acts_for_the_user_it_names(
    auth: AuthService,
    make_user: Callable[..., User],
    admin: Principal,
    service_user: User,
) -> None:
    binh = make_user("binh@example.com")
    issued = auth.create_front_end_key(
        admin, service_user_id=service_user.id, name="open webui"
    )

    principal = auth.resolve(issued.token, acting_for="BINH@example.com")

    assert issued.credential.user_id == service_user.id
    assert issued.credential.act_for_users
    assert principal is not None
    assert principal.user == binh
    assert principal.acting_for_others


def test_a_front_end_key_must_name_a_user(
    auth: AuthService, admin: Principal, service_user: User
) -> None:
    issued = auth.create_front_end_key(
        admin, service_user_id=service_user.id, name="open webui"
    )

    assert auth.resolve(issued.token) is None


def test_a_front_end_key_cannot_act_for_an_unknown_or_inactive_user(
    auth: AuthService,
    session: Session,
    make_user: Callable[..., User],
    admin: Principal,
    service_user: User,
) -> None:
    binh = make_user("binh@example.com")
    deactivate(session, binh.id)
    issued = auth.create_front_end_key(
        admin, service_user_id=service_user.id, name="open webui"
    )

    assert auth.resolve(issued.token, acting_for="nobody@example.com") is None
    assert auth.resolve(issued.token, acting_for="binh@example.com") is None


def test_only_an_admin_can_create_a_front_end_key(
    auth: AuthService, an: Principal, service_user: User
) -> None:
    with pytest.raises(NotAllowed):
        auth.create_front_end_key(
            an, service_user_id=service_user.id, name="open webui"
        )


def test_an_admin_must_be_logged_in_to_create_a_front_end_key(
    auth: AuthService, admin: Principal, service_user: User
) -> None:
    admin_key = key_principal(auth, auth.create_key(admin, name="script").token)

    with pytest.raises(NotAllowed):
        auth.create_front_end_key(
            admin_key, service_user_id=service_user.id, name="open webui"
        )


def test_a_front_end_key_needs_a_real_service_user(
    auth: AuthService, admin: Principal
) -> None:
    with pytest.raises(InvalidRequest):
        auth.create_front_end_key(
            admin, service_user_id=uuid.uuid4(), name="open webui"
        )


def test_a_front_end_key_cannot_expire_in_the_past(
    auth: AuthService, admin: Principal, service_user: User, clock: FakeClock
) -> None:
    with pytest.raises(InvalidRequest):
        auth.create_front_end_key(
            admin,
            service_user_id=service_user.id,
            name="open webui",
            expires_at=clock.now - timedelta(days=1),
        )


def test_acting_for_a_user_does_not_allow_managing_their_keys(
    auth: AuthService,
    make_user: Callable[..., User],
    admin: Principal,
    service_user: User,
) -> None:
    make_user("binh@example.com")
    issued = auth.create_front_end_key(
        admin, service_user_id=service_user.id, name="open webui"
    )
    as_binh = auth.resolve(issued.token, acting_for="binh@example.com")
    assert as_binh is not None

    with pytest.raises(NotAllowed):
        auth.create_key(as_binh, name="sneaky")
    with pytest.raises(NotAllowed):
        auth.list_keys(as_binh)
    with pytest.raises(NotAllowed):
        auth.revoke_key(as_binh, issued.credential.id)


# listing and revoking


def test_list_keys_shows_only_the_owners_api_keys(
    auth: AuthService, make_user: Callable[..., User], an: Principal
) -> None:
    make_user("binh@example.com")
    _, binh = sign_in(auth, "binh@example.com")
    mine = auth.create_key(an, name="script")
    auth.create_key(binh, name="theirs")

    listed = auth.list_keys(an)

    # no sessions, no one else's keys
    assert [key.id for key in listed] == [mine.credential.id]


def test_a_revoked_key_stops_working(auth: AuthService, an: Principal) -> None:
    issued = auth.create_key(an, name="script")

    auth.revoke_key(an, issued.credential.id)

    assert auth.resolve(issued.token) is None
    (listed,) = auth.list_keys(an)
    assert listed.revoked_at is not None


def test_no_one_can_revoke_someone_elses_key(
    auth: AuthService, make_user: Callable[..., User], an: Principal
) -> None:
    make_user("binh@example.com")
    _, binh = sign_in(auth, "binh@example.com")
    issued = auth.create_key(an, name="script")

    with pytest.raises(CredentialNotFound):
        auth.revoke_key(binh, issued.credential.id)
    assert auth.resolve(issued.token) is not None


def test_revoking_an_unknown_or_revoked_key_is_not_found(
    auth: AuthService, an: Principal
) -> None:
    issued = auth.create_key(an, name="script")
    auth.revoke_key(an, issued.credential.id)

    with pytest.raises(CredentialNotFound):
        auth.revoke_key(an, issued.credential.id)
    with pytest.raises(CredentialNotFound):
        auth.revoke_key(an, uuid.uuid4())
