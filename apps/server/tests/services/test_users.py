"""Creating and managing users, and changing your own password."""

import uuid
from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session

from kb_server.domain.credentials import Principal
from kb_server.domain.errors import (
    AdminExists,
    InvalidRequest,
    NotAllowed,
    UserNotFound,
)
from kb_server.domain.users import User
from kb_server.services.auth import AuthService
from kb_server.services.passwords import Passwords
from kb_server.services.users import UserService
from kb_server.storage.models import UserRow
from kb_server.storage.repositories.users import UserRepository
from tests.support import PASSWORD, deactivate, sign_in

NEW_PASSWORD = "another long password"


@pytest.fixture
def admin(auth: AuthService, make_user: Callable[..., User]) -> Principal:
    """An admin, logged in."""
    make_user("admin@example.com", role="admin")
    return sign_in(auth, "admin@example.com")[1]


@pytest.fixture
def reader(auth: AuthService, make_user: Callable[..., User]) -> Principal:
    """A reader, logged in."""
    make_user("an@example.com")
    return sign_in(auth, "an@example.com")[1]


@pytest.fixture
def as_admin_through_front_end(
    auth: AuthService, make_user: Callable[..., User], admin: Principal
) -> Principal:
    """A front-end key acting for the admin, e.g. the admin chatting in Open WebUI."""
    service_user = make_user("openwebui@example.com")
    issued = auth.create_front_end_key(
        admin, service_user_id=service_user.id, name="open webui"
    )
    principal = auth.resolve(issued.token, acting_for="admin@example.com")
    assert principal is not None
    return principal


def create(service: UserService, principal: Principal, email: str) -> User:
    return service.create_user(
        principal, email=email, display_name="New", role="reader", password=PASSWORD
    )


# the first admin


def test_the_first_admin_can_log_in(
    user_service: UserService, auth: AuthService
) -> None:
    user = user_service.create_first_admin(
        email="admin@example.com", display_name="Admin", password=PASSWORD
    )

    assert user.role == "admin"
    assert auth.authenticate("admin@example.com", PASSWORD) == user


def test_there_is_only_one_first_admin(user_service: UserService) -> None:
    user_service.create_first_admin(
        email="admin@example.com", display_name="Admin", password=PASSWORD
    )

    with pytest.raises(AdminExists):
        user_service.create_first_admin(
            email="other@example.com", display_name="Other", password=PASSWORD
        )


def test_an_inactive_admin_still_counts(
    user_service: UserService, session: Session
) -> None:
    first = user_service.create_first_admin(
        email="admin@example.com", display_name="Admin", password=PASSWORD
    )
    deactivate(session, first.id)

    with pytest.raises(AdminExists):
        user_service.create_first_admin(
            email="other@example.com", display_name="Other", password=PASSWORD
        )


# creating users


def test_an_admin_creates_a_user_who_can_log_in(
    user_service: UserService, auth: AuthService, admin: Principal
) -> None:
    user = user_service.create_user(
        admin,
        email="Binh@Example.com",
        display_name="Binh",
        role="manager",
        password=PASSWORD,
    )

    assert user.email == "binh@example.com"
    assert user.role == "manager"
    assert auth.authenticate("binh@example.com", PASSWORD) == user


def test_a_new_users_password_is_stored_hashed(
    user_service: UserService,
    session: Session,
    passwords: Passwords,
    admin: Principal,
) -> None:
    user = create(user_service, admin, "binh@example.com")

    row = session.get(UserRow, user.id)
    assert row is not None
    assert row.password_hash != PASSWORD
    assert passwords.verify(row.password_hash, PASSWORD)


def test_an_admin_can_use_their_own_api_key(
    user_service: UserService, auth: AuthService, admin: Principal
) -> None:
    key = auth.resolve(auth.create_key(admin, name="provisioning").token)
    assert key is not None

    assert create(user_service, key, "binh@example.com").email == "binh@example.com"


def test_only_an_admin_creates_users(
    user_service: UserService, reader: Principal
) -> None:
    with pytest.raises(NotAllowed):
        create(user_service, reader, "binh@example.com")


def test_a_front_end_key_cannot_manage_users_even_for_an_admin(
    user_service: UserService,
    make_user: Callable[..., User],
    as_admin_through_front_end: Principal,
) -> None:
    binh = make_user("binh@example.com")
    principal = as_admin_through_front_end

    with pytest.raises(NotAllowed):
        create(user_service, principal, "new@example.com")
    with pytest.raises(NotAllowed):
        user_service.list_users(principal)
    with pytest.raises(NotAllowed):
        user_service.get_user(principal, binh.id)
    with pytest.raises(NotAllowed):
        user_service.update_user(principal, binh.id, is_active=False)


# reading users


def test_an_admin_lists_users_by_email(
    user_service: UserService, make_user: Callable[..., User], admin: Principal
) -> None:
    make_user("chi@example.com")
    make_user("binh@example.com")

    listed = user_service.list_users(admin)

    assert [user.email for user in listed] == [
        "admin@example.com",
        "binh@example.com",
        "chi@example.com",
    ]


def test_an_admin_gets_a_user_by_id(
    user_service: UserService, make_user: Callable[..., User], admin: Principal
) -> None:
    binh = make_user("binh@example.com")

    assert user_service.get_user(admin, binh.id) == binh


def test_an_unknown_user_is_not_found(
    user_service: UserService, admin: Principal
) -> None:
    with pytest.raises(UserNotFound):
        user_service.get_user(admin, uuid.uuid4())
    with pytest.raises(UserNotFound):
        user_service.update_user(admin, uuid.uuid4(), display_name="Nobody")


def test_only_an_admin_reads_users(
    user_service: UserService, reader: Principal
) -> None:
    with pytest.raises(NotAllowed):
        user_service.list_users(reader)
    with pytest.raises(NotAllowed):
        user_service.get_user(reader, reader.user.id)


# updating users


def test_an_update_changes_only_the_given_fields_and_is_saved(
    user_service: UserService,
    users: UserRepository,
    make_user: Callable[..., User],
    admin: Principal,
) -> None:
    binh = make_user("binh@example.com")

    updated = user_service.update_user(admin, binh.id, role="manager")

    assert updated.role == "manager"
    assert updated.display_name == binh.display_name
    assert updated.is_active
    assert users.get(binh.id) == updated


def test_deactivating_a_user_ends_their_sessions_and_keys(
    user_service: UserService,
    auth: AuthService,
    make_user: Callable[..., User],
    admin: Principal,
) -> None:
    binh = make_user("binh@example.com")
    session_token, binh_session = sign_in(auth, "binh@example.com")
    key = auth.create_key(binh_session, name="script")

    updated = user_service.update_user(admin, binh.id, is_active=False)

    assert not updated.is_active
    assert auth.resolve(session_token) is None
    assert auth.resolve(key.token) is None
    assert auth.login("binh@example.com", PASSWORD) is None


def test_a_reactivated_user_can_log_in_again(
    user_service: UserService,
    auth: AuthService,
    make_user: Callable[..., User],
    admin: Principal,
) -> None:
    binh = make_user("binh@example.com")
    user_service.update_user(admin, binh.id, is_active=False)

    user_service.update_user(admin, binh.id, is_active=True)

    assert auth.login("binh@example.com", PASSWORD) is not None


def test_an_admin_can_demote_another_admin(
    user_service: UserService, make_user: Callable[..., User], admin: Principal
) -> None:
    other = make_user("other-admin@example.com", role="admin")

    assert user_service.update_user(admin, other.id, role="reader").role == "reader"


def test_an_admin_cannot_demote_or_deactivate_themselves(
    user_service: UserService, admin: Principal
) -> None:
    with pytest.raises(NotAllowed):
        user_service.update_user(admin, admin.user.id, role="reader")
    with pytest.raises(NotAllowed):
        user_service.update_user(admin, admin.user.id, is_active=False)


def test_an_admin_can_rename_themselves(
    user_service: UserService, admin: Principal
) -> None:
    updated = user_service.update_user(admin, admin.user.id, display_name="Boss")

    assert updated.display_name == "Boss"


def test_an_admin_may_resend_their_own_unchanged_role(
    user_service: UserService, admin: Principal
) -> None:
    # a client sending the whole user back changes nothing, so it isn't refused
    updated = user_service.update_user(
        admin, admin.user.id, role="admin", is_active=True, display_name="Boss"
    )

    assert updated.display_name == "Boss"


def test_only_an_admin_updates_users(
    user_service: UserService, make_user: Callable[..., User], reader: Principal
) -> None:
    binh = make_user("binh@example.com")

    with pytest.raises(NotAllowed):
        user_service.update_user(reader, binh.id, display_name="Hacked")
    with pytest.raises(NotAllowed):
        user_service.update_user(reader, reader.user.id, role="admin")


# changing your password


def test_the_new_password_replaces_the_old_one(
    user_service: UserService, auth: AuthService, reader: Principal
) -> None:
    user_service.change_password(reader, current=PASSWORD, new=NEW_PASSWORD)

    assert auth.authenticate("an@example.com", PASSWORD) is None
    assert auth.authenticate("an@example.com", NEW_PASSWORD) is not None


def test_changing_the_password_ends_only_the_other_sessions(
    user_service: UserService, auth: AuthService, make_user: Callable[..., User]
) -> None:
    make_user("an@example.com")
    phone_token, _ = sign_in(auth, "an@example.com")
    laptop_token, laptop = sign_in(auth, "an@example.com")
    key = auth.create_key(laptop, name="script")

    user_service.change_password(laptop, current=PASSWORD, new=NEW_PASSWORD)

    assert auth.resolve(phone_token) is None
    assert auth.resolve(laptop_token) is not None  # the session that changed it
    assert auth.resolve(key.token) is not None  # keys don't depend on the password


def test_changing_the_password_leaves_other_users_alone(
    user_service: UserService,
    auth: AuthService,
    make_user: Callable[..., User],
    reader: Principal,
) -> None:
    make_user("binh@example.com")
    binh_token, _ = sign_in(auth, "binh@example.com")

    user_service.change_password(reader, current=PASSWORD, new=NEW_PASSWORD)

    assert auth.resolve(binh_token) is not None


def test_the_current_password_must_be_right(
    user_service: UserService, auth: AuthService, reader: Principal
) -> None:
    with pytest.raises(InvalidRequest):
        user_service.change_password(reader, current="wrong password", new=NEW_PASSWORD)

    assert auth.authenticate("an@example.com", PASSWORD) is not None


def test_an_api_key_cannot_change_the_password(
    user_service: UserService, auth: AuthService, reader: Principal
) -> None:
    key = auth.resolve(auth.create_key(reader, name="script").token)
    assert key is not None

    with pytest.raises(NotAllowed):
        user_service.change_password(key, current=PASSWORD, new=NEW_PASSWORD)


def test_a_front_end_key_cannot_change_the_password(
    user_service: UserService, as_admin_through_front_end: Principal
) -> None:
    with pytest.raises(NotAllowed):
        user_service.change_password(
            as_admin_through_front_end, current=PASSWORD, new=NEW_PASSWORD
        )