"""Managing groups and their members."""

import uuid
from collections.abc import Callable

import pytest

from kb_server.domain.credentials import Principal
from kb_server.domain.errors import (
    GroupNameTaken,
    GroupNotFound,
    NotAllowed,
    UserNotFound,
)
from kb_server.domain.groups import EVERYONE, Group
from kb_server.domain.users import User
from kb_server.services.auth import AuthService
from kb_server.services.groups import GroupService
from kb_server.services.users import UserService
from kb_server.storage.repositories.groups import GroupRepository
from tests.support import PASSWORD, sign_in


@pytest.fixture
def admin(auth: AuthService, make_user: Callable[..., User]) -> Principal:
    make_user("admin@example.com", role="admin")
    return sign_in(auth, "admin@example.com")[1]


@pytest.fixture
def reader(auth: AuthService, make_user: Callable[..., User]) -> Principal:
    make_user("an@example.com")
    return sign_in(auth, "an@example.com")[1]


@pytest.fixture
def hr(group_service: GroupService, admin: Principal) -> Group:
    return group_service.create_group(admin, "hr")


@pytest.fixture
def everyone(groups: GroupRepository) -> Group:
    group = groups.get_by_name(EVERYONE)
    assert group is not None
    return group


@pytest.fixture
def binh(user_service: UserService, admin: Principal) -> User:
    """A user created the normal way, so a member of everyone."""
    return user_service.create_user(
        admin,
        email="binh@example.com",
        display_name="Binh",
        role="reader",
        password=PASSWORD,
    )


@pytest.fixture
def as_admin_through_front_end(
    auth: AuthService, make_user: Callable[..., User], admin: Principal
) -> Principal:
    service_user = make_user("openwebui@example.com")
    issued = auth.create_front_end_key(
        admin, service_user_id=service_user.id, name="open webui"
    )
    principal = auth.resolve(issued.token, acting_for="admin@example.com")
    assert principal is not None
    return principal


# groups


def test_an_admin_creates_and_lists_groups(
    group_service: GroupService, admin: Principal
) -> None:
    group_service.create_group(admin, "it")
    group_service.create_group(admin, "hr")

    names = [group.name for group in group_service.list_groups(admin)]

    assert names == [EVERYONE, "hr", "it"]


def test_a_group_name_can_only_be_used_once(
    group_service: GroupService, admin: Principal, hr: Group
) -> None:
    with pytest.raises(GroupNameTaken):
        group_service.create_group(admin, "HR")


def test_an_admin_deletes_a_group(
    group_service: GroupService, admin: Principal, hr: Group
) -> None:
    group_service.delete_group(admin, hr.id)

    assert [group.name for group in group_service.list_groups(admin)] == [EVERYONE]


def test_everyone_cannot_be_deleted(
    group_service: GroupService, admin: Principal, everyone: Group
) -> None:
    with pytest.raises(NotAllowed):
        group_service.delete_group(admin, everyone.id)

    assert group_service.list_groups(admin) == [everyone]


def test_an_unknown_group_is_not_found(
    group_service: GroupService, admin: Principal, reader: Principal
) -> None:
    unknown = uuid.uuid4()

    with pytest.raises(GroupNotFound):
        group_service.delete_group(admin, unknown)
    with pytest.raises(GroupNotFound):
        group_service.list_members(admin, unknown)
    with pytest.raises(GroupNotFound):
        group_service.add_member(admin, user_id=reader.user.id, group_id=unknown)
    with pytest.raises(GroupNotFound):
        group_service.remove_member(admin, user_id=reader.user.id, group_id=unknown)


# members


def test_an_admin_adds_and_removes_a_member(
    group_service: GroupService, admin: Principal, reader: Principal, hr: Group
) -> None:
    group_service.add_member(admin, user_id=reader.user.id, group_id=hr.id)
    assert group_service.list_members(admin, hr.id) == [reader.user]

    group_service.remove_member(admin, user_id=reader.user.id, group_id=hr.id)
    assert group_service.list_members(admin, hr.id) == []


def test_adding_and_removing_again_is_harmless(
    group_service: GroupService, admin: Principal, reader: Principal, hr: Group
) -> None:
    group_service.add_member(admin, user_id=reader.user.id, group_id=hr.id)
    group_service.add_member(admin, user_id=reader.user.id, group_id=hr.id)
    assert group_service.list_members(admin, hr.id) == [reader.user]

    group_service.remove_member(admin, user_id=reader.user.id, group_id=hr.id)
    group_service.remove_member(admin, user_id=reader.user.id, group_id=hr.id)
    assert group_service.list_members(admin, hr.id) == []


def test_an_unknown_user_is_not_found(
    group_service: GroupService, admin: Principal, hr: Group
) -> None:
    with pytest.raises(UserNotFound):
        group_service.add_member(admin, user_id=uuid.uuid4(), group_id=hr.id)
    with pytest.raises(UserNotFound):
        group_service.remove_member(admin, user_id=uuid.uuid4(), group_id=hr.id)


def test_the_group_is_checked_before_the_user(
    group_service: GroupService, admin: Principal
) -> None:
    with pytest.raises(GroupNotFound):
        group_service.add_member(admin, user_id=uuid.uuid4(), group_id=uuid.uuid4())


def test_everyone_cannot_lose_members(
    group_service: GroupService, admin: Principal, binh: User, everyone: Group
) -> None:
    with pytest.raises(NotAllowed):
        group_service.remove_member(admin, user_id=binh.id, group_id=everyone.id)

    assert binh in group_service.list_members(admin, everyone.id)


# who may


def test_only_an_admin_manages_groups(
    group_service: GroupService, reader: Principal, hr: Group
) -> None:
    with pytest.raises(NotAllowed):
        group_service.create_group(reader, "it")
    with pytest.raises(NotAllowed):
        group_service.list_groups(reader)
    with pytest.raises(NotAllowed):
        group_service.delete_group(reader, hr.id)
    with pytest.raises(NotAllowed):
        group_service.list_members(reader, hr.id)
    with pytest.raises(NotAllowed):
        group_service.add_member(reader, user_id=reader.user.id, group_id=hr.id)
    with pytest.raises(NotAllowed):
        group_service.remove_member(reader, user_id=reader.user.id, group_id=hr.id)


def test_a_front_end_acting_for_an_admin_cannot_manage_groups(
    group_service: GroupService, as_admin_through_front_end: Principal, hr: Group
) -> None:
    principal = as_admin_through_front_end

    with pytest.raises(NotAllowed):
        group_service.create_group(principal, "it")
    with pytest.raises(NotAllowed):
        group_service.add_member(principal, user_id=principal.user.id, group_id=hr.id)
