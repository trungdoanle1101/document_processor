from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session

from kb_server.domain.errors import GroupNameTaken
from kb_server.domain.groups import EVERYONE
from kb_server.domain.users import User
from kb_server.storage.models import UserRow
from kb_server.storage.repositories.groups import GroupRepository


def test_everyone_exists_from_the_start(groups: GroupRepository) -> None:
    everyone = groups.get_by_name(EVERYONE)

    assert everyone is not None
    assert [group.name for group in groups.list_all()] == [EVERYONE]


def test_a_new_group_is_saved_with_its_name_normalised(groups: GroupRepository) -> None:
    group = groups.add("  HR ")

    assert group.name == "hr"
    assert group.created_at.tzinfo is not None
    assert groups.get(group.id) == group
    assert groups.get_by_name("Hr") == group


def test_a_group_name_is_taken_whatever_its_case(groups: GroupRepository) -> None:
    groups.add("hr")

    with pytest.raises(GroupNameTaken):
        groups.add("HR")

    # the savepoint undid only the failed insert
    assert groups.get_by_name("hr") is not None


def test_groups_are_listed_by_name(groups: GroupRepository) -> None:
    groups.add("it")
    groups.add("hr")

    assert [group.name for group in groups.list_all()] == ["everyone", "hr", "it"]


def test_members_are_listed_by_email(
    groups: GroupRepository, make_user: Callable[..., User]
) -> None:
    hr = groups.add("hr")
    for email in ["chi@example.com", "an@example.com", "binh@example.com"]:
        groups.add_member(user_id=make_user(email).id, group_id=hr.id)

    members = groups.list_members(hr.id)

    assert [user.email for user in members] == [
        "an@example.com",
        "binh@example.com",
        "chi@example.com",
    ]


def test_adding_a_member_twice_is_harmless(
    groups: GroupRepository, make_user: Callable[..., User]
) -> None:
    hr = groups.add("hr")
    an = make_user("an@example.com")

    groups.add_member(user_id=an.id, group_id=hr.id)
    groups.add_member(user_id=an.id, group_id=hr.id)

    assert groups.list_members(hr.id) == [an]


def test_removing_a_member_leaves_the_others(
    groups: GroupRepository, make_user: Callable[..., User]
) -> None:
    hr = groups.add("hr")
    an = make_user("an@example.com")
    binh = make_user("binh@example.com")
    groups.add_member(user_id=an.id, group_id=hr.id)
    groups.add_member(user_id=binh.id, group_id=hr.id)

    groups.remove_member(user_id=an.id, group_id=hr.id)
    groups.remove_member(user_id=an.id, group_id=hr.id)  # not a member any more: fine

    assert groups.list_members(hr.id) == [binh]


def test_deleting_a_group_removes_its_memberships(
    groups: GroupRepository, make_user: Callable[..., User]
) -> None:
    hr = groups.add("hr")
    an = make_user("an@example.com")
    groups.add_member(user_id=an.id, group_id=hr.id)

    groups.delete(hr.id)

    assert groups.get(hr.id) is None
    assert groups.list_members(hr.id) == []


def test_deleting_a_user_removes_their_memberships(
    session: Session, groups: GroupRepository, make_user: Callable[..., User]
) -> None:
    # there's no route for this yet, but the foreign key cascades
    hr = groups.add("hr")
    an = make_user("an@example.com")
    groups.add_member(user_id=an.id, group_id=hr.id)

    session.delete(session.get(UserRow, an.id))
    session.flush()
    session.expire_all()

    assert groups.list_members(hr.id) == []
