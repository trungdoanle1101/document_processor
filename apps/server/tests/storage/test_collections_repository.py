import uuid
from collections.abc import Callable

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kb_server.domain.errors import CollectionSlugTaken
from kb_server.domain.groups import Group
from kb_server.domain.users import User
from kb_server.storage.models import UserRow
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.groups import GroupRepository


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def it(groups: GroupRepository) -> Group:
    return groups.add("it")


def add(collections: CollectionRepository, slug: str = "hr-docs") -> uuid.UUID:
    return collections.add(slug=slug, title="HR documents", description="").id


# the collection itself


def test_a_new_collection_is_saved(collections: CollectionRepository) -> None:
    collection = collections.add(
        slug="hr-docs", title="Quy định nhân sự", description="Nội quy"
    )

    assert collections.get(collection.id) == collection
    assert collections.get_by_slug("hr-docs") == collection
    assert collection.created_at.tzinfo is not None


def test_a_slug_can_only_be_used_once(collections: CollectionRepository) -> None:
    add(collections, "hr-docs")

    with pytest.raises(CollectionSlugTaken):
        add(collections, "hr-docs")

    assert collections.get_by_slug("hr-docs") is not None


def test_an_update_saves_all_three_fields(collections: CollectionRepository) -> None:
    collection_id = add(collections)

    collections.update(collection_id, slug="hr", title="HR", description="People")

    updated = collections.get(collection_id)
    assert updated is not None
    assert (updated.slug, updated.title, updated.description) == ("hr", "HR", "People")


def test_updating_to_a_taken_slug_is_refused_and_changes_nothing(
    collections: CollectionRepository,
) -> None:
    add(collections, "it-docs")
    collection_id = add(collections, "hr-docs")

    with pytest.raises(CollectionSlugTaken):
        collections.update(collection_id, slug="it-docs", title="x", description="x")

    unchanged = collections.get(collection_id)
    assert unchanged is not None
    assert unchanged.slug == "hr-docs"


def test_collections_are_listed_by_slug(collections: CollectionRepository) -> None:
    add(collections, "it-docs")
    add(collections, "hr-docs")

    assert [c.slug for c in collections.list_all()] == ["hr-docs", "it-docs"]


def test_deleting_a_collection_removes_its_links_but_not_groups_or_users(
    collections: CollectionRepository,
    groups: GroupRepository,
    make_user: Callable[..., User],
    hr: Group,
) -> None:
    collection_id = add(collections)
    manager = make_user("m@example.com", role="manager")
    collections.set_groups(collection_id, [hr.id])
    collections.add_manager(collection_id=collection_id, user_id=manager.id)

    collections.delete(collection_id)

    assert collections.get(collection_id) is None
    assert collections.list_groups(collection_id) == []
    assert collections.list_managers(collection_id) == []
    assert groups.get(hr.id) == hr
    assert not groups.is_in_use(hr.id)


# default groups


def test_set_groups_replaces_the_whole_set(
    collections: CollectionRepository, hr: Group, it: Group
) -> None:
    collection_id = add(collections)
    collections.set_groups(collection_id, [hr.id])

    collections.set_groups(collection_id, [it.id, it.id])

    assert collections.list_groups(collection_id) == [it]


def test_set_groups_can_empty_the_set(
    collections: CollectionRepository, hr: Group
) -> None:
    collection_id = add(collections)
    collections.set_groups(collection_id, [hr.id])

    collections.set_groups(collection_id, [])

    assert collections.list_groups(collection_id) == []


def test_groups_are_listed_by_name(
    collections: CollectionRepository, hr: Group, it: Group
) -> None:
    collection_id = add(collections)

    collections.set_groups(collection_id, [it.id, hr.id])

    assert collections.list_groups(collection_id) == [hr, it]


def test_groups_by_collection_covers_several_collections_at_once(
    collections: CollectionRepository, hr: Group, it: Group
) -> None:
    first = add(collections, "first")
    second = add(collections, "second")
    empty = add(collections, "empty")
    collections.set_groups(first, [hr.id, it.id])
    collections.set_groups(second, [it.id])

    found = collections.groups_by_collection([first, second, empty])

    assert found == {first: [hr, it], second: [it]}  # no entry for the empty one


def test_a_group_in_use_cannot_be_deleted_by_the_database_either(
    session: Session,
    collections: CollectionRepository,
    groups: GroupRepository,
    hr: Group,
) -> None:
    collections.set_groups(add(collections), [hr.id])

    assert groups.is_in_use(hr.id)
    with pytest.raises(IntegrityError), session.begin_nested():
        groups.delete(hr.id)


# visibility


def test_a_collection_is_visible_to_users_sharing_a_default_group(
    collections: CollectionRepository,
    groups: GroupRepository,
    make_user: Callable[..., User],
    hr: Group,
    it: Group,
) -> None:
    an = make_user("an@example.com")
    binh = make_user("binh@example.com")
    groups.add_member(user_id=an.id, group_id=hr.id)
    groups.add_member(user_id=binh.id, group_id=it.id)
    hr_docs = add(collections, "hr-docs")
    collections.set_groups(hr_docs, [hr.id])

    assert collections.is_visible_to(hr_docs, an.id)
    assert not collections.is_visible_to(hr_docs, binh.id)


def test_a_collection_without_groups_is_visible_to_no_one_by_groups(
    collections: CollectionRepository,
    groups: GroupRepository,
    make_user: Callable[..., User],
    hr: Group,
) -> None:
    an = make_user("an@example.com")
    groups.add_member(user_id=an.id, group_id=hr.id)

    assert not collections.is_visible_to(add(collections), an.id)


def test_list_visible_to_shows_only_shared_collections_once_each(
    collections: CollectionRepository,
    groups: GroupRepository,
    make_user: Callable[..., User],
    hr: Group,
    it: Group,
) -> None:
    an = make_user("an@example.com")
    groups.add_member(user_id=an.id, group_id=hr.id)
    groups.add_member(user_id=an.id, group_id=it.id)
    both = add(collections, "both")
    collections.set_groups(both, [hr.id, it.id])  # two shared groups: still listed once
    collections.set_groups(add(collections, "hr-only"), [hr.id])
    add(collections, "private")

    visible = collections.list_visible_to(an.id)

    assert [c.slug for c in visible] == ["both", "hr-only"]


# managers


def test_managers_are_listed_by_email_and_adding_twice_is_harmless(
    collections: CollectionRepository, make_user: Callable[..., User]
) -> None:
    collection_id = add(collections)
    chi = make_user("chi@example.com", role="manager")
    binh = make_user("binh@example.com", role="manager")

    for user in (chi, binh, chi):
        collections.add_manager(collection_id=collection_id, user_id=user.id)

    assert collections.list_managers(collection_id) == [binh, chi]
    assert collections.list_manager_ids(collection_id) == {binh.id, chi.id}


def test_removing_a_manager_leaves_the_others(
    collections: CollectionRepository, make_user: Callable[..., User]
) -> None:
    collection_id = add(collections)
    chi = make_user("chi@example.com", role="manager")
    binh = make_user("binh@example.com", role="manager")
    collections.add_manager(collection_id=collection_id, user_id=chi.id)
    collections.add_manager(collection_id=collection_id, user_id=binh.id)

    collections.remove_manager(collection_id=collection_id, user_id=chi.id)
    collections.remove_manager(collection_id=collection_id, user_id=chi.id)

    assert collections.list_managers(collection_id) == [binh]


def test_deleting_a_user_removes_them_as_manager(
    session: Session, collections: CollectionRepository, make_user: Callable[..., User]
) -> None:
    collection_id = add(collections)
    manager = make_user("m@example.com", role="manager")
    collections.add_manager(collection_id=collection_id, user_id=manager.id)

    session.delete(session.get(UserRow, manager.id))
    session.flush()

    assert collections.list_manager_ids(collection_id) == set()
