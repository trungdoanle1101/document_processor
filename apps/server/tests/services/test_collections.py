"""Collections: who may see, create, change and manage them."""

import uuid
from collections.abc import Callable

import pytest

from kb_server.domain.collections import CollectionDetails
from kb_server.domain.credentials import Principal, can_manage
from kb_server.domain.errors import (
    CollectionNotFound,
    CollectionSlugTaken,
    GroupInUse,
    InvalidRequest,
    NotAllowed,
    UserNotFound,
)
from kb_server.domain.groups import Group
from kb_server.domain.users import User, UserRole
from kb_server.services.auth import AuthService
from kb_server.services.collections import CollectionService
from kb_server.services.groups import GroupService
from kb_server.services.users import UserService
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.groups import GroupRepository
from tests.support import sign_in


def principal(
    auth: AuthService, make_user: Callable[..., User], email: str, role: UserRole
) -> Principal:
    make_user(email, role=role)
    return sign_in(auth, email)[1]


@pytest.fixture
def admin(auth: AuthService, make_user: Callable[..., User]) -> Principal:
    return principal(auth, make_user, "admin@example.com", "admin")


@pytest.fixture
def manager(auth: AuthService, make_user: Callable[..., User]) -> Principal:
    return principal(auth, make_user, "manager@example.com", "manager")


@pytest.fixture
def reader(auth: AuthService, make_user: Callable[..., User]) -> Principal:
    return principal(auth, make_user, "an@example.com", "reader")


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def it(groups: GroupRepository) -> Group:
    return groups.add("it")


@pytest.fixture
def hr_docs(
    collection_service: CollectionService, admin: Principal, hr: Group
) -> CollectionDetails:
    return collection_service.create_collection(
        admin, slug="hr-docs", title="HR documents", description="", group_ids=[hr.id]
    )


def front_end_for(
    auth: AuthService, make_user: Callable[..., User], admin: Principal, email: str
) -> Principal:
    service_user = make_user("openwebui@example.com")
    key = auth.create_front_end_key(admin, service_user_id=service_user.id, name="owui")
    acting = auth.resolve(key.token, acting_for=email)
    assert acting is not None
    return acting


# can_manage


def test_can_manage(
    auth: AuthService,
    make_user: Callable[..., User],
    admin: Principal,
    manager: Principal,
    reader: Principal,
) -> None:
    listed = {manager.user.id, reader.user.id}

    assert can_manage(admin, set())
    assert can_manage(manager, listed)
    assert not can_manage(manager, set())
    assert not can_manage(reader, listed)  # a reader's row doesn't count
    assert not can_manage(
        front_end_for(auth, make_user, admin, "manager@example.com"), listed
    )


# creating


def test_an_admin_creates_a_collection_with_its_groups(
    collection_service: CollectionService, admin: Principal, hr: Group, it: Group
) -> None:
    details = collection_service.create_collection(
        admin,
        slug="  HR-Docs ",
        title="Quy định nhân sự",
        description="Nội quy",
        group_ids=[it.id, hr.id, hr.id],
    )

    assert details.collection.slug == "hr-docs"
    assert details.groups == [hr, it]
    assert collection_service.get_collection(admin, details.collection.id) == details


def test_a_collection_can_start_with_no_groups(
    collection_service: CollectionService, admin: Principal
) -> None:
    details = collection_service.create_collection(
        admin, slug="drafts", title="Drafts", description="", group_ids=[]
    )

    assert details.groups == []


def test_unknown_groups_are_refused_and_nothing_is_created(
    collection_service: CollectionService, admin: Principal, hr: Group
) -> None:
    unknown = uuid.uuid4()

    with pytest.raises(InvalidRequest, match=str(unknown)):
        collection_service.create_collection(
            admin, slug="x", title="X", description="", group_ids=[hr.id, unknown]
        )

    assert collection_service.list_collections(admin) == []


def test_a_slug_can_only_be_used_once(
    collection_service: CollectionService, admin: Principal, hr_docs: CollectionDetails
) -> None:
    with pytest.raises(CollectionSlugTaken):
        collection_service.create_collection(
            admin, slug="HR-DOCS", title="Again", description="", group_ids=[]
        )


def test_only_an_admin_creates_collections(
    collection_service: CollectionService,
    auth: AuthService,
    make_user: Callable[..., User],
    admin: Principal,
    manager: Principal,
) -> None:
    for who in (manager, front_end_for(auth, make_user, admin, "admin@example.com")):
        with pytest.raises(NotAllowed):
            collection_service.create_collection(
                who, slug="x", title="X", description="", group_ids=[]
            )


# seeing


def test_managers_and_admins_see_every_collection(
    collection_service: CollectionService,
    admin: Principal,
    manager: Principal,
    hr_docs: CollectionDetails,
) -> None:
    private = collection_service.create_collection(
        admin, slug="private", title="Private", description="", group_ids=[]
    )

    for who in (admin, manager):
        assert collection_service.list_collections(who) == [hr_docs, private]
        assert collection_service.get_collection(who, private.collection.id) == private


def test_a_reader_sees_only_collections_sharing_a_group(
    collection_service: CollectionService,
    groups: GroupRepository,
    admin: Principal,
    reader: Principal,
    hr: Group,
    hr_docs: CollectionDetails,
) -> None:
    hidden_id = hr_docs.collection.id
    assert collection_service.list_collections(reader) == []
    with pytest.raises(CollectionNotFound):
        collection_service.get_collection(reader, hidden_id)

    groups.add_member(user_id=reader.user.id, group_id=hr.id)

    assert collection_service.list_collections(reader) == [hr_docs]
    assert collection_service.get_collection(reader, hidden_id) == hr_docs


def test_an_unknown_collection_is_not_found(
    collection_service: CollectionService, admin: Principal
) -> None:
    with pytest.raises(CollectionNotFound):
        collection_service.get_collection(admin, uuid.uuid4())


# updating and deleting


def test_an_update_changes_only_the_given_fields(
    collection_service: CollectionService, admin: Principal, hr_docs: CollectionDetails
) -> None:
    updated = collection_service.update_collection(
        admin, hr_docs.collection.id, title="Nhân sự"
    )

    assert updated.collection.title == "Nhân sự"
    assert updated.collection.slug == "hr-docs"
    assert updated.groups == hr_docs.groups
    assert collection_service.get_collection(admin, hr_docs.collection.id) == updated


def test_an_update_normalises_the_slug(
    collection_service: CollectionService, admin: Principal, hr_docs: CollectionDetails
) -> None:
    updated = collection_service.update_collection(
        admin, hr_docs.collection.id, slug=" HR "
    )

    assert updated.collection.slug == "hr"


def test_updating_to_a_taken_slug_is_refused(
    collection_service: CollectionService, admin: Principal, hr_docs: CollectionDetails
) -> None:
    collection_service.create_collection(
        admin, slug="it-docs", title="IT", description="", group_ids=[]
    )

    with pytest.raises(CollectionSlugTaken):
        collection_service.update_collection(
            admin, hr_docs.collection.id, slug="it-docs"
        )


def test_only_an_admin_updates_or_deletes(
    collection_service: CollectionService,
    manager: Principal,
    hr_docs: CollectionDetails,
) -> None:
    collection_id = hr_docs.collection.id

    with pytest.raises(NotAllowed):
        collection_service.update_collection(manager, collection_id, title="Mine")
    with pytest.raises(NotAllowed):
        collection_service.delete_collection(manager, collection_id)


def test_an_admin_deletes_a_collection(
    collection_service: CollectionService, admin: Principal, hr_docs: CollectionDetails
) -> None:
    collection_service.delete_collection(admin, hr_docs.collection.id)

    with pytest.raises(CollectionNotFound):
        collection_service.get_collection(admin, hr_docs.collection.id)
    with pytest.raises(CollectionNotFound):
        collection_service.delete_collection(admin, hr_docs.collection.id)


# default groups


def test_an_admin_replaces_the_default_groups(
    collection_service: CollectionService,
    admin: Principal,
    it: Group,
    hr_docs: CollectionDetails,
) -> None:
    updated = collection_service.set_groups(admin, hr_docs.collection.id, [it.id])

    assert updated.groups == [it]
    assert collection_service.get_collection(admin, hr_docs.collection.id) == updated


def test_a_manager_of_the_collection_can_set_its_groups(
    collection_service: CollectionService,
    admin: Principal,
    manager: Principal,
    hr_docs: CollectionDetails,
) -> None:
    collection_id = hr_docs.collection.id
    with pytest.raises(NotAllowed):
        collection_service.set_groups(manager, collection_id, [])

    collection_service.add_manager(
        admin, collection_id=collection_id, user_id=manager.user.id
    )

    assert collection_service.set_groups(manager, collection_id, []).groups == []


def test_a_reader_cannot_set_groups_even_when_listed_as_manager(
    collection_service: CollectionService,
    collections: CollectionRepository,
    groups: GroupRepository,
    reader: Principal,
    hr: Group,
    hr_docs: CollectionDetails,
) -> None:
    # e.g. a manager who was later demoted: the row stays but doesn't count
    groups.add_member(user_id=reader.user.id, group_id=hr.id)  # so they can see it
    collections.add_manager(collection_id=hr_docs.collection.id, user_id=reader.user.id)

    with pytest.raises(NotAllowed):
        collection_service.set_groups(reader, hr_docs.collection.id, [])


def test_set_groups_on_a_hidden_collection_is_not_found(
    collection_service: CollectionService, reader: Principal, hr_docs: CollectionDetails
) -> None:
    with pytest.raises(CollectionNotFound):
        collection_service.set_groups(reader, hr_docs.collection.id, [])


def test_set_groups_refuses_unknown_groups_and_keeps_the_old_ones(
    collection_service: CollectionService, admin: Principal, hr_docs: CollectionDetails
) -> None:
    with pytest.raises(InvalidRequest):
        collection_service.set_groups(admin, hr_docs.collection.id, [uuid.uuid4()])

    assert collection_service.get_collection(admin, hr_docs.collection.id) == hr_docs


def test_a_front_end_cannot_set_groups_even_for_an_admin(
    collection_service: CollectionService,
    auth: AuthService,
    make_user: Callable[..., User],
    admin: Principal,
    hr_docs: CollectionDetails,
) -> None:
    acting = front_end_for(auth, make_user, admin, "admin@example.com")

    with pytest.raises(NotAllowed):
        collection_service.set_groups(acting, hr_docs.collection.id, [])


# managers


def test_an_admin_adds_and_removes_managers(
    collection_service: CollectionService,
    admin: Principal,
    manager: Principal,
    hr_docs: CollectionDetails,
) -> None:
    collection_id = hr_docs.collection.id

    collection_service.add_manager(
        admin, collection_id=collection_id, user_id=manager.user.id
    )
    collection_service.add_manager(
        admin, collection_id=collection_id, user_id=manager.user.id
    )
    collection_service.add_manager(
        admin, collection_id=collection_id, user_id=admin.user.id
    )
    listed = collection_service.list_managers(admin, collection_id)
    assert [user.email for user in listed] == [
        "admin@example.com",
        "manager@example.com",
    ]

    collection_service.remove_manager(
        admin, collection_id=collection_id, user_id=manager.user.id
    )
    listed = collection_service.list_managers(admin, collection_id)
    assert [user.email for user in listed] == ["admin@example.com"]


def test_a_reader_cannot_be_made_a_manager(
    collection_service: CollectionService,
    admin: Principal,
    reader: Principal,
    hr_docs: CollectionDetails,
) -> None:
    with pytest.raises(InvalidRequest):
        collection_service.add_manager(
            admin, collection_id=hr_docs.collection.id, user_id=reader.user.id
        )


def test_unknown_collections_and_users_are_not_found(
    collection_service: CollectionService,
    admin: Principal,
    manager: Principal,
    hr_docs: CollectionDetails,
) -> None:
    unknown = uuid.uuid4()
    collection_id = hr_docs.collection.id

    with pytest.raises(CollectionNotFound):
        collection_service.add_manager(
            admin, collection_id=unknown, user_id=manager.user.id
        )
    with pytest.raises(CollectionNotFound):
        collection_service.remove_manager(
            admin, collection_id=unknown, user_id=manager.user.id
        )
    with pytest.raises(UserNotFound):
        collection_service.add_manager(
            admin, collection_id=collection_id, user_id=unknown
        )
    with pytest.raises(UserNotFound):
        collection_service.remove_manager(
            admin, collection_id=collection_id, user_id=unknown
        )


def test_only_an_admin_chooses_managers(
    collection_service: CollectionService,
    admin: Principal,
    manager: Principal,
    hr_docs: CollectionDetails,
) -> None:
    collection_id = hr_docs.collection.id
    collection_service.add_manager(
        admin, collection_id=collection_id, user_id=manager.user.id
    )

    with pytest.raises(NotAllowed):
        collection_service.add_manager(
            manager, collection_id=collection_id, user_id=manager.user.id
        )
    with pytest.raises(NotAllowed):
        collection_service.remove_manager(
            manager, collection_id=collection_id, user_id=manager.user.id
        )


def test_only_managers_and_admins_see_the_managers(
    collection_service: CollectionService,
    groups: GroupRepository,
    manager: Principal,
    reader: Principal,
    hr: Group,
    hr_docs: CollectionDetails,
) -> None:
    collection_id = hr_docs.collection.id
    assert collection_service.list_managers(manager, collection_id) == []

    with pytest.raises(CollectionNotFound):  # can't see it at all
        collection_service.list_managers(reader, collection_id)
    groups.add_member(user_id=reader.user.id, group_id=hr.id)
    with pytest.raises(NotAllowed):  # can see it, but not who manages it
        collection_service.list_managers(reader, collection_id)


def test_a_demoted_managers_row_no_longer_counts(
    collection_service: CollectionService,
    user_service: UserService,
    groups: GroupRepository,
    auth: AuthService,
    admin: Principal,
    manager: Principal,
    hr: Group,
    hr_docs: CollectionDetails,
) -> None:
    collection_id = hr_docs.collection.id
    groups.add_member(
        user_id=manager.user.id, group_id=hr.id
    )  # still sees it as a reader
    collection_service.add_manager(
        admin, collection_id=collection_id, user_id=manager.user.id
    )
    user_service.update_user(admin, manager.user.id, role="reader")
    _, demoted = sign_in(auth, "manager@example.com")

    with pytest.raises(NotAllowed):
        collection_service.set_groups(demoted, collection_id, [])
    # the row is still there for when they're promoted again
    assert [
        u.email for u in collection_service.list_managers(admin, collection_id)
    ] == ["manager@example.com"]


# groups in use


def test_a_group_used_by_a_collection_cannot_be_deleted(
    group_service: GroupService,
    collection_service: CollectionService,
    admin: Principal,
    hr: Group,
    hr_docs: CollectionDetails,
) -> None:
    with pytest.raises(GroupInUse):
        group_service.delete_group(admin, hr.id)

    collection_service.set_groups(admin, hr_docs.collection.id, [])
    group_service.delete_group(admin, hr.id)
