import hashlib
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kb_server.domain.collections import Collection
from kb_server.domain.documents import DOCUMENT_STATUSES, Document, DocumentStatus
from kb_server.domain.groups import Group
from kb_server.domain.users import User
from kb_server.storage.models import (
    DocumentGroupRow,
    DocumentRow,
    DocumentVersionRow,
    IndexingJobRow,
    StoredFileRow,
)
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.jobs import JobRepository


def stored_file(session: Session, content: bytes) -> uuid.UUID:
    """A stored_files row (added directly: the file store isn't part of these tests)."""
    sha256 = hashlib.sha256(content).hexdigest()
    row = StoredFileRow(
        sha256=sha256,
        size=len(content),
        media_type="text/plain",
        storage_key=f"k/{sha256}",
    )
    session.add(row)
    session.flush()
    return row.id


@pytest.fixture
def uploader(make_user: Callable[..., User]) -> User:
    return make_user("manager@example.com", role="manager")


@pytest.fixture
def hr_docs(collections: CollectionRepository) -> Collection:
    return collections.add(slug="hr-docs", title="HR", description="")


@pytest.fixture
def it_docs(collections: CollectionRepository) -> Collection:
    return collections.add(slug="it-docs", title="IT", description="")


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def it(groups: GroupRepository) -> Group:
    return groups.add("it")


@pytest.fixture
def add_document(
    session: Session,
    documents: DocumentRepository,
    uploader: User,
    hr_docs: Collection,
) -> Callable[..., Document]:
    """add_document(content=b"…", group_ids=[…], collection=…) -> a new document."""

    def add(
        *,
        content: bytes = b"Dieu 1",
        group_ids: tuple[uuid.UUID, ...] = (),
        collection: Collection | None = None,
        **overrides: Any,
    ) -> Document:
        document, _ = documents.add(
            **{
                "collection_id": (collection or hr_docs).id,
                "title": "Quy chế nhân sự",
                "status": "active",
                "group_ids": group_ids,
                "file_id": stored_file(session, content),
                "original_filename": "quy-che.docx",
                "created_by": uploader.id,
                **overrides,
            }
        )
        return document

    return add


# adding


def test_a_new_document_comes_with_version_1(
    session: Session, documents: DocumentRepository, uploader: User, hr_docs: Collection
) -> None:
    file_id = stored_file(session, b"Dieu 1")

    document, version = documents.add(
        collection_id=hr_docs.id,
        title="Quy chế nhân sự",
        status="active",
        group_ids=[],
        file_id=file_id,
        original_filename="quy-che.docx",
        created_by=uploader.id,
    )

    assert document.collection_id == hr_docs.id
    assert document.title == "Quy chế nhân sự"
    assert document.status == "active"
    assert document.current_version == 1
    assert document.created_by == uploader.id
    assert document.created_at.tzinfo is not None
    assert document.updated_at.tzinfo is not None

    assert version.document_id == document.id
    assert version.version == 1
    assert version.file_id == file_id
    assert version.original_filename == "quy-che.docx"
    assert version.uploaded_by == uploader.id
    assert version.provenance is None  # filled in by the worker
    assert version.created_at.tzinfo is not None


def test_a_new_document_reads_back_the_same(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document()

    assert documents.get(document.id) == document
    version = documents.get_version(document.id, 1)
    assert version is not None
    assert version.document_id == document.id


def test_a_draft_document_can_be_added(add_document: Callable[..., Document]) -> None:
    assert add_document(status="draft").status == "draft"


# compare_metadata ignores an Enum's values, so only a write shows that the CHECK allows them
@pytest.mark.parametrize("status", DOCUMENT_STATUSES)
def test_the_database_allows_every_document_status(
    add_document: Callable[..., Document], status: DocumentStatus
) -> None:
    assert add_document(status=status).status == status


def test_the_read_groups_are_saved_once_each(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    hr: Group,
    it: Group,
) -> None:
    document = add_document(group_ids=(hr.id, it.id, hr.id))

    assert documents.list_group_ids(document.id) == {hr.id, it.id}


def test_a_document_can_have_no_groups(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document()

    assert documents.list_group_ids(document.id) == set()


def test_each_document_gets_its_own_id(add_document: Callable[..., Document]) -> None:
    first = add_document(content=b"one")
    second = add_document(content=b"two")

    assert first.id != second.id


def test_two_documents_can_share_a_stored_file(
    session: Session,
    documents: DocumentRepository,
    uploader: User,
    hr_docs: Collection,
    it_docs: Collection,
) -> None:
    file_id = stored_file(session, b"the same regulation")

    first, first_version = documents.add(
        collection_id=hr_docs.id,
        title="A",
        status="active",
        group_ids=[],
        file_id=file_id,
        original_filename="a.docx",
        created_by=uploader.id,
    )
    second, second_version = documents.add(
        collection_id=it_docs.id,
        title="B",
        status="active",
        group_ids=[],
        file_id=file_id,
        original_filename="b.docx",
        created_by=uploader.id,
    )

    assert first.id != second.id
    assert first_version.file_id == second_version.file_id == file_id


# reading


def test_unknown_documents_and_versions_are_none(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document()

    assert documents.get(uuid.uuid4()) is None
    assert documents.get_version(document.id, 2) is None
    assert documents.get_version(uuid.uuid4(), 1) is None
    assert documents.list_group_ids(uuid.uuid4()) == set()


# finding a duplicate file


def test_find_current_with_file_finds_the_document(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    hr_docs: Collection,
) -> None:
    document = add_document(content=b"Dieu 1")

    found = documents.find_current_with_file(
        hr_docs.id, hashlib.sha256(b"Dieu 1").hexdigest()
    )

    assert found == document


def test_find_current_with_file_looks_only_in_the_collection(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    it_docs: Collection,
) -> None:
    add_document(content=b"Dieu 1")

    assert (
        documents.find_current_with_file(
            it_docs.id, hashlib.sha256(b"Dieu 1").hexdigest()
        )
        is None
    )


def test_find_current_with_file_ignores_other_files(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    hr_docs: Collection,
) -> None:
    add_document(content=b"Dieu 1")

    assert (
        documents.find_current_with_file(
            hr_docs.id, hashlib.sha256(b"other").hexdigest()
        )
        is None
    )


def test_find_current_with_file_skips_a_document_being_deleted(
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    hr_docs: Collection,
) -> None:
    add_document(content=b"Dieu 1", status="deleting")

    assert (
        documents.find_current_with_file(
            hr_docs.id, hashlib.sha256(b"Dieu 1").hexdigest()
        )
        is None
    )


def test_find_current_with_file_ignores_older_versions(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
    uploader: User,
    hr_docs: Collection,
) -> None:
    # replacing files comes later: add version 2 by hand and make it current
    document = add_document(content=b"version 1")
    session.add(
        DocumentVersionRow(
            document_id=document.id,
            version=2,
            file_id=stored_file(session, b"version 2"),
            original_filename="quy-che-v2.docx",
            uploaded_by=uploader.id,
        )
    )
    session.flush()
    session.execute(
        update(DocumentRow)
        .where(DocumentRow.id == document.id)
        .values(current_version=2)
    )

    old = hashlib.sha256(b"version 1").hexdigest()
    new = hashlib.sha256(b"version 2").hexdigest()
    assert documents.find_current_with_file(hr_docs.id, old) is None
    found = documents.find_current_with_file(hr_docs.id, new)
    assert found is not None
    assert found.id == document.id


# who can read


def test_a_document_is_visible_to_users_sharing_a_group(
    documents: DocumentRepository,
    groups: GroupRepository,
    add_document: Callable[..., Document],
    make_user: Callable[..., User],
    hr: Group,
    it: Group,
) -> None:
    an = make_user("an@example.com")
    binh = make_user("binh@example.com")
    groups.add_member(user_id=an.id, group_id=hr.id)
    groups.add_member(user_id=binh.id, group_id=it.id)
    document = add_document(group_ids=(hr.id,))

    assert documents.is_visible_to(document.id, an.id)
    assert not documents.is_visible_to(document.id, binh.id)


def test_a_document_without_groups_is_visible_to_no_one_by_groups(
    documents: DocumentRepository,
    groups: GroupRepository,
    add_document: Callable[..., Document],
    make_user: Callable[..., User],
    hr: Group,
) -> None:
    an = make_user("an@example.com")
    groups.add_member(user_id=an.id, group_id=hr.id)

    assert not documents.is_visible_to(add_document().id, an.id)


def test_leaving_the_group_hides_the_document(
    documents: DocumentRepository,
    groups: GroupRepository,
    add_document: Callable[..., Document],
    make_user: Callable[..., User],
    hr: Group,
) -> None:
    an = make_user("an@example.com")
    groups.add_member(user_id=an.id, group_id=hr.id)
    document = add_document(group_ids=(hr.id,))

    groups.remove_member(user_id=an.id, group_id=hr.id)

    assert not documents.is_visible_to(document.id, an.id)


# what the database refuses (the services check these first; this is the backstop)


def test_an_unknown_group_is_refused(
    session: Session, add_document: Callable[..., Document]
) -> None:
    with pytest.raises(IntegrityError), session.begin_nested():
        add_document(group_ids=(uuid.uuid4(),))


def test_a_group_used_by_a_document_is_in_use_and_cannot_be_deleted(
    session: Session,
    groups: GroupRepository,
    add_document: Callable[..., Document],
    hr: Group,
) -> None:
    add_document(group_ids=(hr.id,))

    assert groups.is_in_use(hr.id)
    with pytest.raises(IntegrityError), session.begin_nested():
        groups.delete(hr.id)


def test_a_collection_with_documents_cannot_be_deleted(
    session: Session,
    collections: CollectionRepository,
    add_document: Callable[..., Document],
    hr_docs: Collection,
) -> None:
    add_document()

    with pytest.raises(IntegrityError), session.begin_nested():
        collections.delete(hr_docs.id)


def test_a_stored_file_in_use_cannot_be_deleted(
    session: Session, add_document: Callable[..., Document]
) -> None:
    document = add_document()
    version = session.get(DocumentVersionRow, (document.id, 1))
    assert version is not None

    with pytest.raises(IntegrityError), session.begin_nested():
        session.delete(session.get(StoredFileRow, version.file_id))
        session.flush()


# deleting


def rows_for(session: Session, row_type: Any, document_id: uuid.UUID) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(row_type)
            .where(row_type.document_id == document_id)
        )
        or 0
    )


def test_mark_deleting_marks_an_active_document(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document()

    assert documents.mark_deleting(document.id) is True
    found = documents.get(document.id)
    assert found is not None and found.status == "deleting"


def test_mark_deleting_a_document_already_being_deleted_reports_false(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document(status="deleting")

    assert documents.mark_deleting(document.id) is False


def test_mark_deleting_an_unknown_document_reports_false(
    documents: DocumentRepository,
) -> None:
    assert documents.mark_deleting(uuid.uuid4()) is False


def test_delete_removes_the_document_its_versions_groups_and_jobs(
    session: Session,
    documents: DocumentRepository,
    jobs: JobRepository,
    add_document: Callable[..., Document],
    hr: Group,
) -> None:
    document = add_document(status="deleting", group_ids=(hr.id,))
    jobs.add(kind="delete", document_id=document.id, version=None, created_by=None)

    assert documents.delete(document.id) is True

    assert documents.get(document.id) is None
    for row_type in (DocumentVersionRow, DocumentGroupRow, IndexingJobRow):
        assert rows_for(session, row_type, document.id) == 0


def test_delete_keeps_the_stored_file_for_the_worker_to_decide(
    session: Session,
    documents: DocumentRepository,
    add_document: Callable[..., Document],
) -> None:
    document = add_document(status="deleting")

    documents.delete(document.id)

    assert session.scalar(select(func.count()).select_from(StoredFileRow)) == 1


def test_delete_refuses_a_document_not_marked_for_deletion(
    documents: DocumentRepository, add_document: Callable[..., Document]
) -> None:
    document = add_document()

    assert documents.delete(document.id) is False
    assert documents.get(document.id) == document


def test_delete_an_unknown_document_reports_false(
    documents: DocumentRepository,
) -> None:
    assert documents.delete(uuid.uuid4()) is False
