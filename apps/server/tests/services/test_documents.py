"""Uploading and reading documents."""

import hashlib
import io
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from kb_server.domain.collections import Collection
from kb_server.domain.credentials import Principal
from kb_server.domain.documents import DocumentDetails
from kb_server.domain.errors import (
    CollectionNotFound,
    DocumentNotFound,
    DuplicateFile,
    FileTooLarge,
    InvalidRequest,
    NotAllowed,
    UnsupportedFileType,
)
from kb_server.domain.files import DOCX, TEXT
from kb_server.domain.groups import Group
from kb_server.domain.users import User, UserRole
from kb_server.services.auth import AuthService
from kb_server.services.documents import DocumentService
from kb_server.storage.files import FileStore, original_key
from kb_server.storage.models import DocumentRow, IndexingJobRow, StoredFileRow
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.groups import GroupRepository
from tests.support import MAX_UPLOAD_MB, sign_in

CONTENT = "Điều 1. Phạm vi điều chỉnh\n".encode()


# people and places


@pytest.fixture
def login(
    auth: AuthService, make_user: Callable[..., User]
) -> Callable[[str, UserRole], Principal]:
    def make(email: str, role: UserRole) -> Principal:
        make_user(email, role=role)
        return sign_in(auth, email)[1]

    return make


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def it(groups: GroupRepository) -> Group:
    return groups.add("it")


@pytest.fixture
def hr_docs(collections: CollectionRepository, hr: Group) -> Collection:
    collection = collections.add(slug="hr-docs", title="HR documents", description="")
    collections.set_groups(collection.id, [hr.id])
    return collection


@pytest.fixture
def it_docs(collections: CollectionRepository, it: Group) -> Collection:
    collection = collections.add(slug="it-docs", title="IT documents", description="")
    collections.set_groups(collection.id, [it.id])
    return collection


@pytest.fixture
def admin(login: Callable[[str, UserRole], Principal]) -> Principal:
    return login("admin@example.com", "admin")


@pytest.fixture
def manager(
    login: Callable[[str, UserRole], Principal],
    collections: CollectionRepository,
    hr_docs: Collection,
) -> Principal:
    """A manager of hr-docs."""
    principal = login("manager@example.com", "manager")
    collections.add_manager(collection_id=hr_docs.id, user_id=principal.user.id)
    return principal


@pytest.fixture
def other_manager(login: Callable[[str, UserRole], Principal]) -> Principal:
    """A manager, but not of hr-docs."""
    return login("other-manager@example.com", "manager")


@pytest.fixture
def hr_reader(
    login: Callable[[str, UserRole], Principal], groups: GroupRepository, hr: Group
) -> Principal:
    principal = login("an@example.com", "reader")
    groups.add_member(user_id=principal.user.id, group_id=hr.id)
    return principal


@pytest.fixture
def outsider(login: Callable[[str, UserRole], Principal]) -> Principal:
    """A reader in no group that matters here."""
    return login("binh@example.com", "reader")


@pytest.fixture
def upload(
    document_service: DocumentService, manager: Principal, hr_docs: Collection
) -> Callable[..., DocumentDetails]:
    """upload(content=…, filename=…, as_=…, **fields): upload with sensible defaults."""

    def run(
        *,
        content: bytes = CONTENT,
        filename: str = "quy-che.txt",
        as_: Principal | None = None,
        **fields: Any,
    ) -> DocumentDetails:
        values: dict[str, Any] = {
            "collection_id": hr_docs.id,
            "title": None,
            "group_ids": None,
            "private": False,
            **fields,
        }
        return document_service.upload_document(
            as_ or manager, content=io.BytesIO(content), filename=filename, **values
        )

    return run


def front_end_acting_for(
    auth: AuthService,
    make_user: Callable[..., User],
    admin: Principal,
    email: str,
) -> Principal:
    service_user = make_user("openwebui@example.com")
    key = auth.create_front_end_key(admin, service_user_id=service_user.id, name="owui")
    principal = auth.resolve(key.token, acting_for=email)
    assert principal is not None
    return principal


def docx_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<x/>")
        archive.writestr("word/document.xml", "<x/>")
    return buffer.getvalue()


def count(session: Session, row_type: type[Any]) -> int:
    return session.scalar(select(func.count()).select_from(row_type)) or 0


def stored_paths(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


def mark_deleting(session: Session, document_id: uuid.UUID) -> None:
    """Set the status by hand: what DELETE /documents/{id} will do."""
    session.execute(
        update(DocumentRow)
        .where(DocumentRow.id == document_id)
        .values(status="deleting")
    )


# a successful upload


def test_an_upload_creates_the_document_its_first_version_and_a_job(
    upload: Callable[..., DocumentDetails], manager: Principal, hr_docs: Collection
) -> None:
    details = upload()

    document = details.document
    assert document.collection_id == hr_docs.id
    assert document.status == "active"
    assert document.current_version == 1
    assert document.created_by == manager.user.id

    assert details.version.document_id == document.id
    assert details.version.version == 1
    assert details.version.original_filename == "quy-che.txt"
    assert details.version.uploaded_by == manager.user.id
    assert details.version.file_id == details.file.id

    assert details.file.sha256 == hashlib.sha256(CONTENT).hexdigest()
    assert details.file.size == len(CONTENT)
    assert details.file.media_type == TEXT

    job = details.job
    assert job is not None
    assert (job.kind, job.status, job.version) == ("prepare", "queued", 1)
    assert job.document_id == document.id
    assert job.created_by == manager.user.id


def test_the_file_is_in_the_store_under_its_key(
    upload: Callable[..., DocumentDetails], file_store: FileStore
) -> None:
    details = upload()

    assert details.file.storage_key == original_key(details.file.sha256)
    with file_store.open(details.file.storage_key) as stored:
        assert stored.read() == CONTENT


def test_a_docx_is_recognised(upload: Callable[..., DocumentDetails]) -> None:
    details = upload(content=docx_bytes(), filename="quy-che.docx")

    assert details.file.media_type == DOCX


def test_an_admin_can_upload_to_any_collection(
    upload: Callable[..., DocumentDetails], admin: Principal
) -> None:
    assert upload(as_=admin).document.created_by == admin.user.id


# title and filename


def test_the_title_defaults_to_the_filename_without_its_extension(
    upload: Callable[..., DocumentDetails],
) -> None:
    assert upload(
        filename="Quy chế nhân sự.docx", content=docx_bytes()
    ).document.title == ("Quy chế nhân sự")


def test_a_given_title_is_used(upload: Callable[..., DocumentDetails]) -> None:
    assert upload(title="Quy chế 2026").document.title == "Quy chế 2026"


@pytest.mark.parametrize(
    "sent",
    ["C:\\Users\\an\\quy-che.txt", "/home/an/quy-che.txt", "../../quy-che.txt"],
    ids=["windows path", "posix path", "dots"],
)
def test_only_the_last_part_of_the_filename_is_kept(
    upload: Callable[..., DocumentDetails], sent: str
) -> None:
    details = upload(filename=sent)

    assert details.version.original_filename == "quy-che.txt"
    assert details.document.title == "quy-che"


def test_a_long_filename_is_shortened_keeping_its_extension(
    upload: Callable[..., DocumentDetails],
) -> None:
    details = upload(filename="x" * 600 + ".txt")

    assert len(details.version.original_filename) == 255
    assert details.version.original_filename.endswith(".txt")
    assert len(details.document.title) <= 255


# read groups


def test_the_collections_groups_are_copied_by_default(
    upload: Callable[..., DocumentDetails], hr: Group
) -> None:
    assert upload().groups == [hr]


def test_private_means_no_groups(upload: Callable[..., DocumentDetails]) -> None:
    assert upload(private=True).groups == []


def test_given_groups_are_used_once_each(
    upload: Callable[..., DocumentDetails], hr: Group, it: Group
) -> None:
    assert upload(group_ids=[it.id, it.id]).groups == [it]


def test_private_and_group_ids_together_are_refused(
    upload: Callable[..., DocumentDetails], session: Session, hr: Group
) -> None:
    with pytest.raises(InvalidRequest):
        upload(private=True, group_ids=[hr.id])

    assert count(session, DocumentRow) == 0


def test_unknown_groups_are_refused_and_nothing_is_created(
    upload: Callable[..., DocumentDetails], session: Session, file_store_root: Path
) -> None:
    with pytest.raises(InvalidRequest):
        upload(group_ids=[uuid.uuid4()])

    assert count(session, DocumentRow) == 0
    assert count(session, StoredFileRow) == 0
    assert stored_paths(file_store_root) == []


def test_changing_the_collections_groups_later_doesnt_change_the_document(
    upload: Callable[..., DocumentDetails],
    document_service: DocumentService,
    collections: CollectionRepository,
    manager: Principal,
    hr_docs: Collection,
    hr: Group,
    it: Group,
) -> None:
    details = upload()

    collections.set_groups(hr_docs.id, [it.id])

    assert document_service.get_document(manager, details.document.id).groups == [hr]


# the same file twice


def test_the_same_file_twice_in_a_collection_is_refused(
    upload: Callable[..., DocumentDetails], session: Session
) -> None:
    first = upload()

    with pytest.raises(DuplicateFile) as refused:
        upload(filename="another-name.txt")

    assert refused.value.document_id == first.document.id
    assert count(session, DocumentRow) == 1


def test_the_same_file_in_another_collection_shares_the_stored_file(
    upload: Callable[..., DocumentDetails],
    session: Session,
    admin: Principal,
    it_docs: Collection,
) -> None:
    first = upload()

    second = upload(as_=admin, collection_id=it_docs.id)

    assert second.document.id != first.document.id
    assert second.file.id == first.file.id
    assert count(session, StoredFileRow) == 1


# who may upload


def test_a_manager_of_another_collection_cannot_upload(
    upload: Callable[..., DocumentDetails], other_manager: Principal
) -> None:
    with pytest.raises(NotAllowed):
        upload(as_=other_manager)


def test_a_reader_who_can_see_the_collection_cannot_upload(
    upload: Callable[..., DocumentDetails], hr_reader: Principal
) -> None:
    with pytest.raises(NotAllowed):
        upload(as_=hr_reader)


def test_a_reader_who_cannot_see_the_collection_is_told_it_doesnt_exist(
    upload: Callable[..., DocumentDetails], outsider: Principal
) -> None:
    with pytest.raises(CollectionNotFound):
        upload(as_=outsider)


def test_an_unknown_collection_is_not_found(
    upload: Callable[..., DocumentDetails], admin: Principal
) -> None:
    with pytest.raises(CollectionNotFound):
        upload(as_=admin, collection_id=uuid.uuid4())


def test_a_front_end_cannot_upload_even_for_an_admin(
    upload: Callable[..., DocumentDetails],
    auth: AuthService,
    make_user: Callable[..., User],
    admin: Principal,
) -> None:
    acting = front_end_acting_for(auth, make_user, admin, "admin@example.com")

    with pytest.raises(NotAllowed):
        upload(as_=acting)


# refused files leave nothing behind


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (b"x" * (MAX_UPLOAD_MB * 1024 * 1024 + 1), FileTooLarge),
        (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", UnsupportedFileType),
        (b"", InvalidRequest),
    ],
    ids=["too large", "unsupported", "empty"],
)
def test_a_refused_file_leaves_nothing_behind(
    upload: Callable[..., DocumentDetails],
    session: Session,
    file_store_root: Path,
    content: bytes,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        upload(content=content)

    assert count(session, DocumentRow) == 0
    assert count(session, StoredFileRow) == 0
    assert count(session, IndexingJobRow) == 0
    assert stored_paths(file_store_root) == []


# reading


def test_get_document_returns_what_upload_returned(
    upload: Callable[..., DocumentDetails],
    document_service: DocumentService,
    manager: Principal,
) -> None:
    details = upload()

    assert document_service.get_document(manager, details.document.id) == details


def test_who_can_read_a_document_with_groups(
    upload: Callable[..., DocumentDetails],
    document_service: DocumentService,
    admin: Principal,
    manager: Principal,
    other_manager: Principal,
    hr_reader: Principal,
    outsider: Principal,
) -> None:
    document_id = upload().document.id

    for reader in (admin, manager, hr_reader):
        assert (
            document_service.get_document(reader, document_id).document.id
            == document_id
        )
    for stranger in (other_manager, outsider):
        with pytest.raises(DocumentNotFound):
            document_service.get_document(stranger, document_id)


def test_who_can_read_a_private_document(
    upload: Callable[..., DocumentDetails],
    document_service: DocumentService,
    admin: Principal,
    manager: Principal,
    other_manager: Principal,
    hr_reader: Principal,
) -> None:
    document_id = upload(private=True).document.id

    for reader in (admin, manager):
        assert document_service.get_document(reader, document_id).groups == []
    for stranger in (other_manager, hr_reader):
        with pytest.raises(DocumentNotFound):
            document_service.get_document(stranger, document_id)


def test_a_front_end_reads_as_the_user_it_acts_for(
    upload: Callable[..., DocumentDetails],
    document_service: DocumentService,
    auth: AuthService,
    make_user: Callable[..., User],
    admin: Principal,
    hr_reader: Principal,
    outsider: Principal,
) -> None:
    document_id = upload().document.id
    service_user = make_user("openwebui@example.com")
    key = auth.create_front_end_key(admin, service_user_id=service_user.id, name="owui")
    as_reader = auth.resolve(key.token, acting_for="an@example.com")
    as_outsider = auth.resolve(key.token, acting_for="binh@example.com")
    assert as_reader is not None and as_outsider is not None

    assert (
        document_service.get_document(as_reader, document_id).document.id == document_id
    )
    with pytest.raises(DocumentNotFound):
        document_service.get_document(as_outsider, document_id)


def test_an_unknown_document_is_not_found(
    document_service: DocumentService, admin: Principal
) -> None:
    with pytest.raises(DocumentNotFound):
        document_service.get_document(admin, uuid.uuid4())


# a document being deleted


def test_a_document_being_deleted_is_not_found_even_by_an_admin(
    upload: Callable[..., DocumentDetails],
    document_service: DocumentService,
    session: Session,
    admin: Principal,
    manager: Principal,
) -> None:
    document_id = upload().document.id
    mark_deleting(session, document_id)

    for reader in (admin, manager):
        with pytest.raises(DocumentNotFound):
            document_service.get_document(reader, document_id)


def test_the_same_file_can_be_uploaded_while_its_document_is_being_deleted(
    upload: Callable[..., DocumentDetails], session: Session
) -> None:
    first = upload()
    mark_deleting(session, first.document.id)

    second = upload(filename="another-name.txt")

    assert second.document.id != first.document.id
    assert second.file.id == first.file.id
