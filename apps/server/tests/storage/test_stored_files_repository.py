import hashlib
import uuid
from collections.abc import Callable

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kb_server.domain.files import StoredFile
from kb_server.domain.users import User
from kb_server.storage.files import original_key
from kb_server.storage.models import StoredFileRow
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.files import StoredFileRepository


def store(stored_files: StoredFileRepository, content: bytes) -> StoredFile:
    sha256 = hashlib.sha256(content).hexdigest()
    return stored_files.get_or_add(
        sha256=sha256,
        size=len(content),
        media_type="text/plain",
        storage_key=original_key(sha256),
    )


def count_rows(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(StoredFileRow)) or 0


def test_a_new_file_is_added(stored_files: StoredFileRepository) -> None:
    stored = store(stored_files, b"Dieu 1")

    assert stored.sha256 == hashlib.sha256(b"Dieu 1").hexdigest()
    assert stored.size == 6
    assert stored.media_type == "text/plain"
    assert stored.storage_key == original_key(stored.sha256)
    assert stored.created_at.tzinfo is not None
    assert stored_files.get(stored.id) == stored
    assert stored_files.get_by_sha256(stored.sha256) == stored


def test_the_same_content_returns_the_same_row(
    session: Session, stored_files: StoredFileRepository
) -> None:
    first = store(stored_files, b"Dieu 1")

    again = store(stored_files, b"Dieu 1")

    assert again == first
    assert count_rows(session) == 1


def test_different_content_gets_its_own_row(
    session: Session, stored_files: StoredFileRepository
) -> None:
    first = store(stored_files, b"Dieu 1")
    second = store(stored_files, b"Dieu 2")

    assert first.id != second.id
    assert count_rows(session) == 2


def test_losing_the_race_returns_the_winners_row(
    session: Session,
    stored_files: StoredFileRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Another upload of the same file adds its row between our lookup and our insert.
    # Simulate it: the row exists, but our first lookup misses it.
    sha256 = hashlib.sha256(b"Dieu 1").hexdigest()
    winner = StoredFileRow(
        sha256=sha256, size=6, media_type="text/plain", storage_key=original_key(sha256)
    )
    session.add(winner)
    session.flush()
    real_lookup = stored_files.get_by_sha256
    lookups: list[str] = []

    def miss_the_first_time(sha256: str) -> StoredFile | None:
        lookups.append(sha256)
        return None if len(lookups) == 1 else real_lookup(sha256)

    monkeypatch.setattr(stored_files, "get_by_sha256", miss_the_first_time)

    stored = store(stored_files, b"Dieu 1")

    assert stored.id == winner.id
    assert len(lookups) == 2  # missed, inserted, hit the unique sha256, looked again
    assert count_rows(session) == 1  # the savepoint undid only our insert


def test_unknown_files_are_none(stored_files: StoredFileRepository) -> None:
    assert stored_files.get(uuid.uuid4()) is None
    assert stored_files.get_by_sha256("0" * 64) is None


# deleting files no version uses


@pytest.fixture
def use(
    documents: DocumentRepository,
    collections: CollectionRepository,
    make_user: Callable[..., User],
) -> Callable[[StoredFile], None]:
    """use(file): a document whose version 1 is this file."""
    uploader = make_user("manager@example.com", role="manager")
    collection = collections.add(slug="hr-docs", title="HR", description="")

    def add(file: StoredFile) -> None:
        documents.add(
            collection_id=collection.id,
            title="Quy chế",
            status="active",
            group_ids=[],
            file_id=file.id,
            original_filename="quy-che.txt",
            created_by=uploader.id,
        )

    return add


def test_an_unused_file_is_deleted_and_returned(
    stored_files: StoredFileRepository,
) -> None:
    unused = store(stored_files, b"Dieu 1")

    assert stored_files.delete_unused([unused.id]) == [unused]
    assert stored_files.get(unused.id) is None


def test_a_file_a_version_uses_is_kept(
    stored_files: StoredFileRepository, use: Callable[[StoredFile], None]
) -> None:
    used = store(stored_files, b"Dieu 1")
    use(used)

    assert stored_files.delete_unused([used.id]) == []
    assert stored_files.get(used.id) == used


def test_only_the_unused_ones_of_the_given_files_go(
    stored_files: StoredFileRepository, use: Callable[[StoredFile], None]
) -> None:
    used = store(stored_files, b"Dieu 1")
    use(used)
    unused = store(stored_files, b"Dieu 2")

    assert stored_files.delete_unused([used.id, unused.id]) == [unused]
    assert stored_files.get(used.id) == used
    assert stored_files.get(unused.id) is None


def test_an_unused_file_that_wasnt_given_is_kept(
    stored_files: StoredFileRepository,
) -> None:
    given = store(stored_files, b"Dieu 1")
    other = store(stored_files, b"Dieu 2")

    stored_files.delete_unused([given.id])

    assert stored_files.get(other.id) == other


def test_no_files_given_deletes_nothing(
    session: Session, stored_files: StoredFileRepository
) -> None:
    store(stored_files, b"Dieu 1")

    assert stored_files.delete_unused([]) == []
    assert count_rows(session) == 1
