import hashlib
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kb_server.domain.documents import Document
from kb_server.domain.jobs import IndexingJob
from kb_server.domain.users import User
from kb_server.storage.models import DocumentRow, IndexingJobRow, StoredFileRow
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.jobs import JobRepository

NOW = datetime(2026, 11, 8, 9, 0, tzinfo=UTC)


@pytest.fixture
def uploader(make_user: Callable[..., User]) -> User:
    return make_user("manager@example.com", role="manager")


@pytest.fixture
def document(
    session: Session,
    documents: DocumentRepository,
    collections: CollectionRepository,
    uploader: User,
) -> Document:
    sha256 = hashlib.sha256(b"Dieu 1").hexdigest()
    file_row = StoredFileRow(
        sha256=sha256, size=6, media_type="text/plain", storage_key=f"k/{sha256}"
    )
    session.add(file_row)
    session.flush()
    collection = collections.add(slug="hr-docs", title="HR", description="")
    document, _ = documents.add(
        collection_id=collection.id,
        title="Quy chế",
        status="active",
        group_ids=[],
        file_id=file_row.id,
        original_filename="quy-che.docx",
        created_by=uploader.id,
    )
    return document


@pytest.fixture
def queued(jobs: JobRepository, document: Document, uploader: User) -> IndexingJob:
    return jobs.add(
        kind="prepare", document_id=document.id, version=1, created_by=uploader.id
    )


def reload(jobs: JobRepository, job_id: uuid.UUID) -> IndexingJob:
    job = jobs.get(job_id)
    assert job is not None
    return job


# adding


def test_a_new_job_is_queued_and_untouched(
    queued: IndexingJob, document: Document, uploader: User
) -> None:
    assert queued.kind == "prepare"
    assert queued.document_id == document.id
    assert queued.version == 1
    assert queued.status == "queued"
    assert queued.attempts == 0
    assert queued.error is None
    assert queued.created_by == uploader.id
    assert queued.created_at.tzinfo is not None
    assert queued.started_at is None
    assert queued.finished_at is None


def test_a_job_reads_back_the_same(jobs: JobRepository, queued: IndexingJob) -> None:
    assert jobs.get(queued.id) == queued
    assert jobs.get(uuid.uuid4()) is None


def test_a_job_without_a_version_or_user_is_allowed(
    jobs: JobRepository, document: Document
) -> None:
    job = jobs.add(
        kind="sync_attributes", document_id=document.id, version=None, created_by=None
    )

    assert job.version is None
    assert job.created_by is None


def test_a_job_for_a_version_that_doesnt_exist_is_refused(
    session: Session, jobs: JobRepository, document: Document
) -> None:
    with pytest.raises(IntegrityError), session.begin_nested():
        jobs.add(kind="prepare", document_id=document.id, version=2, created_by=None)


# the latest job of a version


def test_latest_for_version_is_the_newest_job(
    session: Session, jobs: JobRepository, queued: IndexingJob, document: Document
) -> None:
    newer = jobs.add(kind="index", document_id=document.id, version=1, created_by=None)
    # created_at is the transaction's now(), the same for both jobs here: tell them apart
    for job, created_at in [(queued, NOW - timedelta(minutes=5)), (newer, NOW)]:
        session.execute(
            update(IndexingJobRow)
            .where(IndexingJobRow.id == job.id)
            .values(created_at=created_at)
        )

    latest = jobs.latest_for_version(document.id, 1)

    assert latest is not None
    assert latest.id == newer.id


def test_latest_for_version_ignores_other_versions_and_documents(
    jobs: JobRepository, queued: IndexingJob, document: Document
) -> None:
    jobs.add(
        kind="sync_attributes", document_id=document.id, version=None, created_by=None
    )

    assert jobs.latest_for_version(document.id, 2) is None
    assert jobs.latest_for_version(uuid.uuid4(), 1) is None
    latest = jobs.latest_for_version(document.id, 1)
    assert latest is not None
    assert latest.id == queued.id


# claiming


def test_claiming_starts_the_job(jobs: JobRepository, queued: IndexingJob) -> None:
    assert jobs.claim(queued.id, now=NOW)

    job = reload(jobs, queued.id)
    assert job.status == "running"
    assert job.started_at == NOW
    assert job.attempts == 1
    assert job.finished_at is None


def test_a_job_can_only_be_claimed_once(
    jobs: JobRepository, queued: IndexingJob
) -> None:
    assert jobs.claim(queued.id, now=NOW)

    assert not jobs.claim(queued.id, now=NOW + timedelta(seconds=1))
    job = reload(jobs, queued.id)
    assert job.attempts == 1  # the second claim changed nothing
    assert job.started_at == NOW


def test_claiming_an_unknown_job_is_false(jobs: JobRepository) -> None:
    assert not jobs.claim(uuid.uuid4(), now=NOW)


# finishing


def test_a_running_job_can_be_marked_done(
    jobs: JobRepository, queued: IndexingJob
) -> None:
    jobs.claim(queued.id, now=NOW)

    jobs.mark_done(queued.id, now=NOW + timedelta(minutes=2))

    job = reload(jobs, queued.id)
    assert job.status == "done"
    assert job.finished_at == NOW + timedelta(minutes=2)
    assert job.error is None


def test_a_running_job_can_fail_for_good(
    jobs: JobRepository, queued: IndexingJob
) -> None:
    jobs.claim(queued.id, now=NOW)

    jobs.mark_failed(
        queued.id, error="no headings found", now=NOW + timedelta(minutes=1)
    )

    job = reload(jobs, queued.id)
    assert job.status == "failed"
    assert job.error == "no headings found"
    assert job.finished_at == NOW + timedelta(minutes=1)


def test_a_requeued_job_can_be_claimed_again(
    jobs: JobRepository, queued: IndexingJob
) -> None:
    jobs.claim(queued.id, now=NOW)

    jobs.requeue(queued.id, error="Qdrant timed out")

    job = reload(jobs, queued.id)
    assert job.status == "queued"
    assert job.error == "Qdrant timed out"
    assert job.started_at is None
    assert job.attempts == 1

    later = NOW + timedelta(minutes=1)
    assert jobs.claim(queued.id, now=later)
    job = reload(jobs, queued.id)
    assert (job.status, job.attempts, job.started_at) == ("running", 2, later)


# wrong transitions are bugs: they raise and change nothing


@pytest.mark.parametrize("status", ["queued", "done", "failed"])
def test_only_a_running_job_can_be_finished_or_requeued(
    session: Session, jobs: JobRepository, queued: IndexingJob, status: str
) -> None:
    session.execute(
        update(IndexingJobRow)
        .where(IndexingJobRow.id == queued.id)
        .values(status=status)
    )
    before = reload(jobs, queued.id)

    with pytest.raises(RuntimeError):
        jobs.mark_done(queued.id, now=NOW)
    with pytest.raises(RuntimeError):
        jobs.mark_failed(queued.id, error="x", now=NOW)
    with pytest.raises(RuntimeError):
        jobs.requeue(queued.id, error="x")

    assert reload(jobs, queued.id) == before


def test_finishing_an_unknown_job_raises(jobs: JobRepository) -> None:
    with pytest.raises(RuntimeError):
        jobs.mark_done(uuid.uuid4(), now=NOW)


def test_a_job_is_deleted_with_its_document(
    session: Session, jobs: JobRepository, queued: IndexingJob, document: Document
) -> None:
    session.delete(session.get(DocumentRow, document.id))
    session.flush()
    session.expire_all()

    assert jobs.get(queued.id) is None
