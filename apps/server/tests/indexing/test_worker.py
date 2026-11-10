"""The worker: claiming a job, running its handler and recording the outcome.

Real Postgres rows (uploaded through DocumentService, so each document comes with its
prepare job), the real ProfileChooser, and the handler tests' word-counting embedder and
in-memory index. The worker's sessions join the test's transaction, so its commits are
savepoints; the test session reads them after expire_all().
"""

import io
import logging
import uuid
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import Session, sessionmaker

from document_processor.retrieval.in_memory_index import InMemoryChunkIndex
from document_processor.retrieval.index import ChunkFilter, IndexedChunk
from kb_server.domain.collections import Collection
from kb_server.domain.credentials import Principal
from kb_server.domain.documents import DocumentDetails, DocumentVersion
from kb_server.domain.groups import Group
from kb_server.domain.jobs import IndexingJob, JobKind
from kb_server.domain.users import User
from kb_server.indexing.handlers import IndexTools, JobTools
from kb_server.indexing.worker import Worker
from kb_server.services.documents import DocumentService
from kb_server.storage.files import FileStore
from kb_server.storage.models import DocumentRow, DocumentVersionRow, IndexingJobRow
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.jobs import JobRepository
from tests.indexing.test_handlers import (
    INDEX_LABEL,
    REGULATION,
    VN_REGULATION,
    CountingEmbedder,
    make_chooser,
)
from tests.support import FakeClock

MakeWorker = Callable[..., Worker]
Upload = Callable[..., DocumentDetails]


# setup


@pytest.fixture
def admin(make_user: Callable[..., User]) -> Principal:
    return Principal(
        user=make_user("admin@example.com", role="admin"),
        credential_id=uuid.uuid4(),
        credential_kind="session",
        acting_for_others=False,
    )


@pytest.fixture
def hr(groups: GroupRepository) -> Group:
    return groups.add("hr")


@pytest.fixture
def hr_docs(collections: CollectionRepository, hr: Group) -> Collection:
    collection = collections.add(slug="hr-docs", title="HR documents", description="")
    collections.set_groups(collection.id, [hr.id])
    return collection


@pytest.fixture
def upload(
    document_service: DocumentService, admin: Principal, hr_docs: Collection
) -> Upload:
    """upload(content=…, filename=…): an active document in hr-docs, with its prepare job."""

    def run(
        *, content: bytes = REGULATION.encode(), filename: str = "quy-dinh.txt"
    ) -> DocumentDetails:
        return document_service.upload_document(
            admin,
            collection_id=hr_docs.id,
            content=io.BytesIO(content),
            filename=filename,
            title=None,
            group_ids=None,
            private=False,
        )

    return run


@pytest.fixture
def embedder() -> CountingEmbedder:
    return CountingEmbedder()


@pytest.fixture
def index(embedder: CountingEmbedder) -> InMemoryChunkIndex:
    return InMemoryChunkIndex(
        embedder_identity=embedder.identity, vector_specs=embedder.vector_specs
    )


@pytest.fixture
def make_worker(
    session_factory: sessionmaker[Session],
    file_store: FileStore,
    index: InMemoryChunkIndex,
    embedder: CountingEmbedder,
    clock: FakeClock,
) -> MakeWorker:
    """make_worker(embedder=…, max_attempts=…): a worker over the test's rows and index."""

    def make(
        *, embedder_: CountingEmbedder | None = None, max_attempts: int = 3
    ) -> Worker:
        tools = JobTools(
            profile_chooser=make_chooser(),
            index_tools=IndexTools(
                embedder=embedder_ or embedder, index=index, index_label=INDEX_LABEL
            ),
            file_store=file_store,
        )
        return Worker(
            session_factory=session_factory,
            tools=tools,
            max_attempts=max_attempts,
            now=clock,
        )

    return make


@pytest.fixture
def worker(make_worker: MakeWorker) -> Worker:
    return make_worker()


class UnreachableEmbedder(CountingEmbedder):
    """Fails every embedding with the error's message given, like a model server that's down."""

    def __init__(self, message: str = "model server unreachable") -> None:
        super().__init__()
        self._message = message

    def encode_passages(self, texts: Sequence[str]) -> list[Any]:
        raise ConnectionError(self._message)


# an empty queue


def test_run_once_returns_false_when_no_job_is_waiting(worker: Worker) -> None:
    assert worker.run_once() is False


# the prepare job


def test_a_prepare_job_records_its_preparation_and_queues_the_index_job(
    worker: Worker, upload: Upload, session: Session, clock: FakeClock
) -> None:
    details = upload()

    assert worker.run_once() is True

    prepare = job(session, _first_job_id(details))
    assert (prepare.status, prepare.attempts) == ("done", 1)
    assert prepare.started_at == clock.now
    assert prepare.finished_at == clock.now
    assert [(j.kind, j.version, j.status) for j in jobs_of(session, details)] == [
        ("prepare", 1, "done"),
        ("index", 1, "queued"),
    ]
    preparation = provenance(session, details)["preparation"]
    assert preparation["profile"] == VN_REGULATION
    assert preparation["profiles_without_structure"] == []
    assert preparation["settings"]["embedder"] == CountingEmbedder().identity


def test_the_index_job_is_queued_by_whoever_queued_the_prepare_job(
    worker: Worker, upload: Upload, session: Session, admin: Principal
) -> None:
    details = upload()

    worker.run_once()

    assert {j.created_by for j in jobs_of(session, details)} == {admin.user.id}


def test_a_draft_document_is_prepared_but_not_indexed(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()
    session.execute(
        update(DocumentRow)
        .where(DocumentRow.id == details.document.id)
        .values(status="draft")
    )

    worker.run_once()

    assert [(j.kind, j.status) for j in jobs_of(session, details)] == [
        ("prepare", "done")
    ]
    assert "preparation" in provenance(session, details)


# the index job


def test_an_index_job_indexes_the_chunks_and_adds_its_provenance(
    worker: Worker, upload: Upload, session: Session, index: InMemoryChunkIndex
) -> None:
    details = upload()

    worker.run_once()
    worker.run_once()

    assert [(j.kind, j.status) for j in jobs_of(session, details)] == [
        ("prepare", "done"),
        ("index", "done"),
    ]
    recorded = provenance(session, details)
    assert recorded["index"] == {
        "embedder": CountingEmbedder().identity,
        "index": INDEX_LABEL,
    }
    assert recorded["preparation"]["profile"] == VN_REGULATION
    assert indexed(index, details)


def test_the_index_gets_the_documents_groups_and_version(
    worker: Worker, upload: Upload, index: InMemoryChunkIndex, hr: Group
) -> None:
    details = upload()

    worker.run_once()
    worker.run_once()

    chunk = indexed(index, details)[0]
    assert chunk.attributes["access_group_ids"] == (str(hr.id),)
    assert chunk.attributes["collection_id"] == str(details.document.collection_id)
    assert chunk.attributes["document_version"] == 1


def test_preparing_a_version_again_drops_its_index_provenance_and_queues_indexing(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()
    worker.run_once()
    worker.run_once()
    queue(session, details, kind="prepare", version=1)

    worker.run_once()

    assert "index" not in provenance(session, details)
    assert sorted((j.kind, j.status) for j in jobs_of(session, details)) == [
        ("index", "done"),
        ("index", "queued"),
        ("prepare", "done"),
        ("prepare", "done"),
    ]


# failures


def test_an_error_is_kept_and_the_job_requeued_while_attempts_are_left(
    make_worker: MakeWorker, upload: Upload, session: Session
) -> None:
    details = upload()
    worker = make_worker(embedder_=UnreachableEmbedder(), max_attempts=2)
    worker.run_once()

    assert worker.run_once() is True

    index_job = jobs_of(session, details)[-1]
    assert (index_job.kind, index_job.status) == ("index", "queued")
    assert index_job.attempts == 1
    assert index_job.error == "ConnectionError: model server unreachable"
    assert index_job.started_at is None


def test_a_job_fails_for_good_on_its_last_attempt(
    make_worker: MakeWorker, upload: Upload, session: Session, clock: FakeClock
) -> None:
    details = upload()
    worker = make_worker(embedder_=UnreachableEmbedder(), max_attempts=2)
    worker.run_once()

    worker.run_once()
    clock.advance(timedelta(minutes=1))
    worker.run_once()

    index_job = jobs_of(session, details)[-1]
    assert (index_job.status, index_job.attempts) == ("failed", 2)
    assert index_job.finished_at == clock.now
    assert worker.run_once() is False


def test_a_permanent_error_fails_the_job_at_once(
    worker: Worker, upload: Upload, session: Session, file_store_root: Path
) -> None:
    details = upload()
    (file_store_root / details.file.storage_key).unlink()

    worker.run_once()

    prepare = jobs_of(session, details)[0]
    assert (prepare.status, prepare.attempts) == ("failed", 1)
    assert prepare.error is not None
    assert prepare.error.startswith("PermanentJobError: ")
    assert len(jobs_of(session, details)) == 1


def test_a_failure_logs_the_error_type_but_not_its_message(
    make_worker: MakeWorker, upload: Upload, caplog: pytest.LogCaptureFixture
) -> None:
    upload()
    # an error message that quotes the document, as a framework error might
    worker = make_worker(embedder_=UnreachableEmbedder("Điều 1. Phạm vi điều chỉnh"))
    worker.run_once()

    with caplog.at_level(logging.INFO, logger="kb_server.indexing.worker"):
        worker.run_once()

    assert "ConnectionError" in caplog.text
    assert "Phạm vi" not in caplog.text


def test_a_job_whose_rows_cant_be_loaded_is_failed_and_the_next_job_runs(
    worker: Worker, upload: Upload, session: Session
) -> None:
    broken = upload()
    healthy = upload(content="Điều 1. Một văn bản khác\n".encode(), filename="khac.txt")
    queued_at(session, _first_job_id(broken), minutes=-1)
    # current_version has no foreign key, so it can point at a version row that's missing
    session.execute(
        update(DocumentRow)
        .where(DocumentRow.id == broken.document.id)
        .values(current_version=9)
    )

    assert worker.run_once() is True

    failed = job(session, _first_job_id(broken))
    assert failed.status == "failed"
    assert failed.error is not None
    assert "v9" in failed.error
    assert job(session, _first_job_id(healthy)).status == "done"


# superseded jobs


def test_a_superseded_job_is_skipped_and_the_next_job_runs_in_the_same_call(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()
    add_current_version(session, details, version=2)
    newer = queue(session, details, kind="prepare", version=2)
    queued_at(session, _first_job_id(details), minutes=-1)

    assert worker.run_once() is True

    assert job(session, _first_job_id(details)).status == "done"
    assert job(session, newer.id).status == "done"
    assert [(j.kind, j.version) for j in jobs_of(session, details)][-1] == ("index", 2)
    assert provenance(session, details, version=1) is None


def test_a_superseded_index_job_embeds_nothing(
    worker: Worker, upload: Upload, session: Session, embedder: CountingEmbedder
) -> None:
    details = upload()
    worker.run_once()
    add_current_version(session, details, version=2)

    worker.run_once()

    assert jobs_of(session, details)[-1].status == "done"
    assert embedder.passages == []


def test_a_job_for_the_whole_document_is_never_superseded(
    worker: Worker, upload: Upload, session: Session
) -> None:
    details = upload()
    session.execute(delete(IndexingJobRow))
    add_current_version(session, details, version=2)
    sync = queue(session, details, kind="sync_attributes", version=None)

    worker.run_once()

    # it reached its handler, which doesn't support it yet
    recorded = job(session, sync.id)
    assert recorded.status == "failed"
    assert recorded.error == "PermanentJobError: sync_attributes not yet supported"


# helpers


def job(session: Session, job_id: uuid.UUID) -> IndexingJob:
    """The job as the worker left it."""
    session.expire_all()
    found = JobRepository(session).get(job_id)
    assert found is not None
    return found


def jobs_of(session: Session, details: DocumentDetails) -> list[IndexingJob]:
    """The document's jobs as the worker left them, by version, then prepare before index.

    Jobs of the same kind and version come in no particular order.
    """
    session.expire_all()
    ids = session.scalars(
        select(IndexingJobRow.id).where(
            IndexingJobRow.document_id == details.document.id
        )
    ).all()
    found = [job(session, job_id) for job_id in ids]
    order: dict[JobKind, int] = {"prepare": 0, "index": 1, "sync_attributes": 2}
    return sorted(found, key=lambda j: (j.version or 0, order[j.kind]))


def provenance(session: Session, details: DocumentDetails, *, version: int = 1) -> Any:
    session.expire_all()
    found = DocumentRepository(session).get_version(details.document.id, version)
    assert found is not None
    return found.provenance


def queue(
    session: Session,
    details: DocumentDetails,
    *,
    kind: JobKind,
    version: int | None,
) -> IndexingJob:
    return JobRepository(session).add(
        kind=kind, document_id=details.document.id, version=version, created_by=None
    )


def queued_at(session: Session, job_id: uuid.UUID, *, minutes: int) -> None:
    """Move a job's created_at. Jobs added in one test share the transaction's now()."""
    session.execute(
        update(IndexingJobRow)
        .where(IndexingJobRow.id == job_id)
        .values(created_at=IndexingJobRow.created_at + timedelta(minutes=minutes))
    )


def add_current_version(
    session: Session, details: DocumentDetails, *, version: int
) -> DocumentVersion:
    """A newer version of the same file becomes current, as a re-upload would make it."""
    session.execute(
        insert(DocumentVersionRow).values(
            document_id=details.document.id,
            version=version,
            file_id=details.file.id,
            original_filename=details.version.original_filename,
            uploaded_by=details.document.created_by,
        )
    )
    session.execute(
        update(DocumentRow)
        .where(DocumentRow.id == details.document.id)
        .values(current_version=version)
    )
    found = DocumentRepository(session).get_version(details.document.id, version)
    assert found is not None
    return found


def indexed(index: InMemoryChunkIndex, details: DocumentDetails) -> list[IndexedChunk]:
    return index.list_chunks(
        ChunkFilter(document_ids=(str(details.document.id),)), limit=1000
    )


def _first_job_id(details: DocumentDetails) -> uuid.UUID:
    """The prepare job the upload queued."""
    assert details.job is not None
    return details.job.id
