import logging
import threading
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import assert_never

from sqlalchemy.orm import Session, sessionmaker

from kb_server.domain.documents import Document
from kb_server.domain.files import StoredFile
from kb_server.domain.jobs import IndexingJob
from kb_server.indexing.handlers import (
    ClaimedJob,
    DeleteResult,
    IndexResult,
    JobTools,
    PermanentJobError,
    PrepareResult,
    run_delete_job,
    run_index_job,
    run_prepare_job,
)
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.files import StoredFileRepository
from kb_server.storage.repositories.jobs import JobRepository

logger = logging.getLogger(__name__)

_LEASE_EXPIRED = "lease expired: the work stopped during this job"


class Worker:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        tools: JobTools,
        max_attempts: int,
        poll_interval: timedelta,
        job_timeout: timedelta,
        now: Callable[[], datetime],
    ) -> None:
        self._session_factory = session_factory
        self._tools = tools
        self._max_attempts = max_attempts
        self._poll_interval = poll_interval
        self._job_timeout = job_timeout
        self._now = now

    def run_forever(self, *, stop: threading.Event) -> None:
        """Run jobs until stop is set, waiting poll_interval whenever none is waiting.

        A database error ends the loop: the process exits,
        and on restart the stuck-job sweep requeues the job it was running.
        """
        logger.info(
            "worker started: polling every %gs, at most %d attempts per job",
            self._poll_interval.total_seconds(),
            self._max_attempts,
        )
        while not stop.is_set():
            if not self.run_once():
                stop.wait(self._poll_interval.total_seconds())
        logger.info("worker stopped")

    def run_once(self) -> bool:
        """Claim the oldest job that still needs running, run it and record the outcome.

        False if no job was waiting.
        """
        claimed = self._claim()
        if claimed is None:
            return False

        try:
            result = _run_handler(claimed, tools=self._tools)
            match result:
                case DeleteResult():
                    self._finish_deletion(claimed)
                case _:
                    self._record_success(claimed, result)
        except Exception as error:  # noqa: BLE001
            self._record_failure(claimed.job, error)

        return True

    def recover_stuck_jobs(self) -> int:
        """Requeue jobs left running longer than job_timeout, or fail them on their last attempt.

        Returns how many there were. Run at startup, before run_forever.
        """
        with self._session_factory.begin() as session:
            jobs = JobRepository(session)
            running_jobs = jobs.list_running(
                started_before=self._now() - self._job_timeout
            )
            for job in running_jobs:
                if self._is_last_attempt(job):
                    logger.warning(
                        "job %s (%s, attempt %d): lease expired on its last attempt, failed",
                        job.id,
                        job.kind,
                        job.attempts,
                    )
                    jobs.mark_failed(
                        job.id,
                        error=_LEASE_EXPIRED,
                        now=self._now(),
                    )
                else:
                    logger.warning(
                        "job %s (%s, attempt %d): lease expired, requeued",
                        job.id,
                        job.kind,
                        job.attempts,
                    )
                    jobs.requeue(
                        job.id,
                        error=_LEASE_EXPIRED,
                    )
        return len(running_jobs)

    def _is_last_attempt(self, job: IndexingJob) -> bool:
        """True once the job has used all its attempts."""
        return job.attempts >= self._max_attempts

    def _claim(self) -> ClaimedJob | None:
        """Claim the oldest job that still needs running and load what it's about.

        Fails jobs that can't be loaded and marks superseded ones done on the way, all
        in one transaction. None if no job is left.
        """
        with self._session_factory.begin() as session:
            documents = DocumentRepository(session)
            jobs = JobRepository(session)
            stored_files = StoredFileRepository(session)

            while (job := jobs.claim_next(now=self._now())) is not None:
                try:
                    claimed = _load_claimed(
                        job, documents=documents, stored_files=stored_files
                    )
                except PermanentJobError as error:
                    # the message holds ids only, so it's safe to log
                    logger.warning("job %s failed: can't be loaded: %s", job.id, error)
                    jobs.mark_failed(job.id, error=str(error), now=self._now())
                    continue

                if _is_superseded(job, document=claimed.document):
                    jobs.mark_done(job.id, now=self._now())
                    continue
                return claimed
        return None

    def _record_success(
        self, claimed: ClaimedJob, result: PrepareResult | IndexResult
    ) -> None:
        """Record the result in the version's provenance and mark the job done.

        A prepare job of an active document also queues its index job. One transaction.
        """
        with self._session_factory.begin() as session:
            documents = DocumentRepository(session)
            jobs = JobRepository(session)

            match result:
                case PrepareResult():
                    provenance = {"preparation": result.provenance()}
                    documents.set_provenance(
                        claimed.version.document_id, claimed.version.version, provenance
                    )
                    if result.should_index:
                        jobs.add(
                            kind="index",
                            document_id=claimed.version.document_id,
                            version=claimed.version.version,
                            created_by=claimed.job.created_by,
                        )
                    jobs.mark_done(claimed.job.id, now=self._now())
                case IndexResult():
                    # run_index_job already refuses a version without provenance
                    assert claimed.version.provenance is not None

                    provenance = {
                        **claimed.version.provenance,
                        "index": result.provenance(),
                    }
                    documents.set_provenance(
                        claimed.version.document_id, claimed.version.version, provenance
                    )
                    jobs.mark_done(claimed.job.id, now=self._now())
                case _:
                    assert_never(result)

    def _record_failure(self, job: IndexingJob, error: Exception) -> None:
        """Fail the job for good, or requeue it while attempts are left.

        A PermanentJobError fails it at once. The error is kept on the job either way.
        """
        with self._session_factory.begin() as session:
            jobs = JobRepository(session)
            message = f"{type(error).__name__}: {error}"

            if isinstance(error, PermanentJobError) or self._is_last_attempt(job):
                # the type only: the exception's message could quote document text
                logger.warning(
                    "job %s (%s, attempt %d) failed: %s",
                    job.id,
                    job.kind,
                    job.attempts,
                    type(error).__name__,
                )
                jobs.mark_failed(job.id, error=message, now=self._now())
            else:
                logger.info(
                    "job %s (%s, attempt %d) will be retried: %s",
                    job.id,
                    job.kind,
                    job.attempts,
                    type(error).__name__,
                )
                jobs.requeue(job.id, error=message)

    def _finish_deletion(self, claimed: ClaimedJob) -> None:
        """Delete the document's row, then the originals nothing else uses.

        Nothing is recorded: the job's row goes with the document.
        """
        unused_files = self._delete_rows(claimed.document.id)
        self._delete_originals(unused_files)

    def _delete_rows(self, document_id: uuid.UUID) -> list[StoredFile]:
        with self._session_factory.begin() as session:
            documents = DocumentRepository(session)
            stored_files = StoredFileRepository(session)
            file_ids = documents.list_file_ids(document_id)
            documents.delete(document_id)
            return stored_files.delete_unused(file_ids)

    def _delete_originals(self, files: Sequence[StoredFile]) -> None:
        store = self._tools.file_store
        for file in files:
            try:
                store.delete(file.storage_key)
            except OSError as error:
                logger.warning(
                    "original of stored file %s not deleted (%s); kb-server gc will remove it",
                    file.id,
                    type(error).__name__,
                )


def _load_claimed(
    job: IndexingJob,
    *,
    documents: DocumentRepository,
    stored_files: StoredFileRepository,
) -> ClaimedJob:
    """Load a claimed job's document, current version, stored file and groups.

    PermanentJobError if a row it needs is missing.
    """
    document = documents.get(job.document_id)
    if document is None:
        raise PermanentJobError(f"job {job.id}: can't find document {job.document_id}")

    version = documents.get_version(document.id, document.current_version)
    if version is None:
        raise PermanentJobError(
            f"document {document.id} has no row for its current version v{document.current_version}"
        )

    file = stored_files.get(version.file_id)
    if file is None:
        raise PermanentJobError(
            f"document {document.id} v{version.version}: has no corresponding file {version.file_id}"
        )

    group_ids = documents.list_group_ids(document.id)

    return ClaimedJob(
        job=job,
        document=document,
        version=version,
        file=file,
        group_ids=frozenset(group_ids),
    )


def _is_superseded(job: IndexingJob, *, document: Document) -> bool:
    """True when the job's work is no longer wanted: its document is being deleted
    (unless it's the delete job), or it's for a version that isn't current."""
    if document.status == "deleting":
        return job.kind != "delete"

    return job.version is not None and job.version != document.current_version


def _run_handler(
    claimed: ClaimedJob, *, tools: JobTools
) -> PrepareResult | IndexResult | DeleteResult:
    """Run the handler for the job's kind and return its result. Raises what it raises."""
    match claimed.job.kind:
        case "prepare":
            return run_prepare_job(
                claimed, chooser=tools.profile_chooser, store=tools.file_store
            )
        case "index":
            return run_index_job(
                claimed, tools=tools.index_tools, store=tools.file_store
            )
        case "sync_attributes":
            raise PermanentJobError("sync_attributes not yet supported")
        case "delete":
            return run_delete_job(
                claimed, index=tools.index_tools.index, store=tools.file_store
            )
        case _:
            assert_never(claimed.job.kind)
