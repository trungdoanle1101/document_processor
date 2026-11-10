import uuid
from datetime import datetime
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.orm import Session

from kb_server.domain.jobs import IndexingJob, JobKind
from kb_server.storage.models import IndexingJobRow


class JobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        *,
        kind: JobKind,
        document_id: uuid.UUID,
        version: int | None,
        created_by: uuid.UUID | None,
    ) -> IndexingJob:
        """Add a new job and set it as queued."""
        row = IndexingJobRow(
            kind=kind,
            document_id=document_id,
            version=version,
            created_by=created_by,
            status="queued",
        )
        self._session.add(row)
        self._session.flush()
        return to_indexing_job(row)

    def get(self, job_id: uuid.UUID) -> IndexingJob | None:
        row = self._session.get(IndexingJobRow, job_id)
        return to_indexing_job(row) if row is not None else None

    def latest_for_version(
        self, document_id: uuid.UUID, version: int
    ) -> IndexingJob | None:
        """Get the latest job for a document version."""
        row = self._session.scalar(
            select(IndexingJobRow)
            .where(
                IndexingJobRow.document_id == document_id,
                IndexingJobRow.version == version,
            )
            .order_by(IndexingJobRow.created_at.desc())
            .limit(1)
        )
        return to_indexing_job(row) if row is not None else None

    def claim_next(self, *, now: datetime) -> IndexingJob | None:
        row = self._session.scalar(
            select(IndexingJobRow)
            .where(IndexingJobRow.status == "queued")
            .order_by(IndexingJobRow.created_at)
            .limit(1)
        )
        if not row:
            return None
        row.status, row.attempts, row.started_at = "running", row.attempts + 1, now
        self._session.flush()
        return to_indexing_job(row) if row is not None else None

    def claim(self, job_id: uuid.UUID, *, now: datetime) -> bool:
        """queued -> running. False if someone else got it first."""
        result = self._session.execute(
            update(IndexingJobRow)
            .where(IndexingJobRow.id == job_id, IndexingJobRow.status == "queued")
            .values(
                status="running", started_at=now, attempts=IndexingJobRow.attempts + 1
            )
        )
        return cast(CursorResult[Any], result).rowcount == 1

    def requeue(self, job_id: uuid.UUID, *, error: str) -> None:
        """running -> queued, keeping the error, so it's tried again."""
        result = self._session.execute(
            update(IndexingJobRow)
            .where(IndexingJobRow.id == job_id, IndexingJobRow.status == "running")
            .values(status="queued", error=error, started_at=None)
        )
        if cast(CursorResult[Any], result).rowcount != 1:
            raise RuntimeError(f"job {job_id} isn't running, so it can't be requeued")

    def mark_done(self, job_id: uuid.UUID, *, now: datetime) -> None:
        """running -> done."""
        result = self._session.execute(
            update(IndexingJobRow)
            .where(IndexingJobRow.id == job_id, IndexingJobRow.status == "running")
            .values(status="done", finished_at=now)
        )
        if cast(CursorResult[Any], result).rowcount != 1:
            raise RuntimeError(
                f"job {job_id} isn't running, so it can't be marked done"
            )

    def mark_failed(self, job_id: uuid.UUID, *, error: str, now: datetime) -> None:
        """running -> failed, for good."""
        result = self._session.execute(
            update(IndexingJobRow)
            .where(IndexingJobRow.id == job_id, IndexingJobRow.status == "running")
            .values(status="failed", error=error, finished_at=now)
        )
        if cast(CursorResult[Any], result).rowcount != 1:
            raise RuntimeError(
                f"job {job_id} isn't running, so it can't be marked failed"
            )


def to_indexing_job(row: IndexingJobRow) -> IndexingJob:
    return IndexingJob(
        id=row.id,
        kind=row.kind,
        document_id=row.document_id,
        version=row.version,
        status=row.status,
        attempts=row.attempts,
        error=row.error,
        created_by=row.created_by,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )
