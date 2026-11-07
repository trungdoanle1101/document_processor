import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, get_args

JobKind = Literal["prepare", "index", "sync_attributes"]
JobStatus = Literal["queued", "running", "done", "failed"]

JOB_KINDS = get_args(JobKind)
JOB_STATUSES = get_args(JobStatus)


@dataclass(frozen=True, kw_only=True)
class IndexingJob:
    id: uuid.UUID
    kind: JobKind
    document_id: uuid.UUID
    version: int | None
    status: JobStatus
    attempts: int
    error: str | None
    created_by: uuid.UUID | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
