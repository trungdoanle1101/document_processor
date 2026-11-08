import uuid
from datetime import datetime
from http import HTTPStatus
from typing import Annotated

from fastapi import Form, UploadFile
from fastapi.routing import APIRouter
from pydantic import BaseModel, StringConstraints

from kb_server.api.deps import CurrentPrincipal, DocumentServiceDep
from kb_server.api.groups import GroupOut
from kb_server.domain.documents import DocumentDetails, DocumentStatus
from kb_server.domain.jobs import JobStatus

router = APIRouter(prefix="/documents", tags=["documents"])

Title = Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)]


class FileOut(BaseModel):
    filename: str
    size: int
    media_type: str
    sha256: str


class IndexingJobOut(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    error: str | None


class DocumentOut(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    status: DocumentStatus
    current_version: int
    created_at: datetime
    updated_at: datetime
    groups: list[GroupOut]
    file: FileOut
    job: IndexingJobOut | None

    @classmethod
    def from_details(cls, details: DocumentDetails) -> "DocumentOut":
        return DocumentOut(
            id=details.document.id,
            collection_id=details.document.collection_id,
            title=details.document.title,
            status=details.document.status,
            current_version=details.document.current_version,
            created_at=details.document.created_at,
            updated_at=details.document.updated_at,
            groups=[GroupOut.model_validate(group) for group in details.groups],
            file=FileOut(
                filename=details.version.original_filename,
                size=details.file.size,
                media_type=details.file.media_type,
                sha256=details.file.sha256,
            ),
            job=IndexingJobOut(
                job_id=details.job.id,
                status=details.job.status,
                error=details.job.error,
            )
            if details.job is not None
            else None,
        )


@router.post("", status_code=HTTPStatus.CREATED)
def upload_document(
    principal: CurrentPrincipal,
    document_service: DocumentServiceDep,
    file: UploadFile,
    collection_id: Annotated[uuid.UUID, Form()],
    title: Annotated[Title | None, Form()] = None,
    group_ids: Annotated[list[uuid.UUID] | None, Form()] = None,
    private: Annotated[bool, Form()] = False,
) -> DocumentOut:
    details = document_service.upload_document(
        principal,
        collection_id=collection_id,
        content=file.file,
        filename=file.filename or "upload",
        title=title,
        group_ids=group_ids,
        private=private,
    )
    return DocumentOut.from_details(details)


@router.get("/{document_id}")
def get_document(
    document_id: uuid.UUID,
    principal: CurrentPrincipal,
    document_service: DocumentServiceDep,
) -> DocumentOut:
    details = document_service.get_document(principal, document_id)
    return DocumentOut.from_details(details)
