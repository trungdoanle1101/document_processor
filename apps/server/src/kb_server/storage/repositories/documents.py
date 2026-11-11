import uuid
from collections.abc import Iterable, Mapping
from typing import Any, cast

from sqlalchemy import CursorResult, and_, insert, select, update
from sqlalchemy.orm import Session

from kb_server.domain.documents import Document, DocumentStatus, DocumentVersion
from kb_server.storage.models import (
    DocumentGroupRow,
    DocumentRow,
    DocumentVersionRow,
    StoredFileRow,
    UserGroupRow,
)


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        *,
        collection_id: uuid.UUID,
        title: str,
        status: DocumentStatus,
        group_ids: Iterable[uuid.UUID],
        file_id: uuid.UUID,
        original_filename: str,
        created_by: uuid.UUID,
    ) -> tuple[Document, DocumentVersion]:
        """A new document with version 1 and its read groups, written together."""
        document_id = uuid.uuid4()
        document_row = DocumentRow(
            id=document_id,
            collection_id=collection_id,
            title=title,
            status=status,
            current_version=1,
            created_by=created_by,
        )
        version_row = DocumentVersionRow(
            document_id=document_id,
            version=1,
            file_id=file_id,
            original_filename=original_filename,
            uploaded_by=created_by,
        )

        self._session.add_all([document_row, version_row])
        self._session.flush()

        group_rows = [
            {"document_id": document_id, "group_id": group_id}
            for group_id in set(group_ids)
        ]

        if group_rows:
            self._session.execute(insert(DocumentGroupRow), group_rows)

        return to_document(document_row), to_document_version(version_row)

    def get(self, document_id: uuid.UUID) -> Document | None:
        row = self._session.get(DocumentRow, document_id)
        return to_document(row) if row is not None else None

    def get_version(
        self, document_id: uuid.UUID, version: int
    ) -> DocumentVersion | None:
        row = self._session.get(DocumentVersionRow, (document_id, version))
        return to_document_version(row) if row is not None else None

    def list_group_ids(self, document_id: uuid.UUID) -> set[uuid.UUID]:
        ids = self._session.scalars(
            select(DocumentGroupRow.group_id).where(
                DocumentGroupRow.document_id == document_id
            )
        ).all()
        return set(ids)

    def find_current_with_file(
        self, collection_id: uuid.UUID, sha256: str
    ) -> Document | None:
        """A document in the collection whose current version is this file."""
        row = self._session.scalar(
            select(DocumentRow)
            .join(
                DocumentVersionRow,
                and_(
                    DocumentRow.id == DocumentVersionRow.document_id,
                    DocumentRow.current_version == DocumentVersionRow.version,
                ),
            )
            .join(StoredFileRow, DocumentVersionRow.file_id == StoredFileRow.id)
            .where(
                DocumentRow.collection_id == collection_id,
                StoredFileRow.sha256 == sha256,
                DocumentRow.status != "deleting",
            )
        )
        return to_document(row) if row is not None else None

    def is_visible_to(self, document_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Whether the user shares a read group with the document."""
        shared_groups = (
            select(DocumentGroupRow.group_id)
            .join(UserGroupRow, DocumentGroupRow.group_id == UserGroupRow.group_id)
            .where(
                DocumentGroupRow.document_id == document_id,
                UserGroupRow.user_id == user_id,
            )
        )
        return bool(self._session.scalar(select(shared_groups.exists())))

    def set_provenance(
        self, document_id: uuid.UUID, version: int, provenance: Mapping[str, Any]
    ) -> bool:
        """Replace the version's provenance. False if there is no such version."""
        result = self._session.execute(
            update(DocumentVersionRow)
            .where(
                DocumentVersionRow.document_id == document_id,
                DocumentVersionRow.version == version,
            )
            .values(provenance=dict(provenance))
        )

        return cast(CursorResult[Any], result).rowcount == 1


def to_document(row: DocumentRow) -> Document:
    return Document(
        id=row.id,
        collection_id=row.collection_id,
        title=row.title,
        status=row.status,
        current_version=row.current_version,
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def to_document_version(row: DocumentVersionRow) -> DocumentVersion:
    return DocumentVersion(
        document_id=row.document_id,
        version=row.version,
        file_id=row.file_id,
        original_filename=row.original_filename,
        provenance=row.provenance,
        uploaded_by=row.uploaded_by,
        created_at=row.created_at,
    )
