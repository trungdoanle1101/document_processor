import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kb_server.domain.files import StoredFile
from kb_server.storage.models import StoredFileRow


class StoredFileRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, file_id: uuid.UUID) -> StoredFile | None:
        row = self._session.get(StoredFileRow, file_id)
        return to_stored_file(row) if row is not None else None

    def get_by_sha256(self, sha256: str) -> StoredFile | None:
        row = self._session.scalar(
            select(StoredFileRow).where(StoredFileRow.sha256 == sha256)
        )
        return to_stored_file(row) if row is not None else None

    def get_or_add(
        self, *, sha256: str, size: int, media_type: str, storage_key: str
    ) -> StoredFile:
        """The stored file with this hash, adding it if it's new."""
        found = self.get_by_sha256(sha256)
        if found is not None:
            return found
        try:
            with self._session.begin_nested():
                row = StoredFileRow(
                    sha256=sha256,
                    size=size,
                    media_type=media_type,
                    storage_key=storage_key,
                )
                self._session.add(row)
                self._session.flush()
        except IntegrityError:
            found = self.get_by_sha256(sha256)
            assert found is not None
            return found
        return to_stored_file(row)


def to_stored_file(row: StoredFileRow) -> StoredFile:
    return StoredFile(
        id=row.id,
        sha256=row.sha256,
        size=row.size,
        media_type=row.media_type,
        storage_key=row.storage_key,
        created_at=row.created_at,
    )
