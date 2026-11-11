import uuid
from collections.abc import Sequence
from pathlib import PureWindowsPath
from typing import BinaryIO

from kb_server.domain.credentials import Principal, can_manage
from kb_server.domain.documents import Document, DocumentDetails
from kb_server.domain.errors import (
    CollectionNotFound,
    DocumentNotFound,
    DuplicateFile,
    InvalidRequest,
    NotAllowed,
)
from kb_server.domain.groups import Group
from kb_server.services.collections import can_see_collection
from kb_server.services.groups import get_groups
from kb_server.services.uploads import stage_upload
from kb_server.storage.files import FileStore, original_key
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.files import StoredFileRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.jobs import JobRepository


class DocumentService:
    def __init__(
        self,
        *,
        documents: DocumentRepository,
        collections: CollectionRepository,
        groups: GroupRepository,
        stored_files: StoredFileRepository,
        jobs: JobRepository,
        files: FileStore,
        max_upload_bytes: int,
    ) -> None:
        self._documents = documents
        self._collections = collections
        self._groups = groups
        self._stored_files = stored_files
        self._jobs = jobs
        self._files = files
        self._max_upload_bytes = max_upload_bytes

    def upload_document(
        self,
        principal: Principal,
        *,
        collection_id: uuid.UUID,
        content: BinaryIO,
        filename: str,
        title: str | None,
        group_ids: Sequence[uuid.UUID] | None,
        private: bool,
    ) -> DocumentDetails:
        # Check if the collection exists
        collection = self._collections.get(collection_id)
        if collection is None or not can_see_collection(
            self._collections, principal, collection_id
        ):
            raise CollectionNotFound()

        # Check the principal's permission on this collection
        if not can_manage(principal, self._collections.list_manager_ids(collection_id)):
            raise NotAllowed(
                "only an admin or a manager of this collection can upload a document"
            )

        # Check groups
        if private and group_ids is not None:
            raise InvalidRequest("choose either group_ids or private, not both")
        if private:
            groups: list[Group] = []
        elif group_ids is not None:
            groups = get_groups(self._groups, group_ids)
        else:
            groups = self._collections.list_groups(collection_id)

        # Stage upload
        with stage_upload(content, max_bytes=self._max_upload_bytes) as staged:
            # Check if the file (identified by its sha256 hash) is already registered in the collection
            existing = self._documents.find_current_with_file(
                collection_id, staged.sha256
            )
            if existing is not None:
                raise DuplicateFile(existing.id)

            with staged.path.open("rb") as data:
                # Write the file to the store; automatically skip if already stored
                storage_key = original_key(staged.sha256)
                self._files.put(
                    storage_key,
                    data,
                    size=staged.size,
                    media_type=staged.media_type,
                )

            stored_file = self._stored_files.get_or_add(
                sha256=staged.sha256,
                size=staged.size,
                media_type=staged.media_type,
                storage_key=storage_key,
            )

            # Clean filename
            file_path = PureWindowsPath(filename)
            filename = file_path.stem[: 255 - len(file_path.suffix)] + file_path.suffix
            title = title or PureWindowsPath(filename).stem

            document, version = self._documents.add(
                collection_id=collection_id,
                title=title,
                status="active",
                group_ids=[group.id for group in groups],
                file_id=stored_file.id,
                original_filename=filename,
                created_by=principal.user.id,
            )

            job = self._jobs.add(
                kind="prepare",
                document_id=document.id,
                version=1,
                created_by=principal.user.id,
            )

        return DocumentDetails(
            document=document, version=version, file=stored_file, groups=groups, job=job
        )

    def get_document(
        self, principal: Principal, document_id: uuid.UUID
    ) -> DocumentDetails:
        document = self._documents.get(document_id)
        if (
            document is None
            or document.status == "deleting"
            or not self._can_read(principal, document)
        ):
            raise DocumentNotFound()

        version = self._documents.get_version(document.id, document.current_version)
        assert version is not None

        file = self._stored_files.get(version.file_id)
        assert file is not None

        groups = self._groups.get_many(self._documents.list_group_ids(document.id))

        job = self._jobs.latest_for_version(document.id, document.current_version)

        return DocumentDetails(
            document=document, version=version, file=file, groups=groups, job=job
        )

    def _can_read(self, principal: Principal, document: Document) -> bool:
        role = principal.user.role
        if role == "admin":
            return True

        if (
            role == "manager"
            and principal.user.id
            in self._collections.list_manager_ids(document.collection_id)
        ):
            return True

        return self._documents.is_visible_to(document.id, principal.user.id)
