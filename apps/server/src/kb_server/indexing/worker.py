from datetime import datetime

from kb_server.domain.jobs import IndexingJob
from kb_server.indexing.handlers import ClaimedJob
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.files import StoredFileRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.jobs import JobRepository


class Worker:
    def __init__(
        self,
        *,
        documents: DocumentRepository,
        stored_files: StoredFileRepository,
        groups: GroupRepository,
        jobs: JobRepository,
    ):
        self._documents = documents
        self._jobs = jobs
        self._stored_files = stored_files
        self._groups = groups

    def run_once(self) -> bool:
        """Run once"""
        # Find the oldest queued job and mark it claimed
        job = self._jobs.claim_next(now=datetime.now())
        if job is None:
            return False

        # Load the claimed job
        claimed = self._load_claimed(job)

        # Check the current version and the claimed job. Mark done if no long current.
        if claimed.job.version != claimed.version.version:
            self._jobs.mark_done(job.id, now=datetime.now())
            return False

        return True

    def _load_claimed(self, job: IndexingJob) -> ClaimedJob:

        document = self._documents.get(job.document_id)
        assert document is not None

        version = self._documents.get_version(document.id, document.current_version)
        assert version is not None

        file = self._stored_files.get(version.file_id)
        assert file is not None

        group_ids = self._documents.list_group_ids(document.id)

        return ClaimedJob(
            job=job,
            document=document,
            version=version,
            file=file,
            group_ids=frozenset(group_ids),
        )
