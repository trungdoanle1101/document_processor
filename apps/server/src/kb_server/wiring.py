from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from argon2 import PasswordHasher
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from document_processor.chunking.outline_chunker import OutlineChunker
from document_processor.chunking.tokenizer import Tokenizer
from document_processor.retrieval.embedding import Embedder
from document_processor.retrieval.index import ChunkIndex
from kb_server.indexing.handlers import (
    IndexTools,
    JobTools,
    ProfileChooser,
)
from kb_server.indexing.worker import Worker
from kb_server.services.auth import AuthService
from kb_server.services.collections import CollectionService
from kb_server.services.documents import DocumentService
from kb_server.services.groups import GroupService
from kb_server.services.passwords import Passwords
from kb_server.services.users import UserService
from kb_server.settings import Settings
from kb_server.storage.files import FileStore, LocalFileStore
from kb_server.storage.repositories.collections import CollectionRepository
from kb_server.storage.repositories.credentials import CredentialRepository
from kb_server.storage.repositories.documents import DocumentRepository
from kb_server.storage.repositories.files import StoredFileRepository
from kb_server.storage.repositories.groups import GroupRepository
from kb_server.storage.repositories.jobs import JobRepository
from kb_server.storage.repositories.users import UserRepository

if TYPE_CHECKING:
    # for annotations only: importing these for real is left to the worker
    from qdrant_client import QdrantClient

    from document_processor.integrations.qdrant.index import AttributeKind

# Qdrant payload indexes for the attributes search filters on. The names are the ones
# run_index_job writes; a test keeps the two in step. Quoted: AttributeKind exists only
# while type checking.
QDRANT_INDEXED_ATTRIBUTES: "Mapping[str, AttributeKind]" = {
    "access_group_ids": "text",
    "collection_id": "text",
    "document_version": "integer",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, kw_only=True)
class Services:
    settings: Settings
    engine: Engine
    files: FileStore
    session_factory: sessionmaker[Session]
    passwords: Passwords
    clock: Callable[[], datetime] = utc_now


def build_services(settings: Settings) -> Services:
    engine = create_engine(url=settings.database_url, pool_pre_ping=True)
    return Services(
        settings=settings,
        engine=engine,
        files=build_file_store(settings),
        session_factory=sessionmaker(engine),
        passwords=Passwords(hasher=PasswordHasher()),
    )


def build_worker(services: Services, *, job_tools: JobTools) -> Worker:
    settings = services.settings
    return Worker(
        session_factory=services.session_factory,
        tools=job_tools,
        max_attempts=settings.job_max_attempts,
        poll_interval=timedelta(seconds=settings.worker_poll_seconds),
        job_timeout=timedelta(minutes=settings.job_timeout_minutes),
        now=services.clock,
    )


def close_services(services: Services) -> None:
    services.engine.dispose()


def build_user_service(session: Session, services: Services) -> UserService:
    return UserService(
        users=UserRepository(session),
        groups=GroupRepository(session),
        credentials=CredentialRepository(session),
        passwords=services.passwords,
        now=services.clock,
    )


def build_auth_service(session: Session, services: Services) -> AuthService:
    return AuthService(
        users=UserRepository(session),
        credentials=CredentialRepository(session),
        passwords=services.passwords,
        session_ttl=timedelta(hours=services.settings.session_ttl_hours),
        now=services.clock,
    )


def build_group_service(session: Session) -> GroupService:
    return GroupService(groups=GroupRepository(session), users=UserRepository(session))


def build_collection_service(session: Session) -> CollectionService:
    return CollectionService(
        collections=CollectionRepository(session),
        groups=GroupRepository(session),
        users=UserRepository(session),
    )


def build_document_service(session: Session, services: Services) -> DocumentService:
    return DocumentService(
        documents=DocumentRepository(session),
        collections=CollectionRepository(session),
        groups=GroupRepository(session),
        stored_files=StoredFileRepository(session),
        jobs=JobRepository(session),
        max_upload_bytes=services.settings.max_upload_mb * 1024 * 1024,
        files=services.files,
    )


def build_file_store(settings: Settings) -> FileStore:
    match settings.storage_kind:
        case "local":
            assert settings.storage_path is not None
            return LocalFileStore(settings.storage_path)
        case "s3":
            raise NotImplementedError("s3 storage not yet implemented")


class WorkerSetupError(Exception):
    """The worker can't start: a setting is missing or wrong, or Qdrant can't be reached.

    The message says which step failed, followed by the underlying reason. It names a
    KB_* variable only where this module checks that variable itself, so the name can't
    go stale when a library starts failing for another reason.
    """


@contextmanager
def open_job_tools(services: Services) -> Generator[JobTools, None, None]:
    """The worker's tools: the embedder, the profile chooser, and the index.

    Loads the model and connects to the index; closes the connection on exit.
    WorkerSetupError if a setting is wrong or Qdrant can't be reached.
    """
    # the worker's dependencies are imported here, so other processes never load them
    from qdrant_client import QdrantClient

    settings = services.settings
    embedder = _build_embedder(settings)
    profile_chooser = _build_profile_chooser(settings, embedder=embedder)
    api_key = settings.qdrant_api_key
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=api_key.get_secret_value() if api_key is not None else None,
    )
    try:
        index = _open_index(settings, client=qdrant_client, embedder=embedder)
        yield JobTools(
            profile_chooser=profile_chooser,
            index_tools=IndexTools(
                embedder=embedder,
                index=index,
                index_label=f"qdrant:{settings.qdrant_collection}",
            ),
            file_store=services.files,
        )
    finally:
        qdrant_client.close()


def _build_embedder(settings: Settings) -> Embedder:
    from document_processor.integrations.huggingface.embedder import HuggingFaceEmbedder
    from document_processor.integrations.huggingface.model_specs import (
        HUGGINGFACE_MODEL_SPECS,
    )

    if settings.embedder_folder is None:
        raise WorkerSetupError(
            "KB_EMBEDDER_FOLDER is not set: "
            "the worker needs the folder holding the embedder model"
        )
    try:
        spec = HUGGINGFACE_MODEL_SPECS.get(settings.embedder_spec)
    except KeyError as error:
        raise WorkerSetupError(
            f"can't find the embedder spec: {error.args[0]}"
        ) from error
    try:
        return HuggingFaceEmbedder(
            spec=spec,
            folder=settings.embedder_folder,
            device=settings.embedder_device,
            batch_size=settings.embedder_batch_size,
        )
    # how the framework reports a missing, incomplete or mismatched model folder
    except (OSError, ValueError) as error:
        raise WorkerSetupError(
            f"can't load the embedder from {settings.embedder_folder}: {error}"
        ) from error


def _build_profile_chooser(settings: Settings, *, embedder: Embedder) -> ProfileChooser:
    """Chunks sized with the embedder's own tokenizer, prepared with the first profile that fits."""
    tokenizer = _tokenizer_of(embedder)
    try:
        chunker = OutlineChunker(tokenizer=tokenizer, max_tokens=settings.chunk_tokens)
    except ValueError as error:
        raise WorkerSetupError(f"can't build the chunker: {error}") from error
    try:
        return ProfileChooser.from_profiles(
            profiles=settings.profiles,
            fallback_profile=settings.fallback_profile,
            chunker=chunker,
            embedder_identity=embedder.identity,
        )
    except KeyError as error:
        raise WorkerSetupError(f"can't set up the profiles: {error.args[0]}") from error


def _tokenizer_of(embedder: Embedder) -> Tokenizer:
    if embedder.tokenizer is None:
        raise WorkerSetupError(
            f"the embedder {embedder.identity} has no tokenizer to size chunks with"
        )
    return embedder.tokenizer


def _open_index(
    settings: Settings, *, client: "QdrantClient", embedder: Embedder
) -> ChunkIndex:
    """The Qdrant collection, created on first use, else checked against the embedder."""
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )

    from document_processor.integrations.qdrant.index import QdrantChunkIndex

    try:
        return QdrantChunkIndex(
            client=client,
            collection=settings.qdrant_collection,
            embedder_identity=embedder.identity,
            vector_specs=embedder.vector_specs,
            indexed_attributes=QDRANT_INDEXED_ATTRIBUTES,
        )
    # a connection failure: the URL is the only input to connecting
    except ResponseHandlingException as error:
        raise WorkerSetupError(
            f"can't reach Qdrant at {settings.qdrant_url} (KB_QDRANT_URL). Is it running?"
        ) from error
    except UnexpectedResponse as error:
        raise WorkerSetupError(
            f"Qdrant at {settings.qdrant_url} refused the request: {error}"
        ) from error
    # e.g. the collection holds vectors from another embedder
    except ValueError as error:
        raise WorkerSetupError(
            f"can't open the Qdrant collection {settings.qdrant_collection}: {error}"
        ) from error
