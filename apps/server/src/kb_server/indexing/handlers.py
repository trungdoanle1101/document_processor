import logging
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Self

from document_processor.chunking.chunker import Chunker
from document_processor.logical.profile import Profile
from document_processor.logical.profiles import PROFILES
from document_processor.preparation.preparer import PreparedDocument, Preparer
from document_processor.retrieval.attributes import Attributes
from document_processor.retrieval.embedding import Embedder
from document_processor.retrieval.index import ChunkIndex
from document_processor.retrieval.ingest import index_document
from kb_server.domain.documents import Document, DocumentVersion
from kb_server.domain.files import DOCX, TEXT, StoredFile
from kb_server.domain.jobs import IndexingJob
from kb_server.indexing.prepared import (
    PreparationSettings,
    PreparedParts,
    load_prepared,
    save_prepared,
    save_review_page,
)
from kb_server.storage.files import FileStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ClaimedJob:
    """What a job is about, loaded when it's claimed."""

    job: IndexingJob
    document: Document
    version: DocumentVersion
    file: StoredFile
    group_ids: frozenset[uuid.UUID]


class PermanentJobError(Exception):
    """Retrying can't help (a missing or unreadable file): the job fails at once."""


@dataclass(frozen=True, kw_only=True)
class ProfileChoice:
    prepared: PreparedDocument
    profile: str  # name@version that was used
    profiles_without_structure: tuple[str, ...]  # in the order they were tried


@dataclass(frozen=True, kw_only=True)
class ProfileChooser:
    """Prepares a file with the first candidate profile that finds structure, else the fallback.

    Carries the PreparationSettings that describe its preparers.
    """

    preparers: tuple[Preparer, ...]
    fallback_preparer: Preparer
    settings: PreparationSettings

    @classmethod
    def from_profiles(
        cls,
        *,
        profiles: Sequence[str],
        fallback_profile: str,
        chunker: Chunker,
        embedder_identity: str,
    ) -> Self:
        resolved = [PROFILES.get(name) for name in profiles]
        fallback = PROFILES.get(fallback_profile)
        settings = PreparationSettings(
            profiles=tuple(_profile_label(p) for p in resolved),
            fallback_profile=_profile_label(fallback),
            chunker=_describe_chunker(chunker),
            embedder=embedder_identity,
        )

        return cls(
            preparers=tuple(Preparer(profile=p, chunker=chunker) for p in resolved),
            fallback_preparer=Preparer(profile=fallback, chunker=chunker),
            settings=settings,
        )

    def prepare(self, path: Path) -> ProfileChoice:
        profiles_without_structure: list[str] = []
        for preparer in self.preparers:
            prepared = preparer.prepare(path)
            if any(node.is_section for node in prepared.logical.nodes):
                return ProfileChoice(
                    prepared=prepared,
                    profile=_profile_label_of(prepared),
                    profiles_without_structure=tuple(profiles_without_structure),
                )
            profiles_without_structure.append(_profile_label_of(prepared))
        prepared = self.fallback_preparer.prepare(path)
        return ProfileChoice(
            prepared=prepared,
            profile=_profile_label_of(prepared),
            profiles_without_structure=tuple(profiles_without_structure),
        )


@dataclass(frozen=True, kw_only=True)
class PrepareResult:
    profile: str
    profiles_without_structure: tuple[str, ...]
    settings: dict[str, Any]  # chooser.settings.describe()
    should_index: bool

    def provenance(self) -> dict[str, Any]:
        """Plain JSON, for document_versions.provenance["preparation"]"""
        return {
            "profile": self.profile,
            "profiles_without_structure": list(self.profiles_without_structure),
            "settings": self.settings,
        }


@dataclass(frozen=True, kw_only=True)
class IndexTools:
    embedder: Embedder
    index: ChunkIndex
    index_label: str

    def __post_init__(self) -> None:
        if self.embedder.identity != self.index.embedder_identity:
            raise ValueError(
                f"the index holds vectors from {self.index.embedder_identity!r}, "
                f"but the embedder is {self.embedder.identity!r}"
            )


@dataclass(frozen=True, kw_only=True)
class IndexResult:
    embedder: str
    index: str

    def provenance(self) -> dict[str, str]:
        """Plain JSON, for document_versions.provenance["index"]"""
        return {
            "embedder": self.embedder,
            "index": self.index,
        }


@dataclass(frozen=True, kw_only=True)
class JobTools:
    profile_chooser: ProfileChooser
    index_tools: IndexTools
    file_store: FileStore


def run_prepare_job(
    job: ClaimedJob, *, chooser: ProfileChooser, store: FileStore
) -> PrepareResult:
    file = job.file
    extension = _file_extension_for(file.media_type)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        source = _copy_original(store, file, work / f"document{extension}")
        choice = _prepare_file(chooser, source)
        _save(store, job, choice, review_page=work / "review.html")
    _log_warnings(job, choice)
    return PrepareResult(
        profile=choice.profile,
        profiles_without_structure=choice.profiles_without_structure,
        settings=chooser.settings.describe(),
        should_index=job.document.status == "active",
    )


def run_index_job(
    job: ClaimedJob, *, tools: IndexTools, store: FileStore
) -> IndexResult:
    """Make this version's prepared chunks the document's only chunks in the index."""
    _check_chunks_sized_for(job, embedder=tools.embedder)
    parts = _load_parts(job, store=store)
    index_document(
        parts.chunks,
        document_id=str(job.document.id),
        embedder=tools.embedder,
        index=tools.index,
        attributes=_index_attributes(job),
    )
    logger.info(
        "document %s v%d indexed: %d chunks",
        job.document.id,
        job.version.version,
        len(parts.chunks),
    )
    return IndexResult(embedder=tools.embedder.identity, index=tools.index_label)


def _describe_chunker(chunker: Chunker) -> dict[str, Any]:
    """The chunker's type and options, without the tokenizer."""
    if not is_dataclass(chunker) or isinstance(chunker, type):
        raise TypeError(
            f"{type(chunker).__name__} must be a dataclass instance to be described"
        )

    options = {
        f.name: getattr(chunker, f.name)
        for f in fields(chunker)
        if f.name != "tokenizer"
    }
    return {"kind": type(chunker).__name__, **options}


def _profile_label(profile: Profile) -> str:
    return f"{profile.name}@{profile.version}"


def _file_extension_for(media_type: str) -> str:
    if media_type == DOCX:
        return ".docx"
    elif media_type == TEXT:
        return ".txt"
    else:
        raise PermanentJobError(f"unsupported media type: {media_type}")


def _copy_original(store: FileStore, file: StoredFile, destination: Path) -> Path:
    try:
        original = store.open(file.storage_key)
    except FileNotFoundError:
        raise PermanentJobError(f"stored file {file.sha256} is missing") from None
    with original, destination.open("wb") as out:
        shutil.copyfileobj(original, out)
    return destination


def _prepare_file(chooser: ProfileChooser, path: Path) -> ProfileChoice:
    try:
        return chooser.prepare(path)
    except OSError:
        raise
    except Exception as error:
        raise PermanentJobError(
            f"couldn't prepare the file: {type(error).__name__}: {error}"[:500]
        ) from error


def _save(
    store: FileStore, job: ClaimedJob, choice: ProfileChoice, *, review_page: Path
) -> None:
    save_prepared(
        store,
        job.document.id,
        job.version.version,
        PreparedParts(logical=choice.prepared.logical, chunks=choice.prepared.chunks),
    )
    choice.prepared.write_review_page(review_page)
    save_review_page(store, job.document.id, job.version.version, review_page)


def _log_warnings(job: ClaimedJob, choice: ProfileChoice) -> None:
    counts = Counter(w.code for w in choice.prepared.summarize().warnings)
    if counts:
        logger.warning(
            "document %s v%d prepared with %s: %s",
            job.document.id,
            job.version.version,
            choice.profile,
            ", ".join(f"{code}={n}" for code, n in sorted(counts.items())),
        )


def _profile_label_of(prepared: PreparedDocument) -> str:
    return f"{prepared.logical.profile_name}@{prepared.logical.profile_version}"


def _check_chunks_sized_for(job: ClaimedJob, *, embedder: Embedder) -> None:
    """Raise PermanentJobError unless the version's chunks were sized with this embedder."""
    document = f"document {job.document.id} v{job.version.version}"
    if job.version.provenance is None:
        raise PermanentJobError(f"{document} has no provenance; prepare it again")

    try:
        prepared_for = job.version.provenance["preparation"]["settings"]["embedder"]
    # TypeError: a part of the path that isn't a mapping, e.g. "preparation": null
    except (KeyError, TypeError):
        raise PermanentJobError(
            f"provenance for {document} doesn't say which embedder sized its chunks; "
            "prepare it again"
        ) from None

    if prepared_for != embedder.identity:
        raise PermanentJobError(
            f"chunks of {document} were sized for {prepared_for!r}, "
            f"but the worker embeds with {embedder.identity!r}; prepare it again"
        )


def _load_parts(job: ClaimedJob, *, store: FileStore) -> PreparedParts:
    parts = load_prepared(store, job.document.id, job.version.version)
    if parts is None:
        raise PermanentJobError(
            f"document {job.document.id} v{job.version.version} has no prepared "
            "parts; prepare it again"
        )
    return parts


def _index_attributes(job: ClaimedJob) -> Attributes:
    return {
        "access_group_ids": tuple(str(group_id) for group_id in sorted(job.group_ids)),
        "collection_id": str(job.document.collection_id),
        "document_version": job.version.version,
    }
