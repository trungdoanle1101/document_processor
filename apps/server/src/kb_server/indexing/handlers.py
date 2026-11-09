import logging
import shutil
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Self

from document_processor.chunking.chunker import Chunker
from document_processor.logical.profile import Profile
from document_processor.logical.profiles import PROFILES
from document_processor.preparation.preparer import PreparedDocument, Preparer
from kb_server.domain.documents import Document, DocumentVersion
from kb_server.domain.files import DOCX, TEXT, StoredFile
from kb_server.domain.jobs import IndexingJob
from kb_server.indexing.prepared import (
    PreparationSettings,
    PreparedParts,
    save_prepared,
)
from kb_server.storage.files import FileStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class JobTarget:
    """What a job is about, loaded when it's claimed."""

    job: IndexingJob
    document: Document
    version: DocumentVersion
    file: StoredFile


class PermanentJobError(Exception):
    """Retrying can't help (a missing or unreadable file): the job fails at once."""


@dataclass(frozen=True, kw_only=True)
class ChosenPreparation:
    prepared: PreparedDocument
    profile: str  # name@version that was used
    tried: tuple[str, ...]  # candidate profiles that found no structure, in order


@dataclass(frozen=True, kw_only=True)
class Preparation:
    """How documents are prepared: the Preparers, and the settings that describe them."""

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

    def prepare(self, path: Path) -> ChosenPreparation:
        """Candidates are tried in order; list them from most specific to most general."""
        tried: list[str] = []
        for preparer in self.preparers:
            prepared = preparer.prepare(path)
            profile = (
                f"{prepared.logical.profile_name}@{prepared.logical.profile_version}"
            )
            if not any(node.is_section for node in prepared.logical.nodes):
                tried.append(profile)
            else:
                break
        else:
            prepared = self.fallback_preparer.prepare(path)
            profile = (
                f"{prepared.logical.profile_name}@{prepared.logical.profile_version}"
            )
        return ChosenPreparation(prepared=prepared, profile=profile, tried=tuple(tried))


@dataclass(frozen=True, kw_only=True)
class PrepareResult:
    profile: str
    tried: tuple[str, ...]
    settings: dict[str, Any]  # preparation.settings.describe()
    queue_index: bool

    def provenance(self) -> dict[str, Any]:
        """Plain JSON, for document_versions.provenance["preparation"]"""
        return {
            "profile": self.profile,
            "tried": list(self.tried),
            "settings": self.settings,
        }


def handle_prepare(
    target: JobTarget, *, preparation: Preparation, store: FileStore
) -> PrepareResult:
    file = target.file
    extension = _suffix_for(file.media_type)

    with (
        tempfile.TemporaryDirectory() as tmp,
    ):
        work = Path(tmp)
        source = _copy_original(store, file, work / f"document{extension}")
        choice = _prepare_file(preparation, source)
        _save(store, target, choice, review_page=work / "review.html")
    _log_warnings(target, choice)
    return PrepareResult(
        profile=choice.profile,
        tried=choice.tried,
        settings=preparation.settings.describe(),
        queue_index=target.document.status == "active",
    )


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


def _suffix_for(media_type: str) -> str:
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
        raise PermanentJobError(f"stored file {file.sha256} is missing")
    with original, destination.open("wb") as out:
        shutil.copyfileobj(original, out)
    return destination


def _prepare_file(preparation: Preparation, path: Path) -> ChosenPreparation:
    try:
        return preparation.prepare(path)
    except OSError:
        raise
    except Exception as error:
        raise PermanentJobError("couldn't prepare file") from error


def _save(
    store: FileStore, target: JobTarget, choice: ChosenPreparation, *, review_page: Path
) -> None:
    save_prepared(
        store,
        target.document.id,
        target.version.version,
        PreparedParts(logical=choice.prepared.logical, chunks=choice.prepared.chunks),
    )
    choice.prepared.write_review_page(review_page)


def _log_warnings(target: JobTarget, choice: ChosenPreparation) -> None:
    counts = Counter(w.code for w in choice.prepared.summarize().warnings)
    if counts:
        logger.warning(
            "document %s v%d prepared with %s %s",
            target.document.id,
            target.version.version,
            choice.profile,
            ", ".join(f"{code}={n}" for code, n in sorted(counts.items())),
        )
