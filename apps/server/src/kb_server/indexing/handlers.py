from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Self

from document_processor.chunking.chunker import Chunker
from document_processor.logical.profile import Profile
from document_processor.logical.profiles import PROFILES
from document_processor.preparation.preparer import Preparer
from kb_server.domain.documents import Document, DocumentVersion
from kb_server.domain.files import StoredFile
from kb_server.domain.jobs import IndexingJob
from kb_server.indexing.prepared import PreparationSettings


@dataclass(frozen=True, kw_only=True)
class JobTarget:
    """What a job is about, loaded when it's claimed."""

    job: IndexingJob
    document: Document
    version: DocumentVersion
    file: StoredFile


@dataclass(frozen=True, kw_only=True)
class Preparation:
    """How documents are prepared: the Preparers, and the settings that describe it."""

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
