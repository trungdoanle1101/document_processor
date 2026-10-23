from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from document_processor.chunking import Chunk, Chunker, save_chunks
from document_processor.core.cleaning import DEFAULT_STEPS, CleaningStep, clean_document
from document_processor.core.models import LogicalDocument, PhysicalDocument
from document_processor.core.serialization import (
    save_logical_document,
    save_physical_document,
)
from document_processor.logical import PROFILES, Profile, build_logical_document
from document_processor.pipeline.extraction import extract_physical_document


@dataclass(frozen=True, kw_only=True)
class SavedFiles:
    physical: Path
    logical: Path
    chunks: Path


@dataclass(frozen=True, kw_only=True)
class PipelineResult:
    source: Path
    physical: PhysicalDocument
    logical: LogicalDocument
    chunks: tuple[Chunk, ...]

    def save(self, out_dir: Path | str) -> SavedFiles:
        out_dir = Path(out_dir)
        stem = self.source.stem
        files = SavedFiles(
            physical=out_dir / f"{stem}.physical.json",
            logical=out_dir / f"{stem}.logical.json",
            chunks=out_dir / f"{stem}.chunks.jsonl",
        )
        save_physical_document(self.physical, files.physical)
        save_logical_document(self.logical, files.logical)
        save_chunks(self.chunks, files.chunks)
        return files


@dataclass(frozen=True, kw_only=True)
class PipelineFailure:
    source: Path
    error: str


@dataclass(frozen=True, kw_only=True)
class Pipeline:
    profile: Profile | str
    chunker: Chunker
    cleaning_steps: tuple[CleaningStep, ...] = DEFAULT_STEPS

    _resolved_profile: Profile = field(init=False, repr=False)

    def __post_init__(self) -> None:
        resolved_profile = (
            PROFILES.get(self.profile)
            if isinstance(self.profile, str)
            else self.profile
        )
        object.__setattr__(self, "_resolved_profile", resolved_profile)

    def run(self, path: Path | str) -> PipelineResult:
        path = Path(path)
        physical = extract_physical_document(path)
        physical = clean_document(physical, self.cleaning_steps)
        logical = build_logical_document(self._resolved_profile, physical)
        chunks = self.chunker.chunk(logical)

        return PipelineResult(
            source=path, physical=physical, logical=logical, chunks=chunks
        )

    def run_batch(
        self, paths: Iterable[Path | str]
    ) -> tuple[PipelineResult | PipelineFailure, ...]:
        outcomes: list[PipelineResult | PipelineFailure] = []
        for path in paths:
            try:
                outcomes.append(self.run(path))
            except (ValueError, KeyError, OSError) as error:
                outcomes.append(
                    PipelineFailure(
                        source=Path(path), error=f"{type(error).__name__}: {error}"
                    )
                )
        return tuple(outcomes)
