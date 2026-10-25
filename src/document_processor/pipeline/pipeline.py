from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from document_processor.adapters import ADAPTERS, Adapter
from document_processor.chunking import Chunk, Chunker, save_chunks
from document_processor.core.cleaning import DEFAULT_STEPS, CleaningStep, clean_document
from document_processor.core.models import LogicalDocument, PhysicalDocument
from document_processor.core.registry import Registry
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
class ReviewWarning:
    code: str
    message: str
    chunk_id: str | None = None
    segment: int | None = None


@dataclass(frozen=True, kw_only=True)
class ResultSummary:
    source: Path
    block_count: int
    segment_count: int
    nodes_by_levels: tuple[tuple[str, int], ...]
    front_matter: tuple[int, int]
    end_matter: tuple[int, int]
    chunk_count: int
    chunk_tokens: tuple[int, ...]
    warnings: tuple[ReviewWarning, ...]
    max_tokens: int

    @property
    def smallest_chunk_tokens(self) -> int | None:
        return min(self.chunk_tokens) if self.chunk_tokens else None

    @property
    def median_chunk_tokens(self) -> float | None:
        return median(self.chunk_tokens) if self.chunk_tokens else None

    @property
    def largest_chunk_tokens(self) -> int | None:
        return max(self.chunk_tokens) if self.chunk_tokens else None

    def format(self) -> str:
        node_count = sum(count for _, count in self.nodes_by_levels)
        levels = (
            " · ".join(f"{level} {count}" for level, count in self.nodes_by_levels)
            or "none"
        )
        if self.chunk_tokens:
            sizes = (
                f"min {self.smallest_chunk_tokens} · median {round(self.median_chunk_tokens or 0)}"
                f" · max {self.largest_chunk_tokens} (budget {self.max_tokens})"
            )
        else:
            sizes = "none"
        lines = [
            self.source.name,
            f"  {self.block_count} blocks → {self.segment_count} segments, {node_count} nodes, {self.chunk_count} chunks",
            f"  front matter [{self.front_matter[0]}, {self.front_matter[1]}) · end matter [{self.end_matter[0]}, {self.end_matter[1]})",
            f"  nodes: {levels}",
            f"  chunk tokens: {sizes}",
        ]
        lines.extend(f"  ⚠ {w.code}  {w.message}" for w in self.warnings)
        if not self.warnings:
            lines.append("  no warnings")
        return "\n".join(lines)


@dataclass(frozen=True, kw_only=True)
class PipelineResult:
    source: Path
    physical: PhysicalDocument
    logical: LogicalDocument
    chunks: tuple[Chunk, ...]
    max_tokens: int
    level_order: tuple[str, ...]

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

    def summarize(self) -> ResultSummary:

        warnings: list[ReviewWarning] = []

        over_budget = [c for c in self.chunks if c.token_count > self.max_tokens]

        warnings.extend(
            [
                ReviewWarning(
                    code="chunk_over_budget",
                    message=f"chunk {c.id} has {c.token_count} tokens (budget {self.max_tokens})",
                    chunk_id=c.id,
                )
                for c in over_budget
            ]
        )

        if not self.logical.nodes:
            warnings.append(
                ReviewWarning(
                    code="no_headings",
                    message=f"profile {self.logical.profile_name!r} produces no node in the logical document. Hint: Determine the suitability of the profile for this document.",
                )
            )

        counts = Counter([node.level for node in self.logical.nodes])
        rank = {name: index for index, name in enumerate(self.level_order)}
        nodes_by_level = tuple(sorted(counts.items(), key=lambda item: rank[item[0]]))

        return ResultSummary(
            source=self.source,
            block_count=len(self.physical.blocks),
            segment_count=len(self.logical.segments),
            chunk_count=len(self.chunks),
            nodes_by_levels=nodes_by_level,
            front_matter=(0, self.logical.body_start),
            end_matter=(self.logical.body_end, len(self.logical.segments)),
            chunk_tokens=tuple(chunk.token_count for chunk in self.chunks),
            warnings=tuple(warnings),
            max_tokens=self.max_tokens,
        )


@dataclass(frozen=True, kw_only=True)
class PipelineFailure:
    source: Path
    error: str


@dataclass(frozen=True, kw_only=True)
class Pipeline:
    profile: Profile | str
    chunker: Chunker
    cleaning_steps: tuple[CleaningStep, ...] = DEFAULT_STEPS
    adapters: Registry[Adapter] = ADAPTERS

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
        physical = extract_physical_document(path, self.adapters)
        physical = clean_document(physical, self.cleaning_steps)
        logical = build_logical_document(self._resolved_profile, physical)
        chunks = self.chunker.chunk(logical)

        levels = self._resolved_profile.levels or ()

        return PipelineResult(
            source=path,
            physical=physical,
            logical=logical,
            chunks=chunks,
            max_tokens=self.chunker.max_tokens,
            level_order=tuple(level.name for level in levels),
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
