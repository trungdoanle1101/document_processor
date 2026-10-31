from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from document_processor.adapters import ADAPTERS, Adapter
from document_processor.chunking import Chunk, Chunker, save_chunks
from document_processor.core.cleaning import DEFAULT_STEPS, CleaningStep, clean_document
from document_processor.core.models import (
    LogicalDocument,
    PhysicalDocument,
    TableSegment,
)
from document_processor.core.registry import Registry
from document_processor.core.serialization import (
    save_logical_document,
    save_physical_document,
)
from document_processor.logical import PROFILES, Profile, build_logical_document
from document_processor.pipeline.extraction import extract_physical_document
from document_processor.pipeline.review import (
    ResultSummary,
    ReviewWarning,
    build_review_data,
)
from document_processor.pipeline.review_page import (
    render_review_frame,
    render_review_html,
)


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

    def write_review_page(self, path: Path | str) -> Path:
        """Write the review page for this result as one self-contained HTML file; returns its path"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_review_html(build_review_data(self)), encoding="utf-8")
        return path

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

        chunks_per_table = Counter(
            c.start for c in self.chunks if c.table_rows is not None
        )
        for segment_index, chunk_count in chunks_per_table.items():
            segment = self.logical.segments[segment_index]
            assert isinstance(segment, TableSegment)
            if chunk_count > 1 and segment.table.n_header_rows == 0:
                first_chunk = next(
                    c
                    for c in self.chunks
                    if c.table_rows is not None and c.start == segment_index
                )
                warnings.append(
                    ReviewWarning(
                        code="table_without_header",
                        message=(
                            f"table at segment {segment_index} (block {segment.block_id}) is split into {chunk_count} chunks "
                            "but has no header rows: chunks after the first have no column names. "
                            "Hint: mark the header row (Repeat Header Rows in Word) or make it bold."
                        ),
                        chunk_id=first_chunk.id,
                        segment=segment_index,
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

    def _repr_html_(self) -> str:
        return render_review_frame(build_review_data(self))


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
