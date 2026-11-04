"""Add, refresh and remove documents, keeping the index and the registry in step.

Work is skipped when nothing that produced a document's chunks has changed, and a
change to who may read a document only rewrites its attributes, without embedding again.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType

from document_processor.chunking import Chunker
from document_processor.core.identity import compute_hash
from document_processor.logical import Profile
from document_processor.pipeline import Pipeline
from document_processor.retrieval.attributes import AttributeValue
from document_processor.retrieval.embedding import Embedder
from document_processor.retrieval.index import ChunkFilter, ChunkIndex
from document_processor.retrieval.ingest import index_document
from knowledge_base.access import ACCESS_GROUP, PART
from knowledge_base.registry import (
    DocumentRecord,
    DocumentRegistry,
    Provenance,
    provenance_differences,
)

_RESERVED = frozenset({ACCESS_GROUP, PART})


@dataclass(frozen=True, kw_only=True)
class IndexingResult:
    record: DocumentRecord
    action: str  # "indexed", "attributes updated" or "unchanged"


@dataclass(frozen=True, kw_only=True)
class StaleDocument:
    record: DocumentRecord
    reasons: tuple[str, ...]  # changed provenance parts, or "source missing"


class Indexer:
    def __init__(
        self,
        *,
        profile: Profile,
        chunker: Chunker,
        chunker_name: str,  # names the chunker and its settings in provenance, e.g. "outline/512"
        embedder: Embedder,
        index: ChunkIndex,
        registry: DocumentRegistry,
        collection: str,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._profile = profile
        self._pipeline = Pipeline(profile=profile, chunker=chunker)
        self._chunker_name = chunker_name
        self._embedder = embedder
        self._index = index
        self._registry = registry
        self._collection = collection
        self._now = now

    def add_document(
        self,
        path: Path | str,
        *,
        access_group: str,
        document_id: str | None = None,
        title: str | None = None,
        attributes: Mapping[str, AttributeValue] = MappingProxyType({}),
        force: bool = False,
    ) -> IndexingResult:
        """Index a file under a stable id (its stem by default), or update what changed."""
        path = Path(path)
        reserved = sorted(_RESERVED & set(attributes))
        if reserved:
            raise ValueError(f"attribute(s) {', '.join(reserved)} are set by the app")
        document_id = document_id or path.stem
        title = title or path.stem
        provenance = self.provenance_of(path)
        current = self._registry.get(document_id)

        if current is not None and not force and current.provenance == provenance:
            if current.access_group == access_group and dict(
                current.attributes
            ) == dict(attributes):
                return IndexingResult(record=current, action="unchanged")
            self._update_attributes(current, access_group, attributes)
            record = self._save_version(
                document_id=document_id,
                title=title,
                source=str(path),
                access_group=access_group,
                attributes=attributes,
                provenance=provenance,
                chunk_count=current.chunk_count,
            )
            return IndexingResult(record=record, action="attributes updated")

        result = self._pipeline.run(path)
        index_document(
            result.chunks,
            document_id=document_id,
            embedder=self._embedder,
            index=self._index,
            attributes={**attributes, ACCESS_GROUP: access_group},
        )
        record = self._save_version(
            document_id=document_id,
            title=title,
            source=str(path),
            access_group=access_group,
            attributes=attributes,
            provenance=provenance,
            chunk_count=len(result.chunks),
        )
        return IndexingResult(record=record, action="indexed")

    def remove_document(self, document_id: str) -> None:
        if self._registry.get(document_id) is None:
            raise KeyError(f"no document {document_id!r}")
        self._index.delete_chunks(ChunkFilter(document_ids=(document_id,)))
        self._registry.delete(document_id)

    def find_stale(self) -> list[StaleDocument]:
        """Documents whose chunks something else would now produce."""
        stale: list[StaleDocument] = []
        for record in self._registry.list_documents():
            source = Path(record.source)
            if not source.is_file():
                stale.append(StaleDocument(record=record, reasons=("source missing",)))
                continue
            reasons = provenance_differences(
                record.provenance, self.provenance_of(source)
            )
            if reasons:
                stale.append(StaleDocument(record=record, reasons=tuple(reasons)))
        return stale

    def refresh_stale(self) -> list[IndexingResult]:
        """Index every stale document again, from its source; missing sources are left alone."""
        return [
            self.add_document(
                stale.record.source,
                document_id=stale.record.document_id,
                title=stale.record.title,
                access_group=stale.record.access_group,
                attributes=stale.record.attributes,
            )
            for stale in self.find_stale()
            if "source missing" not in stale.reasons
        ]

    def provenance_of(self, path: Path) -> Provenance:
        """What indexing this file now would be produced by."""
        return Provenance(
            source_hash=compute_hash(path.read_bytes()),
            profile=f"{self._profile.name}@{self._profile.version}",
            chunker=self._chunker_name,
            embedder=self._embedder.identity,
            collection=self._collection,
        )

    def _update_attributes(
        self,
        current: DocumentRecord,
        access_group: str,
        attributes: Mapping[str, AttributeValue],
    ) -> None:
        removed = {name: None for name in current.attributes if name not in attributes}
        self._index.update_attributes(
            ChunkFilter(document_ids=(current.document_id,)),
            {**removed, **attributes, ACCESS_GROUP: access_group},
        )

    def _save_version(
        self,
        *,
        document_id: str,
        title: str,
        source: str,
        access_group: str,
        attributes: Mapping[str, AttributeValue],
        provenance: Provenance,
        chunk_count: int,
    ) -> DocumentRecord:
        record = DocumentRecord(
            document_id=document_id,
            title=title,
            source=source,
            access_group=access_group,
            attributes=attributes,
            version=len(self._registry.history(document_id)) + 1,
            provenance=provenance,
            chunk_count=chunk_count,
            indexed_at=self._now(),
        )
        self._registry.save(record)
        return record


def parse_attribute(value: object) -> AttributeValue:
    """An attribute from the command line or JSON: YYYY-MM-DD text is a date, a list a tuple."""
    match value:
        case bool() | int() | float():
            return value
        case str():
            try:
                return date.fromisoformat(value)
            except ValueError:
                return value
        case list() if all(isinstance(item, str) for item in value):
            return tuple(value)
        case list() if all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            return tuple(value)
    raise ValueError(f"unsupported attribute value {value!r}")
