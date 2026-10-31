import math
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Literal, final

from document_processor.retrieval.embedding import (
    Embedding,
    VectorSpec,
    check_embedding,
)

type Scalar = str | int | float | bool | date
type AttributeValue = Scalar | tuple[str, ...] | tuple[int, ...]
type Attributes = Mapping[str, AttributeValue]

# None removes the attribute
type AttributeChanges = Mapping[str, AttributeValue | None]


@dataclass(frozen=True, kw_only=True)
class AnyOf:
    """Equals one of the values; on a list attribute, contains one of them."""

    values: tuple[str, ...] | tuple[int, ...] | tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("AnyOf needs at least one value")


@dataclass(frozen=True, kw_only=True)
class Range[T: (float, date)]:
    """Every bound given must hold."""

    above: T | None = None  # >
    at_least: T | None = None  # >=
    below: T | None = None  # <
    at_most: T | None = None  # <=

    def __post_init__(self) -> None:
        bounds = [
            bound
            for bound in (self.above, self.at_least, self.below, self.at_most)
            if bound is not None
        ]
        if not bounds:
            raise ValueError("Range needs at least one bound")
        for bound in bounds:
            _check_attribute_value(bound)


type Condition = AnyOf | Range[float] | Range[date]


@dataclass(frozen=True, kw_only=True, slots=True)
class ChunkKey:
    document_id: str
    chunk_id: str


@dataclass(frozen=True, kw_only=True)
class ChunkFilter:
    """The chunks meeting every part given. ChunkFilter() selects every chunk; use None, not (), for any."""

    document_ids: tuple[str, ...] | None = None
    chunk_keys: tuple[ChunkKey, ...] | None = None  # one of these chunks
    reference_keys: tuple[str, ...] | None = None  # has any of them
    attributes: Mapping[str, Condition] = MappingProxyType({})  # meets all of them

    def __post_init__(self) -> None:
        if self.document_ids == ():
            raise ValueError("document_ids must not be empty; use None for any")
        if self.chunk_keys == ():
            raise ValueError("chunk_keys must not be empty; use None for any")
        if self.reference_keys == ():
            raise ValueError("reference_keys must not be empty; use None for any")
        for name in self.attributes:
            _check_attribute_name(name)
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    @property
    def selects_everything(self) -> bool:
        return (
            self.document_ids is None
            and self.chunk_keys is None
            and self.reference_keys is None
            and not self.attributes
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class GeneratedEntryKey:
    document_id: str
    generator: str
    entry_id: str


@dataclass(frozen=True, kw_only=True)
class IndexedChunk:
    document_id: str
    chunk_id: str  # unique within its document
    position: int  # order within its document
    text: str  # the document's own words, never generated text
    embedded_text: str  # what was embedded
    headings: tuple[str, ...] = ()
    reference_keys: tuple[str, ...] = ()  # e.g. "dieu:14/khoan:2"
    attributes: Attributes = MappingProxyType({})

    def __post_init__(self) -> None:
        for name, value in self.attributes.items():
            _check_attribute_name(name)
            _check_attribute_value(value)
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    @property
    def key(self) -> ChunkKey:
        return ChunkKey(document_id=self.document_id, chunk_id=self.chunk_id)


@dataclass(frozen=True, kw_only=True)
class GeneratedEntry:
    """LLM-written text about some of a document's chunks, embedded as a passage."""

    document_id: str
    generator: str  # identity of what wrote it: model, revision, prompt version
    entry_id: str  # unique within its document and generator
    chunk_ids: tuple[str, ...]  # what finding it finds
    text: str

    def __post_init__(self) -> None:
        if not self.chunk_ids:
            raise ValueError(f"entry {self.entry_id!r} must lead to at least one chunk")
        if not self.text.strip():
            raise ValueError(f"entry {self.entry_id!r} has no text")

    @property
    def key(self) -> GeneratedEntryKey:
        return GeneratedEntryKey(
            document_id=self.document_id,
            generator=self.generator,
            entry_id=self.entry_id,
        )


# choice for generators; a tuple names generators to use
type GeneratedEntryChoice = Literal["all", "none"] | tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class SearchQuery:
    embedding: Embedding  # from Embedder.encode_query
    chunk_filter: ChunkFilter = field(default_factory=ChunkFilter)
    vector_names: tuple[str, ...] | None = None  # None: all, fused
    generated_entries: GeneratedEntryChoice = "all"
    limit: int = 10

    def __post_init__(self) -> None:
        _check_limit(self.limit)
        if self.vector_names == ():
            raise ValueError("vector_names must not be empty; use None for all")
        if self.vector_names is not None and len(set(self.vector_names)) != len(
            self.vector_names
        ):
            raise ValueError(f"vector_names repeats a name: {self.vector_names}")
        if self.generated_entries == ():
            raise ValueError(
                'generated_entries must name generators; use "none" for no entries'
            )


@dataclass(frozen=True, kw_only=True)
class ChunkHit:
    chunk: IndexedChunk
    # higher is better; compare only within one result list
    score: float

    # entries that gave its best match on some vector, best first
    found_through: tuple[GeneratedEntry, ...] = ()


class ChunkIndex(ABC):
    """One vector space of chunks and generated entries, built by one embedder.

    Public methods check their arguments, then call the hook an implementation
    provides; writes run one at a time.
    """

    def __init__(self) -> None:
        self._write_lock = threading.Lock()

    @property
    @abstractmethod
    def embedder_identity(self) -> str: ...

    @property
    @abstractmethod
    def vector_specs(self) -> Mapping[str, VectorSpec]: ...

    @final
    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[IndexedChunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        """Make these the document's only chunks, and remove its generated entries. No chunks removes it."""
        _check_replacement(
            document_id, chunks, embeddings, self.embedder_identity, self.vector_specs
        )
        with self._write_lock:
            self._replace_document(document_id, chunks, embeddings)

    @final
    def replace_generated(
        self,
        document_id: str,
        generator: str,
        entries: Sequence[GeneratedEntry],
        embeddings: Sequence[Embedding],
    ) -> None:
        """Make these the document's only entries from this generator.

        Each entry must lead to chunks the document has now.
        """
        with self._write_lock:
            current = self._list_chunks(ChunkFilter(document_ids=(document_id,)), None)
            document_chunks = {chunk.chunk_id: chunk for chunk in current}
            _check_generated_replacement(
                document_id,
                generator,
                entries,
                embeddings,
                self.embedder_identity,
                self.vector_specs,
                document_chunks,
            )
            self._replace_generated(
                document_id, generator, entries, embeddings
            )

    @final
    def delete_chunks(self, chunk_filter: ChunkFilter) -> None:
        """Remove the selected chunks; entries stop leading to them, and an entry left with none is removed."""
        _check_not_everything(chunk_filter, "delete")
        with self._write_lock:
            self._delete_chunks(chunk_filter)

    @final
    def update_attributes(
        self, chunk_filter: ChunkFilter, changes: AttributeChanges
    ) -> None:
        """Set or remove attributes on the selected chunks; their entries follow."""
        _check_update(chunk_filter, changes)
        with self._write_lock:
            self._update_attributes(chunk_filter, changes)

    @final
    def list_chunks(
        self, chunk_filter: ChunkFilter, *, limit: int
    ) -> list[IndexedChunk]:
        """The selected chunks in document order (document id, then position); never entries."""
        _check_limit(limit)
        return self._list_chunks(chunk_filter, limit)

    @final
    def search(self, query: SearchQuery) -> list[ChunkHit]:
        """The selected chunks, best first, each once.

        A chunk is ranked by its best match: its own text or one of its entries. An entry
        only finds those of its chunks the filter selects. One vector scores by similarity:
        cosine if dense; dot product if sparse, weighted by IDF if its spec says so.
        Several are fused by reciprocal rank over each vector's best candidates (at least
        limit chunks), so a chunk outside every vector's candidates isn't found.
        """
        _check_search(query, self.embedder_identity, self.vector_specs)
        return self._search(query)

    # Hooks: implementations provide these, and receive only checked arguments.

    @abstractmethod
    def _replace_document(
        self,
        document_id: str,
        chunks: Sequence[IndexedChunk],
        embeddings: Sequence[Embedding],
    ) -> None: ...

    @abstractmethod
    def _replace_generated(
        self,
        document_id: str,
        generator: str,
        entries: Sequence[GeneratedEntry],
        embeddings: Sequence[Embedding],
    ) -> None: ...

    @abstractmethod
    def _delete_chunks(self, chunk_filter: ChunkFilter) -> None: ...

    @abstractmethod
    def _update_attributes(
        self, chunk_filter: ChunkFilter, changes: AttributeChanges
    ) -> None: ...

    @abstractmethod
    def _list_chunks(
        self, chunk_filter: ChunkFilter, limit: int | None
    ) -> list[IndexedChunk]:
        """In document order; None: all of them."""

    @abstractmethod
    def _search(self, query: SearchQuery) -> list[ChunkHit]: ...


# Checks ----------------------------------------------------------------------------------

_ATTRIBUTE_NAME = re.compile(
    r"[a-z][a-z0-9_]*"
)  # a dot would be a nested path in Qdrant


def _check_attribute_name(name: str) -> None:
    if not _ATTRIBUTE_NAME.fullmatch(name):
        raise ValueError(
            f"attribute name {name!r} must be lower-case letters, digits and underscores"
        )


def _check_attribute_value(value: AttributeValue) -> None:
    match value:
        case datetime():
            raise TypeError(f"attribute values take dates, not datetimes: {value!r}")
        case float() if not math.isfinite(value):
            raise ValueError(f"attribute values must be finite numbers: {value!r}")
        case _:
            pass


def _check_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")


def _check_replacement(
    document_id: str,
    chunks: Sequence[IndexedChunk],
    embeddings: Sequence[Embedding],
    embedder_identity: str,
    vector_specs: Mapping[str, VectorSpec],
) -> None:
    _check_one_embedding_each("chunks", len(chunks), len(embeddings))
    seen: set[str] = set()
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        if chunk.document_id != document_id:
            raise ValueError(
                f"chunk {chunk.chunk_id!r} belongs to {chunk.document_id!r}, not {document_id!r}"
            )
        if chunk.chunk_id in seen:
            raise ValueError(f"chunk id {chunk.chunk_id!r} appears twice")
        seen.add(chunk.chunk_id)
        _check_embedding_of(
            f"chunk {chunk.chunk_id!r}", embedding, embedder_identity, vector_specs
        )


def _check_generated_replacement(
    document_id: str,
    generator: str,
    entries: Sequence[GeneratedEntry],
    embeddings: Sequence[Embedding],
    embedder_identity: str,
    vector_specs: Mapping[str, VectorSpec],
    document_chunks: Mapping[str, IndexedChunk],
) -> None:
    _check_one_embedding_each("entries", len(entries), len(embeddings))
    seen: set[str] = set()
    for entry, embedding in zip(entries, embeddings, strict=True):
        if (entry.document_id, entry.generator) != (document_id, generator):
            raise ValueError(
                f"entry {entry.entry_id!r} belongs to document {entry.document_id!r} and generator "
                f"{entry.generator!r}, not document {document_id!r} and generator {generator!r}"
            )
        if entry.entry_id in seen:
            raise ValueError(f"entry id {entry.entry_id!r} appears twice")
        seen.add(entry.entry_id)
        missing = [
            chunk_id for chunk_id in entry.chunk_ids if chunk_id not in document_chunks
        ]
        if missing:
            raise ValueError(
                f"entry {entry.entry_id!r} leads to chunks {document_id!r} doesn't have: {', '.join(missing)}"
            )
        _check_embedding_of(
            f"entry {entry.entry_id!r}", embedding, embedder_identity, vector_specs
        )


def _check_not_everything(chunk_filter: ChunkFilter, action: str) -> None:
    if chunk_filter.selects_everything:
        raise ValueError(f"refusing to {action} every chunk")


def _check_update(chunk_filter: ChunkFilter, changes: AttributeChanges) -> None:
    if not changes:
        raise ValueError("no attribute changes given")
    _check_not_everything(chunk_filter, "update")
    for name, value in changes.items():
        _check_attribute_name(name)
        if value is not None:
            _check_attribute_value(value)


def _check_search(
    query: SearchQuery, embedder_identity: str, vector_specs: Mapping[str, VectorSpec]
) -> None:
    _check_embedding_of("query", query.embedding, embedder_identity, vector_specs)
    unknown = [name for name in query.vector_names or () if name not in vector_specs]
    if unknown:
        raise ValueError(
            f"no vectors named {', '.join(unknown)}; the space has {', '.join(sorted(vector_specs))}"
        )


def _check_one_embedding_each(what: str, count: int, embedding_count: int) -> None:
    if count != embedding_count:
        raise ValueError(f"{count} {what} but {embedding_count} embeddings")


def _check_embedding_of(
    owner: str,
    embedding: Embedding,
    embedder_identity: str,
    vector_specs: Mapping[str, VectorSpec],
) -> None:
    try:
        check_embedding(embedding, embedder_identity, vector_specs)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{owner}: {error}") from error
