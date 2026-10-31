import math
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import replace
from types import MappingProxyType
from typing import assert_never

from document_processor.retrieval._fusion import (
    BestMatch,
    fuse,
)
from document_processor.retrieval.attributes import AttributeChanges, Condition
from document_processor.retrieval.embedding import (
    DenseVector,
    DenseVectorSpec,
    Embedding,
    SparseVector,
    SparseVectorSpec,
    VectorSpec,
)
from document_processor.retrieval.index import (
    ChunkFilter,
    ChunkHit,
    ChunkIndex,
    ChunkKey,
    GeneratedEntry,
    GeneratedEntryChoice,
    GeneratedEntryKey,
    IndexedChunk,
    SearchQuery,
)


class InMemoryChunkIndex(ChunkIndex):
    def __init__(
        self, *, embedder_identity: str, vector_specs: Mapping[str, VectorSpec]
    ):
        super().__init__()
        self._embedder_identity = embedder_identity
        self._vector_specs = MappingProxyType(dict(vector_specs))
        self._chunks: dict[ChunkKey, tuple[IndexedChunk, Embedding]] = {}
        self._entries: dict[GeneratedEntryKey, tuple[GeneratedEntry, Embedding]] = {}

    @property
    def embedder_identity(self) -> str:
        return self._embedder_identity

    @property
    def vector_specs(self) -> Mapping[str, VectorSpec]:
        return self._vector_specs

    @property
    def supported_conditions(self) -> AbstractSet[type[Condition]] | None:
        return None

    def _replace_document(
        self,
        document_id: str,
        chunks: Sequence[IndexedChunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        self._entries = {
            key: stored
            for key, stored in self._entries.items()
            if key.document_id != document_id
        }
        kept = {
            key: stored
            for key, stored in self._chunks.items()
            if key.document_id != document_id
        }
        added = {
            chunk.key: (chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        }
        self._chunks = kept | added

    def _replace_generated(
        self,
        document_id: str,
        generator: str,
        entries: Sequence[GeneratedEntry],
        embeddings: Sequence[Embedding],
    ) -> None:
        kept = {
            key: stored
            for key, stored in self._entries.items()
            if (key.document_id, key.generator) != (document_id, generator)
        }
        added = {
            entry.key: (entry, embedding)
            for entry, embedding in zip(entries, embeddings, strict=True)
        }

        self._entries = kept | added

    def _delete_chunks(self, chunk_filter: ChunkFilter) -> None:
        deleted = {
            key
            for key, (chunk, _) in self._chunks.items()
            if _select_chunk(chunk, chunk_filter)
        }
        entries: dict[GeneratedEntryKey, tuple[GeneratedEntry, Embedding]] = {}
        for key, (entry, embedding) in self._entries.items():
            remaining_chunk_ids = tuple(
                chunk_id
                for chunk_id in entry.chunk_ids
                if ChunkKey(document_id=entry.document_id, chunk_id=chunk_id)
                not in deleted
            )
            if remaining_chunk_ids:
                entries[key] = (
                    replace(entry, chunk_ids=remaining_chunk_ids),
                    embedding,
                )
        # entry first so a search never sees an entry pointing at a deleted chunk
        self._entries = entries
        self._chunks = {
            key: stored for key, stored in self._chunks.items() if key not in deleted
        }

    def _update_attributes(
        self, chunk_filter: ChunkFilter, changes: AttributeChanges
    ) -> None:
        chunks: dict[ChunkKey, tuple[IndexedChunk, Embedding]] = {}
        for key, stored in self._chunks.items():
            if not _select_chunk(stored[0], chunk_filter):
                chunks[key] = stored
            else:
                chunks[key] = (_update_chunk(stored[0], changes), stored[1])
        self._chunks = chunks

    def _list_chunks(
        self, chunk_filter: ChunkFilter, limit: int | None
    ) -> list[IndexedChunk]:
        selected = [
            chunk
            for chunk, _ in self._chunks.values()
            if _select_chunk(chunk, chunk_filter)
        ]
        selected.sort(key=lambda chunk: (chunk.document_id, chunk.position))
        return selected if limit is None else selected[:limit]

    def _search(self, query: SearchQuery) -> list[ChunkHit]:
        all_chunks, all_entries = self._chunks, self._entries

        # Filter chunk and entries
        chunks = {
            key: stored
            for key, stored in all_chunks.items()
            if _select_chunk(stored[0], query.chunk_filter)
        }
        entries = {
            key: stored
            for key, stored in all_entries.items()
            if _allow_entry(stored[0], query.generated_entries)
        }

        all_embeddings = [embedding for _, embedding in all_chunks.values()] + [
            embedding for _, embedding in all_entries.values()
        ]

        # Get rankings for each vector_name
        vector_names_to_inspect = query.vector_names or tuple(self._vector_specs)

        rankings: list[list[BestMatch]] = []
        for vector_name in vector_names_to_inspect:
            scorer = _make_scorer(
                vector_name,
                self._vector_specs[vector_name],
                query.embedding,
                all_embeddings,
            )
            rankings.append(_rank_chunks(scorer, chunks, entries))

        return fuse(rankings, query.limit)


def _select_chunk(chunk: IndexedChunk, chunk_filter: ChunkFilter) -> bool:
    if (
        chunk_filter.document_ids is not None
        and chunk.document_id not in chunk_filter.document_ids
    ):
        return False

    if chunk_filter.chunk_keys is not None and chunk.key not in chunk_filter.chunk_keys:
        return False

    if chunk_filter.reference_keys is not None and not (
        set(chunk_filter.reference_keys) & set(chunk.reference_keys)
    ):
        return False

    for name, condition in chunk_filter.attributes.items():
        value = chunk.attributes.get(name)
        if not condition.match(value):
            return False
    return True


def _update_chunk(chunk: IndexedChunk, changes: AttributeChanges) -> IndexedChunk:
    attributes = dict(chunk.attributes)
    for name, value in changes.items():
        if value is None:
            attributes.pop(name, None)
        else:
            attributes[name] = value
    return replace(chunk, attributes=attributes)


def _allow_entry(entry: GeneratedEntry, choice: GeneratedEntryChoice) -> bool:
    if choice == "all":
        return True

    if choice == "none":
        return False

    return entry.generator in choice


# None: not found for this vector
type _Scorer = Callable[[Embedding], float | None]


def _rank_chunks(
    scorer: _Scorer,
    chunks: Mapping[ChunkKey, tuple[IndexedChunk, Embedding]],
    entries: Mapping[GeneratedEntryKey, tuple[GeneratedEntry, Embedding]],
) -> list[BestMatch]:
    best: dict[ChunkKey, BestMatch] = {}
    # score of chunk's own text
    for key, (chunk, embedding) in chunks.items():
        score = scorer(embedding)
        if score is not None:
            best[key] = BestMatch(chunk=chunk, score=score, entry=None)

    # score of entries
    for entry, embedding in entries.values():
        entry_score = scorer(embedding)
        if entry_score is None:
            continue
        for chunk_id in entry.chunk_ids:
            key = ChunkKey(document_id=entry.document_id, chunk_id=chunk_id)
            if key in chunks and (key not in best or entry_score > best[key].score):
                best[key] = BestMatch(
                    chunk=chunks[key][0], score=entry_score, entry=entry
                )
    return sorted(best.values(), key=lambda match: match.score, reverse=True)


def _make_scorer(
    vector_name: str,
    spec: VectorSpec,
    query: Embedding,
    all_embeddings: Sequence[Embedding],
) -> _Scorer:
    if isinstance(spec, DenseVectorSpec):
        dense_query = query.dense(vector_name)
        return lambda embedding: _cosine(dense_query, embedding.dense(vector_name))

    if isinstance(spec, SparseVectorSpec):
        sparse_query = query.sparse(vector_name)
        weights_by_index = _idf(vector_name, all_embeddings) if spec.needs_idf else None
        return lambda embedding: _dot(
            sparse_query, embedding.sparse(vector_name), weights_by_index
        )

    assert_never(spec)


def _cosine(query: DenseVector, vector: DenseVector) -> float:
    norms = math.hypot(*query.values) * math.hypot(*vector.values)
    if norms == 0:
        return 0.0
    return math.sumprod(query.values, vector.values) / norms


def _dot(
    query: SparseVector, vector: SparseVector, idf: Mapping[int, float] | None
) -> float | None:
    vectors_values = dict(zip(vector.indices, vector.values, strict=True))
    query_values = dict(zip(query.indices, query.values, strict=True))
    shared_indices = set(vectors_values.keys()) & set(query_values.keys())
    if not shared_indices:
        return None

    return sum(
        vectors_values[index]
        * query_values[index]
        * (idf[index] if idf is not None else 1.0)
        for index in shared_indices
    )


def _idf(vector_name: str, all_embeddings: Sequence[Embedding]) -> dict[int, float]:
    """Qdrant's formula ln((N - n + 0.5) / (n + 0.5) + 1), over everything stored"""
    # Counts the number of times each index occurs in all embeddings
    counts: dict[int, int] = {}
    for embedding in all_embeddings:
        for index in set(embedding.sparse(vector_name).indices):
            counts[index] = counts.get(index, 0) + 1
    total = len(all_embeddings)
    weights_by_index: dict[int, float] = {}
    for index, n in counts.items():
        weights_by_index[index] = math.log(1 + (total - n + 0.5) / (n + 0.5))
    return weights_by_index
