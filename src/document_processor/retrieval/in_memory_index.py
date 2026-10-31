from collections.abc import Mapping, Sequence
from types import MappingProxyType

from document_processor.retrieval.embedding import VectorSpec, Embedding
from document_processor.retrieval.index import (
    ChunkIndex,
    ChunkKey,
    ChunkHit,
    ChunkFilter,
    GeneratedEntry,
    GeneratedEntryKey,
    IndexedChunk,
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

    def _replace_document(
        self,
        document_id: str,
        chunks: Sequence[IndexedChunk],
        embeddings: Sequence[Embedding],
    ) -> None:
        self._entries = {
            key: stored
            for key, stored in self._entries.items()
            if stored[0].document_id != document_id
        }
        kept = {
            key: stored
            for key, stored in self._chunks.items()
            if stored[0].document_id != document_id
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
        embeddings: Sequence[Embedding]
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
