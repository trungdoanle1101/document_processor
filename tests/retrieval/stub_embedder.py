"""A deterministic Embedder for tests: no model, each word hashed into a bucket."""

import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from document_processor.chunking.tokenizer import Tokenizer
from document_processor.retrieval.embedding import (
    DenseVector,
    DenseVectorSpec,
    Embedding,
    SparseVector,
    SparseVectorSpec,
    Vector,
    VectorSpec,
)


class StubEmbedder:
    """Texts sharing words get similar vectors. Records every call, so tests can check what was encoded.

    dense or sparse may be None to leave that vector out, but not both.
    """

    def __init__(
        self,
        *,
        dense: str | None = "dense",
        dimension: int = 8,
        sparse: str | None = "sparse",
        needs_idf: bool = False,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if dense is None and sparse is None:
            raise ValueError(
                "a stub embedder needs a dense vector, a sparse vector, or both"
            )
        specs: dict[str, VectorSpec] = {}
        if dense is not None:
            specs[dense] = DenseVectorSpec(dimension=dimension)
        if sparse is not None:
            specs[sparse] = SparseVectorSpec(needs_idf=needs_idf)
        self._dense = dense
        self._dimension = dimension
        self._sparse = sparse
        self._vector_specs: Mapping[str, VectorSpec] = MappingProxyType(specs)
        self._tokenizer = tokenizer
        self._identity = f"stub/{dense or '-'}:{dimension}/{sparse or '-'}:{'idf' if needs_idf else 'final'}"
        self.queries: list[str] = []
        self.passage_batches: list[list[str]] = []

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def vector_specs(self) -> Mapping[str, VectorSpec]:
        return self._vector_specs

    @property
    def tokenizer(self) -> Tokenizer | None:
        return self._tokenizer

    def encode_query(self, text: str) -> Embedding:
        self.queries.append(text)
        return self._encode(text)

    def encode_passages(self, texts: Sequence[str]) -> list[Embedding]:
        self.passage_batches.append(list(texts))
        return [self._encode(text) for text in texts]

    def _encode(self, text: str) -> Embedding:
        counts = Counter(_word_hash(word) for word in text.casefold().split())
        vectors: dict[str, Vector] = {}
        if self._dense is not None:
            buckets = [0.0] * self._dimension
            for word_hash, count in counts.items():
                buckets[word_hash % self._dimension] += count
            vectors[self._dense] = DenseVector(values=tuple(buckets))
        if self._sparse is not None:
            indices = tuple(sorted(counts))
            vectors[self._sparse] = SparseVector(
                indices=indices, values=tuple(float(counts[i]) for i in indices)
            )
        return Embedding(embedder_identity=self._identity, vectors=vectors)


def _word_hash(word: str) -> int:
    """Stable across runs and machines, unlike hash()."""
    return zlib.crc32(word.encode("utf-8"))
