"""A deterministic Embedder for tests: no model, no torch.

Each word of the text (lowercased, split on non-word characters) counts once
in a dense bucket and once at a sparse index, both picked by a CRC32 of the
word. Texts sharing words therefore share vector weight, so a query made of
a passage's words ranks that passage first, which is all the search tests
need. It can produce a dense vector, a sparse one, or both, under any names.
Queries and passages are encoded the same way; the stub records every call so
a test can check which method a caller used.

The identity follows from the settings, so two stubs built alike give the
same vectors and the same identity, and stubs built differently do not.
"""

import re
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

_WORD = re.compile(r"\w+")
_SPARSE_INDEX_LIMIT = 2**31


def word_ids(text: str) -> Counter[int]:
    return Counter(zlib.crc32(word.encode("utf-8")) for word in _WORD.findall(text.lower()))


class StubEmbedder:
    def __init__(
        self,
        *,
        dense: str | None = "dense",
        dimension: int = 8,
        sparse: str | None = "sparse",
        needs_idf: bool = False,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        specs: dict[str, VectorSpec] = {}
        if dense is not None:
            specs[dense] = DenseVectorSpec(dimension=dimension)
        if sparse is not None:
            specs[sparse] = SparseVectorSpec(needs_idf=needs_idf)
        if not specs:
            raise ValueError("a stub needs a dense or a sparse vector")
        self._dense_name, self._sparse_name, self._dimension = dense, sparse, dimension
        self._vector_specs = MappingProxyType(specs)
        self._tokenizer = tokenizer
        self._identity = f"stub/{dense}:{dimension}/{sparse}:{'idf' if needs_idf else 'final'}"
        self.passage_batches: list[tuple[str, ...]] = []
        self.queries: list[str] = []

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
        return self._embed(text)

    def encode_passages(self, texts: Sequence[str]) -> list[Embedding]:
        self.passage_batches.append(tuple(texts))
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> Embedding:
        ids = word_ids(text)
        vectors: dict[str, Vector] = {}
        if self._dense_name is not None:
            dense = [0.0] * self._dimension
            for word_id, count in ids.items():
                dense[word_id % self._dimension] += count
            if not ids:
                dense[0] = 1.0  # a zero vector has no cosine similarity
            vectors[self._dense_name] = DenseVector(values=tuple(dense))
        if self._sparse_name is not None:
            weights: Counter[int] = Counter()
            for word_id, count in ids.items():
                weights[word_id % _SPARSE_INDEX_LIMIT] += count
            indices = tuple(sorted(weights))
            vectors[self._sparse_name] = SparseVector(indices=indices, values=tuple(float(weights[i]) for i in indices))
        return Embedding(vectors=vectors)