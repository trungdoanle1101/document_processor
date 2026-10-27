"""A deterministic Embedder for tests: no model, no torch.

Each word of the text (lowercased, split on non-word characters) counts once
in a dense bucket and once at a sparse index, both picked by a CRC32 of the
word. Texts sharing words therefore share vector weight, so a query made of a
chunk's words ranks that chunk first, which is all the search tests need.
Queries and documents are encoded the same way; the stub records every call
so a test can check which method a caller used.

The model identity follows from the settings, so two stubs built alike give
the same vectors and the same identity, and stubs built differently do not.
"""

import re
import zlib
from collections import Counter
from collections.abc import Sequence

from document_processor.retrieval.embedder import Embedding, SparseVector


_WORD = re.compile(r"\w+")
_SPARSE_INDEX_LIMIT = 2**31


def word_ids(text: str) -> Counter[int]:
    return Counter(zlib.crc32(word.encode("utf-8")) for word in _WORD.findall(text.lower()))


class StubEmbedder:
    def __init__(self, *, dimension: int = 8, sparse: bool = True) -> None:
        self._dimension = dimension
        self._sparse = sparse
        self.document_batches: list[tuple[str, ...]] = []
        self.queries: list[str] = []

    @property
    def model_identity(self) -> str:
        return f"stub/dim{self._dimension}/{'sparse' if self._sparse else 'dense'}"

    @property
    def dense_dimension(self) -> int:
        return self._dimension

    @property
    def has_sparse(self) -> bool:
        return self._sparse

    def encode_documents(self, texts: Sequence[str]) -> list[Embedding]:
        self.document_batches.append(tuple(texts))
        return [self._embed(text) for text in texts]

    def encode_query(self, text: str) -> Embedding:
        self.queries.append(text)
        return self._embed(text)

    def _embed(self, text: str) -> Embedding:
        ids = word_ids(text)
        dense = [0.0] * self._dimension
        for word_id, count in ids.items():
            dense[word_id % self._dimension] += count
        if not ids:
            dense[0] = 1.0  # a zero vector has no cosine similarity
        sparse = None
        if self._sparse:
            weights: Counter[int] = Counter()
            for word_id, count in ids.items():
                weights[word_id % _SPARSE_INDEX_LIMIT] += count
            indices = tuple(sorted(weights))
            sparse = SparseVector(indices=indices, values=tuple(float(weights[i]) for i in indices))
        return Embedding(dense=tuple(dense), sparse=sparse)