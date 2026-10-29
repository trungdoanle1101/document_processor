from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from document_processor.chunking.tokenizer import Tokenizer


@dataclass(frozen=True, kw_only=True)
class DenseVector:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("a dense vector must not be empty")


@dataclass(frozen=True, kw_only=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"{len(self.indices)} indices but {len(self.values)} values"
            )

        if any(idx < 0 for idx in self.indices):
            raise ValueError(f"negative index {min(self.indices)}")

        seen_indices: set[int] = set()
        for idx in self.indices:
            if idx in seen_indices:
                raise ValueError(f"index {idx} appears more than once")
            seen_indices.add(idx)


type Vector = DenseVector | SparseVector


@dataclass(frozen=True, kw_only=True)
class DenseVectorSpec:
    dimension: int

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}")


@dataclass(frozen=True, kw_only=True)
class SparseVectorSpec:
    # True when the values are term frequencies and the index must weight terms by their rarity in the collection
    # False when they are final
    needs_idf: bool = False


type VectorSpec = DenseVectorSpec | SparseVectorSpec


@dataclass(frozen=True, kw_only=True)
class Embedding:
    """One text's vectors, by name"""

    vectors: Mapping[str, Vector]

    def __post_init__(self) -> None:
        if not self.vectors:
            raise ValueError("an embedding must hold at least one vector")

        if "" in self.vectors:
            raise ValueError("vector names must not be empty")

        object.__setattr__(self, "vectors", MappingProxyType(dict(self.vectors)))

    def dense(self, name: str) -> DenseVector:
        vector = self._vector(name)
        if not isinstance(vector, DenseVector):
            raise TypeError(f"vector {name!r} is sparse, not dense")
        return vector

    def sparse(self, name: str) -> SparseVector:
        vector = self._vector(name)
        if not isinstance(vector, SparseVector):
            raise TypeError(f"vector {name!r} is dense, not sparse")
        return vector

    def _vector(self, name: str) -> Vector:
        try:
            return self.vectors[name]
        except KeyError:
            raise KeyError(
                f"no vector {name!r}; this embedding has: {', '.join(sorted(self.vectors))}"
            ) from None


class Embedder(Protocol):
    """Turns text into named vectors.
    Every Embedding returned holds exactly the vectors in vector_specs.
    Queries and passages may be encoded differently (prompts), so the callers must use the matching method.
    Equal identities mean equal vectors for the same text;
    the index refuses to mix embedders with different identities in one vector space.
    """

    @property
    def identity(self) -> str: ...

    @property
    def vector_specs(self) -> Mapping[str, VectorSpec]: ...

    @property
    def tokenizer(self) -> Tokenizer | None:
        """Counts what the embedder's models receive, against their limit;
        None when there is no limit;
        """
        ...

    def encode_query(self, text: str) -> Embedding: ...
    def encode_passages(self, texts: Sequence[str]) -> list[Embedding]: ...


def check_embedding(
    embedding: Embedding, vector_spec: Mapping[str, VectorSpec]
) -> None:
    """The embedding holds exactly the specified vectors, each of its kind,
    and if dense, its dimensions.
    """
    if set(embedding.vectors) != set(vector_spec):
        raise ValueError(
            f"the embedding holds {', '.join(sorted(embedding.vectors))}, "
            f"but the specs are for {', '.join(sorted(vector_spec))}"
        )
    for name, spec in vector_spec.items():
        vector = embedding.vectors[name]
        match spec:
            case DenseVectorSpec(dimension=dimension):
                if not isinstance(vector, DenseVector):
                    raise TypeError(f"vector {name!r} should be dense but is sparse")
                if len(vector.values) != dimension:
                    raise ValueError(
                        f"vector {name!r} has {len(vector.values)} values, but its spec says {dimension}"
                    )
            case SparseVectorSpec():
                if not isinstance(vector, SparseVector):
                    raise TypeError(f"vector {name!r} should be sparse but is dense")
