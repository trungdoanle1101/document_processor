from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


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


@dataclass(frozen=True, kw_only=True)
class Embedding:
    dense: tuple[float, ...]
    sparse: SparseVector | None

    def __post_init__(self) -> None:
        if not self.dense:
            raise ValueError("dense vector must not be empty")


class Embedder(Protocol):
    """When has_sparse is true, every returned Embedding has 'sparse' attribute set, and when it's false, none do.
    Queries and documents may be encoded differently, so callers must use matching method.
    """

    @property
    def model_identity(self) -> str: ...

    @property
    def dense_dimension(self) -> int: ...

    @property
    def has_sparse(self) -> bool: ...

    def encode_documents(self, texts: Sequence[str]) -> list[Embedding]: ...
    def encode_query(self, text: str) -> Embedding: ...
