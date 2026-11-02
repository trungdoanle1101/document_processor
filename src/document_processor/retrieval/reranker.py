from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    """Scores how well each passage"""

    @property
    def identity(self) -> str: ...

    def score(self, question: str, passages: Sequence[str]) -> list[float]: ...
