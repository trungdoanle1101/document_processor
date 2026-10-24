from typing import Protocol

from document_processor.chunking.chunk import Chunk
from document_processor.core.models import LogicalDocument


class Chunker(Protocol):
    @property
    def max_tokens(self) -> int: ...

    def chunk(self, document: LogicalDocument) -> tuple[Chunk, ...]: ...
