from typing import Protocol

from document_processor.chunking.chunk import Chunk
from document_processor.core.models import LogicalDocument


class Chunker(Protocol):
    def chunk(self, document: LogicalDocument) -> tuple[Chunk, ...]: ...
