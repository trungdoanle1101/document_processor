from pathlib import Path
from typing import Protocol

from document_processor.core.models import PhysicalDocument


class Adapter(Protocol):
    name: str
    version: str

    def parse_file(self, file_path: Path) -> PhysicalDocument: ...
