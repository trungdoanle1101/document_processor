from pathlib import Path

from document_processor.adapters import ADAPTERS
from document_processor.core.models import PhysicalDocument


def extract_physical_document(path: Path | str) -> PhysicalDocument:
    path = Path(path)
    adapter = ADAPTERS.get(path.suffix.lower())
    return adapter.parse_file(path)
