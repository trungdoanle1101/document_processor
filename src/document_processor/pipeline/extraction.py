from pathlib import Path

from document_processor.adapters import ADAPTERS, Adapter
from document_processor.core.models import PhysicalDocument
from document_processor.core.registry import Registry


def extract_physical_document(
    path: Path | str, adapters: Registry[Adapter] = ADAPTERS
) -> PhysicalDocument:
    path = Path(path)
    adapter = adapters.get(path.suffix.lower())
    return adapter.parse_file(path)
