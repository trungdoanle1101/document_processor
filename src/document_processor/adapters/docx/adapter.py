from pathlib import Path
from docx import Document as DocumentFactory
from io import BytesIO
from document_processor.core.models.physical import (
    PhysicalDocument,
    DocSourceFormat,
)
from document_processor.core.identity import compute_hash

from ._reader import _DocxReader

class DocxAdapter:
    name = "docx"
    version = "1.0.0"

    def parse_bytes(self, data: bytes, filename: str) -> PhysicalDocument:

        hash_str = compute_hash(data)

        try:
            doc = DocumentFactory(BytesIO(data))
        except Exception as e:
            raise ValueError(
                f"File {filename!r}: unable to decode bytes data as docx.Document"
            ) from e

        # Create a reader once per parse
        # Call read_blocks once
        reader = _DocxReader(doc, filename)
        blocks = reader.read_blocks()

        return PhysicalDocument(
            id=hash_str,
            source_filename=filename,
            source_format=DocSourceFormat.DOCX,
            adapter_name=self.name,
            adapter_version=self.version,
            file_hash=hash_str,
            blocks=blocks,
        )

    def parse_file(self, file_path: Path | str) -> PhysicalDocument:
        file_path = Path(file_path)

        # First, check if the format is valid. If not valid, raise and exit
        if not file_path.is_file():
            raise FileNotFoundError(f"{file_path} is not a valid file")
        if file_path.suffix.lower() != ".docx":
            raise ValueError(
                f"Unexpected file type: Wanted .docx, got {file_path.suffix}"
            )

        filename = file_path.name
        file_bytes = file_path.read_bytes()

        return self.parse_bytes(file_bytes, filename)
