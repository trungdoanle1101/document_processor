import codecs
from pathlib import Path

from document_processor.core.identity import compute_hash
from document_processor.core.models.physical import (
    BlockType,
    DocSourceFormat,
    ParagraphFormat,
    PhysicalBlock,
    PhysicalDocument,
    Region,
)
from document_processor.core.text_utils import is_invisible


class PlainTextAdapter:
    name = "plain_text"
    version = "1.0.0"

    def parse_bytes(self, data: bytes, filename: str) -> PhysicalDocument:
        """Parse UTF-8 text into a PhysicalDocument, one block per non-blank line.

        A leading BOM is removed; blank and invisible-only lines are skipped.
        `filename` is a caller-supplied label used for display and error
        messages; the document's identity comes from the hash of `data`.

        Raises:
            ValueError: if a line is not valid UTF-8.
        """

        hash_str = compute_hash(data)

        # Remove BOM
        data = data.removeprefix(codecs.BOM_UTF8)

        lines = data.splitlines()
        blocks = []
        block_counter = 0
        for i, line_bytes in enumerate(lines, start=1):
            try:
                decoded_line = line_bytes.decode(encoding="utf-8")
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"File {filename}: Failed to decode line {i} as utf-8"
                ) from e

            if is_invisible(decoded_line):
                continue

            block_counter += 1
            block = PhysicalBlock(
                id=f"pb{block_counter}",
                block_type=BlockType.PARAGRAPH,
                text=decoded_line,
                region=Region.BODY,
                page=None,
                paragraph_format=ParagraphFormat(alignment=None, style_name=None),
                char_spans=(),
                list_info=None,
                source_reference=f"line:{i}",
                changelog=(),
            )
            blocks.append(block)

        phys_doc = PhysicalDocument(
            id=hash_str,
            source_filename=filename,
            source_format=DocSourceFormat.TXT,
            adapter_name=self.name,
            adapter_version=self.version,
            file_hash=hash_str,
            blocks=tuple(blocks),
        )

        return phys_doc

    def parse_file(self, file_path: str | Path) -> PhysicalDocument:
        """Read a .txt file and parse it with `parse_bytes`.

        Raises:
            FileNotFoundError: if `file_path` is not an existing file.
            ValueError: if the suffix is not .txt, or the content is not valid UTF-8.
        """
        file_path = Path(file_path)

        # First, check if the format is valid. If not valid, raise and exit
        if not file_path.is_file():
            raise FileNotFoundError(f"{file_path} is not a valid file")
        if file_path.suffix.lower() != ".txt":
            raise ValueError(
                f"Unexpected file type: Wanted .txt, got {file_path.suffix}"
            )

        filename = file_path.name
        file_bytes = file_path.read_bytes()
        return self.parse_bytes(data=file_bytes, filename=filename)
