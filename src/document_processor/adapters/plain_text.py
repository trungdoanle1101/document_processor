import codecs
from pathlib import Path

from document_processor.core.models.physical import (
    PhysicalDocument, PhysicalBlock, BlockType, Region,
    DocSourceFormat, ParagraphFormat
)

from document_processor.core.identity import compute_hash
from document_processor.core.text_utils import is_invisible



class PlainTextAdapter:
    name = "PlainTextAdapter"
    version = "1.0.0"


    def read(self, path: str | Path) -> PhysicalDocument:
        path = Path(path)
        
        # First, check if the format is valid. If not valid, raise and exit
        if not path.is_file():
            raise FileNotFoundError(f"{path} is not a valid file")
        if path.suffix.lower() != ".txt":
            raise ValueError(f"Unexpected file type: Wanted .txt, got {path.suffix}")
        
    
        filename = path.name    
        file_bytes = path.read_bytes()
        file_hash = compute_hash(file_bytes)

        # Remove BOM
        file_bytes = file_bytes.removeprefix(codecs.BOM_UTF8)

        lines = file_bytes.splitlines()
        blocks = []
        block_counter = 0
        for i, line_bytes in enumerate(lines, start=1):

            try:
                decoded_line = line_bytes.decode(encoding="utf-8")
            except UnicodeDecodeError as e:
                raise ValueError(f"File {filename}: Failed to decode line {i} as utf-8") from e


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
                changelog=()
            )
            blocks.append(block)
            


        phys_doc = PhysicalDocument(
            id=file_hash,
            source_filename=filename,
            source_format=DocSourceFormat.TXT,
            adapter_name=self.name,
            adapter_version=self.version,
            file_hash=file_hash,
            blocks=tuple(blocks)
        )

        return phys_doc