import json
from pathlib import Path
from dataclasses import asdict

from document_processor.adapters.docx import DocxAdapter
from document_processor.core.models import PhysicalDocument, LogicalDocument
from document_processor.core.cleaning import clean_document
from document_processor.core.serialization import save_physical_document, save_logical_document
from document_processor.logical.build import build_logical_document
from document_processor.logical.profiles.vn_regulation import VN_REGULATION


def dump_blocks(path: str | Path, save_as: str | Path | None = None, clean: bool=False) -> PhysicalDocument:
    adapter = DocxAdapter()
    physical_document = adapter.parse_file(path)

    if clean:
        physical_document = clean_document(physical_document)

    outputs: list[str] = []
    for block in physical_document.blocks:
        d = {
            "id": block.id,
            "block_type": block.block_type.value,
            "region": block.region.value,
            "list_info": asdict(block.list_info) if block.list_info is not None else None,
            "text": block.text[:80]
        }
        # block_output = f"id: {block.id} | block_type: {block.block_type.value} | region: {block.region.value} | alignment: {block.paragraph_format.alignment.value if block.paragraph_format.alignment is not None else None} | displayed_marker: {block.list_info.displayed_marker if block.list_info else None} | text: {block.text[:60]!r}\n"
        outputs.append(d)

    if save_as is None:
        return
    
    save_path = Path(save_as)

    with save_path.open("w", encoding="utf-8") as f:
        # f.writelines(outputs)
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return physical_document


def dump_logical(path: str | Path, save_as: str | Path) -> LogicalDocument:
    adapter = DocxAdapter()
    physical_document = adapter.parse_file(path)
    physical_document = clean_document(physical_document)
    logical_document = build_logical_document(VN_REGULATION, physical_document)

    save_path = Path(save_as)
    save_logical_document(logical_document, save_path)
    return logical_document


if __name__ == "__main__":
    filename = "2025_Quy dinh quan ly kien truc CNTT.docx"
    # filename = "Phu_luc_Quy dinh quan ly kien truc CNTT.docx"
    filepath = f"data/{filename}"
    doc = dump_logical(filepath, save_as=f"{filepath}_logical.json")