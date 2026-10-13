from pathlib import Path

from document_processor.adapters.docx import DocxAdapter
from document_processor.core.cleaning import clean_document


def dump_blocks(path: str | Path, save_as: str | Path | None = None, clean: bool=False) -> None:
    adapter = DocxAdapter()
    physical_document = adapter.parse_file(path)

    if clean:
        physical_document = clean_document(physical_document)

    outputs: list[str] = []
    for block in physical_document.blocks:
        block_output = f"id: {block.id} | block_type: {block.block_type.value} | alignment: {block.paragraph_format.alignment.value if block.paragraph_format.alignment is not None else None} | displayed_marker: {block.list_info.displayed_marker if block.list_info else None} | text: {block.text[:60]!r}\n"
        outputs.append(block_output)

    if save_as is None:
        return
    
    save_path = Path(save_as)

    with save_path.open("w", encoding="utf-8") as f:
        f.writelines(outputs)

if __name__ == "__main__":
    # filename = "2025_Quy dinh quan ly kien truc CNTT.docx"
    filename = "Phu_luc_Quy dinh quan ly kien truc CNTT.docx"
    filepath = f"data/{filename}"
    dump_blocks(filepath, save_as=f"data/{filename}_blocks.txt")
    dump_blocks(filepath, save_as=f"data/{filename}_blocks_cleaned.txt", clean=True)