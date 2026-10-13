from pathlib import Path

from document_processor.adapters.docx import DocxAdapter


def dump_blocks(path: str | Path, save_as: str | Path | None = None) -> None:
    adapter = DocxAdapter()
    physical_document = adapter.parse_file(path)
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
    filename = "1_Khung_tham_chieu_KTCNTT_ve_phan_mem_Mau_tai_lieu_kien_truc.docx"
    filepath = f"data/{filename}"
    dump_blocks(filepath, save_as=f"data/{filename}_blocks.txt")