from document_processor.chunking import OutlineChunker, WordTokenizer
from document_processor.core.models import TableSegment
from document_processor.pipeline import Pipeline

for layout in ("grid", "key_value"):
    chunker = OutlineChunker(tokenizer=WordTokenizer(), max_tokens=300, table_layout=layout)
    result = Pipeline(profile="vn_regulation", chunker=chunker).run("data/2025_Quy dinh quan ly kien truc CNTT.docx")
    print(f"===== {layout}")
    print(result.summarize().format())

    # Every table: size, header rows, and how many pieces it was split into
    for index, segment in enumerate(result.logical.segments):
        if isinstance(segment, TableSegment):
            table = segment.table
            pieces = [c for c in result.chunks if c.table_rows is not None and c.start == index]
            print(f"table @ {index} ({segment.block_id}): {table.n_rows} rows, "
                  f"header {table.n_header_rows}, {len(pieces)} pieces")

    # The pieces of split tables, first 5 lines each
    for chunk in result.chunks:
        if chunk.table_rows is not None:
            print(f"--- {chunk.id} rows {chunk.table_rows}, {chunk.token_count} tokens")
            print("\n".join(chunk.text.splitlines()[:5]))