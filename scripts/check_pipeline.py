from document_processor.pipeline import Pipeline, format_outline
from document_processor.chunking import OutlineChunker, WordTokenizer

if __name__ == "__main__":

    pipeline = Pipeline(
        profile="vn_regulation",
        chunker=OutlineChunker(tokenizer=WordTokenizer(), max_tokens=300, include_breadcrumb_in_size=True, table_layout="grid"),
    )
    result = pipeline.run("data/2025_Quy dinh quan ly kien truc CNTT.docx")

    print(result.summarize().format())   # chunk_over_budget should be gone, or point to one row too big on its own
    for c in result.chunks:
        if c.table_rows is not None:
            print(c.id, c.table_rows, c.token_count, c.headings)
            print(c.text[:300], "\n")


    # result.save("data/out/")
