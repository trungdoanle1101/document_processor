from collections import Counter
from statistics import median

from document_processor.pipeline import OutlineChunker, Pipeline, WordTokenizer, format_outline

if __name__ == "__main__":

    pipeline = Pipeline(
        profile="vn_regulation",
        chunker=OutlineChunker(tokenizer=WordTokenizer(), max_tokens=300),
    )
    result = pipeline.run("data/2025_Quy dinh quan ly kien truc CNTT.docx")

    # 1. The big structure: does the profile read Chương, Mục, Điều and the boundaries right?
    print(format_outline(result.logical, sections_only=True))

    # 2. Chunk sizes and owners
    sizes = [len(c.text.split()) for c in result.chunks]
    print(len(result.chunks), "chunks; median", median(sizes), "words; max", max(sizes),
        "; over budget:", sum(s > 300 for s in sizes))
    print(Counter(c.path[-1].level if c.path else c.part for c in result.chunks))

    # 3. A few chunks as they would be embedded
    for chunk in result.chunks[:3] + result.chunks[-3:]:
        print("-----", chunk.id, chunk.part, (chunk.start, chunk.end))
        print(chunk.contextualize()[:500])

    for c in result.chunks:
        if len(c.text.split()) > 300:
            print(c.id, (c.start, c.end), c.headings, type(result.logical.segments[c.start]).__name__)
            print(c.text[:800])

    result.save("data/out/")
