from collections import Counter
from statistics import median

from document_processor.pipeline import OutlineChunker, Pipeline, WordTokenizer, format_outline

if __name__ == "__main__":

    pipeline = Pipeline(
        profile="vn_regulation",
        chunker=OutlineChunker(tokenizer=WordTokenizer(), max_tokens=300),
    )
    result = pipeline.run("data/2025_Quy dinh quan ly kien truc CNTT.docx")

    summary = result.summarize().format()

    print(summary)


    result.save("data/out/")
