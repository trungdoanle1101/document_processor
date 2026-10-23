from document_processor.logical import Profile, PROFILES
from document_processor.chunking import Chunker, OutlineChunker, WordTokenizer
from document_processor.pipeline import Pipeline, format_outline

if __name__ == "__main__":
    p = Pipeline(
        profile="vn_regulation",
        chunker=OutlineChunker(
            tokenizer=WordTokenizer(),
            max_tokens=100,
            include_breadcrumb_in_size=True,
            keep_sections_apart=True
        )
    )
    file_path = "data/2025_Quy dinh quan ly kien truc CNTT.docx"
    result = p.run(file_path)
    print(format_outline(result.logical, sections_only=False, indent=2))
    result.save("data/result")
