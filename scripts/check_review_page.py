from document_processor.chunking import OutlineChunker, WordTokenizer
from document_processor.pipeline import Pipeline

chunker = OutlineChunker(tokenizer=WordTokenizer(), max_tokens=300)
result = Pipeline(profile="vn_regulation", chunker=chunker).run("data/2025_Quy dinh quan ly kien truc CNTT.docx")
result.write_review_page("out/quy_dinh.review.html")