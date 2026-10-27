from document_processor.chunking import OutlineChunker, WordTokenizer
from document_processor.pipeline import Pipeline

chunker = OutlineChunker(tokenizer=WordTokenizer(), max_tokens=512)
result = Pipeline(profile="vn_regulation", chunker=chunker).run("data/2025_Quy dinh quan ly kien truc CNTT.docx")
result.write_review_page("out/quy_dinh.review.html")

from statistics import quantiles
from document_processor.core.models import TextSegment

doc = result.logical
lengths = []
for node in doc.nodes:
    index = node.start + 1
    if (
        node.title is None
        and doc.get_children(node.id)          # it can be an ancestor in a breadcrumb
        and index < node.end
        and isinstance(segment := doc.segments[index], TextSegment)
    ):
        lengths.append(len(segment.text.split()))

lengths.sort()
p50, p90, p95 = (quantiles(lengths, n=100)[i - 1] for i in (50, 90, 95))
print(f"{len(lengths)} lead-ins · median {p50:.0f} · p90 {p90:.0f} · p95 {p95:.0f} · max {lengths[-1]}")
for cap in (25, 40, 50, 60):
    print(f"cut at {cap}: {sum(n > cap for n in lengths)}")