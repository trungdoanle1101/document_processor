from collections.abc import Sequence
from dataclasses import dataclass

from document_processor.retrieval.embedding import Embedder
from document_processor.retrieval.index import (
    ChunkFilter,
    ChunkHit,
    ChunkIndex,
    GeneratedEntry,
    GeneratorChoice,
    IndexedChunk,
    SearchQuery,
)
from document_processor.retrieval.reranker import Reranker

_NO_FILTER = ChunkFilter()


@dataclass(frozen=True, kw_only=True)
class RetrievedChunk:
    chunk: IndexedChunk
    index_score: float
    rerank_score: float | None
    found_through: tuple[GeneratedEntry, ...] = ()


def search_chunks(
    question: str,
    *,
    embedder: Embedder,
    index: ChunkIndex,
    chunk_filter: ChunkFilter = _NO_FILTER,
    vector_names: tuple[str, ...] | None = None,
    generators: GeneratorChoice = "all",
    limit: int = 10,
    fusion_depth: int = 50,
    reranker: Reranker | None = None,
    rerank_depth: int = 50,
) -> list[RetrievedChunk]:
    """Search for chunks that best answer the question, best first."""
    if not question.strip():
        raise ValueError("question has no text")

    if rerank_depth <= 0:
        raise ValueError(f"rerank_depth must be positive, got {rerank_depth}")

    query_embedding = embedder.encode_query(question)
    first_pass_limit = limit if reranker is None else max(limit, rerank_depth)
    search_query = SearchQuery(
        embedding=query_embedding,
        chunk_filter=chunk_filter,
        vector_names=vector_names,
        generators=generators,
        limit=first_pass_limit,
        fusion_depth=fusion_depth,
    )
    chunk_hits = index.search(search_query)
    rerank_scored: Sequence[tuple[ChunkHit, float | None]] = [
        (hit, None) for hit in chunk_hits
    ]
    if reranker is not None and chunk_hits:
        scores = reranker.score(
            question, [_rerank_passage(hit.chunk) for hit in chunk_hits]
        )
        if len(scores) != len(chunk_hits):
            raise ValueError(
                f"reranker {reranker.identity!r} gave {len(scores)} scores for {len(chunk_hits)} passages"
            )
        reranked = sorted(
            zip(chunk_hits, scores), key=lambda pair: pair[1], reverse=True
        )
        rerank_scored = reranked
    return [
        RetrievedChunk(
            chunk=hit.chunk,
            index_score=hit.score,
            rerank_score=rerank_score,
            found_through=hit.found_through,
        )
        for hit, rerank_score in rerank_scored[:limit]
    ]


def _rerank_passage(chunk: IndexedChunk) -> str:
    """The chunk as the reranker reads it: where it sits, then its words."""
    if not chunk.headings:
        return chunk.text
    return "\n".join((*chunk.headings, chunk.text))
