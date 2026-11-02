"""search_chunks: what it asks the index for, and how a reranker reorders the hits.

Most tests use an index that returns hits fixed in advance, so the index's own ranking
can't hide a mistake here; one test runs through a real InMemoryChunkIndex.
"""

from collections.abc import Sequence
from datetime import date

import pytest

from document_processor.retrieval.attributes import DateBetween
from document_processor.retrieval.in_memory_index import InMemoryChunkIndex
from document_processor.retrieval.index import (
    ChunkFilter,
    ChunkHit,
    GeneratedEntry,
    IndexedChunk,
    SearchQuery,
)
from document_processor.retrieval.search import RetrievedChunk, search_chunks
from tests.retrieval.stub_embedder import StubEmbedder


def chunk(chunk_id: str, *headings: str) -> IndexedChunk:
    text = f"text of {chunk_id}"
    return IndexedChunk(
        document_id="law",
        chunk_id=chunk_id,
        position=0,
        text=text,
        embedded_text=text,
        headings=headings,
    )


def hit(chunk_id: str, score: float) -> ChunkHit:
    return ChunkHit(chunk=chunk(chunk_id), score=score)


class FixedIndex(InMemoryChunkIndex):
    """Returns these hits, best first, cut to the query's limit; remembers every query."""

    def __init__(self, embedder: StubEmbedder, hits: Sequence[ChunkHit]) -> None:
        super().__init__(
            embedder_identity=embedder.identity, vector_specs=embedder.vector_specs
        )
        self.hits = list(hits)
        self.queries: list[SearchQuery] = []

    def _search(self, query: SearchQuery) -> list[ChunkHit]:
        self.queries.append(query)
        return self.hits[: query.limit]


class TableReranker:
    """Scores a passage by the first table key it contains; remembers every call."""

    identity = "table-reranker"

    def __init__(self, table: dict[str, float], default: float = 0.0) -> None:
        self.table = table
        self.default = default
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((question, list(passages)))
        return [
            next((s for key, s in self.table.items() if key in p), self.default)
            for p in passages
        ]


class WrongCountReranker:
    identity = "drops-one"

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        return [1.0] * (len(passages) - 1)


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder()


def ranked(results: list[RetrievedChunk]) -> list[tuple[str, float, float | None]]:
    return [(r.chunk.chunk_id, r.index_score, r.rerank_score) for r in results]


FIVE = [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7), hit("d", 0.6), hit("e", 0.5)]


# Without a reranker ----------------------------------------------------------------------------


def test_without_a_reranker_the_index_order_and_scores_are_kept(
    embedder: StubEmbedder,
) -> None:
    results = search_chunks(
        "giao đất", embedder=embedder, index=FixedIndex(embedder, FIVE), limit=3
    )
    assert ranked(results) == [("a", 0.9, None), ("b", 0.8, None), ("c", 0.7, None)]


def test_without_a_reranker_the_index_is_asked_for_the_limit_only(
    embedder: StubEmbedder,
) -> None:
    index = FixedIndex(embedder, FIVE)
    search_chunks("giao đất", embedder=embedder, index=index, limit=3, rerank_depth=50)
    assert [query.limit for query in index.queries] == [3]


def test_the_question_is_embedded_as_a_query(embedder: StubEmbedder) -> None:
    index = FixedIndex(embedder, FIVE)
    search_chunks("giao đất", embedder=embedder, index=index)
    assert embedder.queries == ["giao đất"]
    assert index.queries[0].embedding == embedder.encode_query("giao đất")


def test_search_options_reach_the_index(embedder: StubEmbedder) -> None:
    index = FixedIndex(embedder, FIVE)
    chunk_filter = ChunkFilter(
        attributes={"issued_on": DateBetween(at_least=date(2024, 1, 1))}
    )
    search_chunks(
        "giao đất",
        embedder=embedder,
        index=index,
        chunk_filter=chunk_filter,
        vector_names=("dense",),
        generators=("questions-v1",),
        fusion_depth=7,
    )
    (query,) = index.queries
    assert (
        query.chunk_filter,
        query.vector_names,
        query.generators,
        query.fusion_depth,
    ) == (
        chunk_filter,
        ("dense",),
        ("questions-v1",),
        7,
    )


def test_found_through_is_passed_on(embedder: StubEmbedder) -> None:
    found = GeneratedEntry(
        document_id="law", generator="g", entry_id="q", chunk_ids=("a",), text="Ai?"
    )
    index = FixedIndex(
        embedder, [ChunkHit(chunk=chunk("a"), score=0.9, found_through=(found,))]
    )
    (result,) = search_chunks("giao đất", embedder=embedder, index=index)
    assert result.found_through == (found,)


# With a reranker -------------------------------------------------------------------------------


def test_a_reranker_reorders_the_hits_and_keeps_both_scores(
    embedder: StubEmbedder,
) -> None:
    reranker = TableReranker({"of c": 3.0, "of a": 2.0, "of b": 1.0})
    results = search_chunks(
        "giao đất",
        embedder=embedder,
        index=FixedIndex(embedder, FIVE[:3]),
        reranker=reranker,
    )
    assert ranked(results) == [("c", 0.7, 3.0), ("a", 0.9, 2.0), ("b", 0.8, 1.0)]


def test_a_reranker_can_bring_up_a_hit_the_index_ranked_below_the_limit(
    embedder: StubEmbedder,
) -> None:
    reranker = TableReranker({"of e": 1.0})
    results = search_chunks(
        "giao đất",
        embedder=embedder,
        index=FixedIndex(embedder, FIVE),
        limit=2,
        reranker=reranker,
    )
    assert [r.chunk.chunk_id for r in results] == ["e", "a"]


def test_the_reranker_sees_rerank_depth_hits(embedder: StubEmbedder) -> None:
    index = FixedIndex(embedder, FIVE)
    reranker = TableReranker({})
    search_chunks(
        "giao đất",
        embedder=embedder,
        index=index,
        limit=2,
        reranker=reranker,
        rerank_depth=4,
    )
    assert [query.limit for query in index.queries] == [4]
    assert len(reranker.calls[0][1]) == 4


def test_a_rerank_depth_below_the_limit_widens_to_the_limit(
    embedder: StubEmbedder,
) -> None:
    index = FixedIndex(embedder, FIVE)
    results = search_chunks(
        "giao đất",
        embedder=embedder,
        index=index,
        limit=3,
        reranker=TableReranker({}),
        rerank_depth=1,
    )
    assert [query.limit for query in index.queries] == [3]
    assert len(results) == 3


def test_equal_rerank_scores_keep_the_index_order(embedder: StubEmbedder) -> None:
    reranker = TableReranker({"of d": 5.0}, default=1.0)
    results = search_chunks(
        "giao đất",
        embedder=embedder,
        index=FixedIndex(embedder, FIVE),
        reranker=reranker,
    )
    assert [r.chunk.chunk_id for r in results] == ["d", "a", "b", "c", "e"]


def test_the_reranker_reads_the_question_and_each_chunks_headings_then_text(
    embedder: StubEmbedder,
) -> None:
    hits = [
        ChunkHit(chunk=chunk("a", "Luật Đất đai", "Điều 118"), score=0.9),
        ChunkHit(chunk=chunk("b"), score=0.8),
    ]
    reranker = TableReranker({})
    search_chunks(
        "Ai được giao đất?",
        embedder=embedder,
        index=FixedIndex(embedder, hits),
        reranker=reranker,
    )
    assert reranker.calls == [
        ("Ai được giao đất?", ["Luật Đất đai\nĐiều 118\ntext of a", "text of b"])
    ]


def test_no_hits_means_no_reranking(embedder: StubEmbedder) -> None:
    reranker = TableReranker({})
    assert (
        search_chunks(
            "giao đất",
            embedder=embedder,
            index=FixedIndex(embedder, []),
            reranker=reranker,
        )
        == []
    )
    assert reranker.calls == []


def test_a_reranker_giving_the_wrong_number_of_scores_is_refused(
    embedder: StubEmbedder,
) -> None:
    with pytest.raises(ValueError, match="'drops-one' gave 4 scores for 5 passages"):
        search_chunks(
            "giao đất",
            embedder=embedder,
            index=FixedIndex(embedder, FIVE),
            reranker=WrongCountReranker(),
        )


# Refused before anything is embedded ----------------------------------------------------------


@pytest.mark.parametrize("question", ["", "  \n"])
def test_a_blank_question_is_refused(embedder: StubEmbedder, question: str) -> None:
    with pytest.raises(ValueError, match="no text"):
        search_chunks(question, embedder=embedder, index=FixedIndex(embedder, FIVE))
    assert embedder.queries == []


@pytest.mark.parametrize("depth", [0, -1])
def test_rerank_depth_must_be_positive(embedder: StubEmbedder, depth: int) -> None:
    with pytest.raises(ValueError, match="rerank_depth must be positive"):
        search_chunks(
            "giao đất",
            embedder=embedder,
            index=FixedIndex(embedder, FIVE),
            rerank_depth=depth,
        )
    assert embedder.queries == []


# Through a real index -------------------------------------------------------------------------


def test_search_runs_through_a_real_index(embedder: StubEmbedder) -> None:
    texts = {
        "land": "giao đất nông nghiệp",
        "tax": "thuế thu nhập cá nhân",
        "urban": "đất ở đô thị",
    }
    index = InMemoryChunkIndex(
        embedder_identity=embedder.identity, vector_specs=embedder.vector_specs
    )
    chunks = [
        IndexedChunk(
            document_id="law", chunk_id=cid, position=i, text=t, embedded_text=t
        )
        for i, (cid, t) in enumerate(texts.items())
    ]
    index.replace_document(
        "law", chunks, embedder.encode_passages(list(texts.values()))
    )

    plain = search_chunks("giao đất", embedder=embedder, index=index, limit=1)
    reranked = search_chunks(
        "giao đất",
        embedder=embedder,
        index=index,
        limit=1,
        reranker=TableReranker({"đô thị": 1.0}),
    )
    assert [r.chunk.chunk_id for r in plain] == ["land"]
    assert [r.chunk.chunk_id for r in reranked] == ["urban"]
