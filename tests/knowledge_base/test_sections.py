"""complete_sections: when hits share a section, the rest of it is added, in document order.

The document mirrors a regulation's article: Điều 7, khoản 2 lists layers a), b), c),
the first split over two chunks; Điều 8 and Điều 3 sit elsewhere.
"""

from collections.abc import Sequence

import pytest

from document_processor.chunking import PathStep
from document_processor.retrieval.attributes import TextIn
from document_processor.retrieval.in_memory_index import InMemoryChunkIndex
from document_processor.retrieval.index import ChunkFilter, IndexedChunk
from document_processor.retrieval.reference import address_keys
from document_processor.retrieval.search import RetrievedChunk
from document_processor.retrieval.sections import complete_sections
from tests.retrieval.stub_embedder import StubEmbedder

DIEU_7 = (("chuong", "III"), ("muc", "II"), ("dieu", "7"))
KHOAN_2 = (*DIEU_7, ("khoan", "2"))
SECTION = "chuong:iii/muc:ii/dieu:7/khoan:2"

# chunk id, address (level, number; None ends the address), group
LAYOUT: list[tuple[str, tuple[tuple[str, str | None], ...], str]] = [
    ("c5", (("chuong", "II"), ("dieu", "3")), "it"),
    ("c18", (*DIEU_7, ("khoan", "1")), "it"),
    ("c19", KHOAN_2, "it"),
    ("c20", (*KHOAN_2, ("diem", "a")), "it"),
    ("c21", (*KHOAN_2, ("diem", "a"), ("gach", None)), "it"),
    ("c22", (*KHOAN_2, ("diem", "b")), "it"),
    ("c23", (*KHOAN_2, ("diem", "b"), ("gach", None)), "it"),
    ("c24", (*KHOAN_2, ("diem", "c")), "it"),
    ("c25", (("chuong", "III"), ("muc", "II"), ("dieu", "8")), "it"),
    ("cover", (), "it"),
]


def path_of(address: Sequence[tuple[str, str | None]]) -> list[PathStep]:
    return [
        PathStep(
            node_id=f"n{i}",
            level=level,
            number=number,
            marker=None,
            title=None,
            start=i,
        )
        for i, (level, number) in enumerate(address)
    ]


def make_chunk(
    position: int, chunk_id: str, address: Sequence[tuple[str, str | None]], group: str
) -> IndexedChunk:
    text = f"text of {chunk_id} " * 5  # 5 * (len(chunk_id) + 9) characters
    return IndexedChunk(
        document_id="qd",
        chunk_id=chunk_id,
        position=position,
        text=text,
        embedded_text=text,
        reference_keys=address_keys(path_of(address)),
        attributes={"access_group": group},
    )


CHUNKS = {
    chunk_id: make_chunk(i, chunk_id, address, group)
    for i, (chunk_id, address, group) in enumerate(LAYOUT)
}


@pytest.fixture
def index() -> InMemoryChunkIndex:
    embedder = StubEmbedder()
    index = InMemoryChunkIndex(
        embedder_identity=embedder.identity, vector_specs=embedder.vector_specs
    )
    chunks = list(CHUNKS.values())
    index.replace_document(
        "qd", chunks, embedder.encode_passages([c.text for c in chunks])
    )
    return index


def hit(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=CHUNKS[chunk_id], index_score=score, rerank_score=score)


class FixedSearch:
    """Returns these hits whatever the question, cut to the limit."""

    def __init__(self, *hits: RetrievedChunk) -> None:
        self.hits = list(hits)

    def __call__(self, question: str, *, limit: int) -> list[RetrievedChunk]:
        return self.hits[:limit]


# The hits from the real question: c5 first, then b), b), Điều 6 (not in this document), c)
HITS = FixedSearch(
    hit("c5", 0.81), hit("c22", 0.80), hit("c23", 0.78), hit("c24", 0.76)
)


def ids(passages: list[RetrievedChunk]) -> list[str]:
    return [p.chunk.chunk_id for p in passages]


def test_a_shared_section_is_completed_in_document_order_ahead_of_other_hits(
    index: InMemoryChunkIndex,
) -> None:
    search = complete_sections(HITS, index=index, max_chars=10_000)
    assert ids(search("lớp tích hợp?", limit=5)) == [
        "c19",
        "c20",
        "c21",
        "c22",
        "c23",
        "c24",
        "c5",
    ]


def test_added_chunks_are_marked_and_borrow_the_best_hits_scores(
    index: InMemoryChunkIndex,
) -> None:
    passages = complete_sections(HITS, index=index, max_chars=10_000)("q", limit=5)
    by_id = {p.chunk.chunk_id: p for p in passages}
    for added in ("c19", "c20", "c21"):
        assert by_id[added].completed_section == SECTION
        assert (by_id[added].index_score, by_id[added].rerank_score) == (0.80, 0.80)
    for found in ("c22", "c23", "c24", "c5"):
        assert by_id[found].completed_section is None
    assert by_id["c24"].rerank_score == 0.76


def test_a_lone_hit_per_section_changes_nothing(index: InMemoryChunkIndex) -> None:
    search = FixedSearch(hit("c22", 0.9), hit("c5", 0.8), hit("c25", 0.7))
    assert (
        complete_sections(search, index=index, max_chars=10_000)("q", limit=5)
        == search.hits
    )


def test_a_section_over_the_budget_is_left_as_it_was(index: InMemoryChunkIndex) -> None:
    added = sum(len(CHUNKS[c].text) for c in ("c19", "c20", "c21"))
    exact = complete_sections(HITS, index=index, max_chars=added)("q", limit=5)
    short = complete_sections(HITS, index=index, max_chars=added - 1)("q", limit=5)
    assert ids(exact)[:3] == ["c19", "c20", "c21"]
    assert short == HITS.hits


def test_a_budget_of_zero_turns_it_off(index: InMemoryChunkIndex) -> None:
    assert complete_sections(HITS, index=index, max_chars=0)("q", limit=5) == HITS.hits


def test_two_hits_inside_one_split_item_complete_its_parent(
    index: InMemoryChunkIndex,
) -> None:
    # c21 is a "-" bullet without a number, so its address stops at diem:a, like c20's
    search = FixedSearch(hit("c21", 0.9), hit("c20", 0.8))
    passages = complete_sections(search, index=index, max_chars=10_000)("q", limit=5)
    assert ids(passages) == ["c19", "c20", "c21", "c22", "c23", "c24"]


def test_nested_sections_add_each_chunk_once(index: InMemoryChunkIndex) -> None:
    # c22 and c24 share khoản 2; c18 (khoản 1) and c19 (khoản 2 itself) share Điều 7,
    # which holds khoản 2: Điều 7 adds only what khoản 2 did not
    search = FixedSearch(
        hit("c22", 0.9), hit("c24", 0.85), hit("c18", 0.8), hit("c19", 0.7)
    )
    passages = complete_sections(search, index=index, max_chars=10_000)("q", limit=5)
    assert ids(passages) == ["c19", "c20", "c21", "c22", "c23", "c24", "c18"]


def test_the_budget_counts_a_shared_chunk_once(index: InMemoryChunkIndex) -> None:
    # khoản 2 adds c20, c21, c23; Điều 7 then adds nothing new, so it costs nothing
    added = sum(len(CHUNKS[c].text) for c in ("c20", "c21", "c23"))
    search = FixedSearch(
        hit("c22", 0.9), hit("c24", 0.85), hit("c18", 0.8), hit("c19", 0.7)
    )
    passages = complete_sections(search, index=index, max_chars=added)("q", limit=5)
    assert ids(passages) == ["c19", "c20", "c21", "c22", "c23", "c24", "c18"]


def test_chunks_the_filter_excludes_are_never_added(index: InMemoryChunkIndex) -> None:
    embedder = StubEmbedder()
    mixed = InMemoryChunkIndex(
        embedder_identity=embedder.identity, vector_specs=embedder.vector_specs
    )
    chunks = [
        CHUNKS["c19"],
        IndexedChunk(
            **{**vars(CHUNKS["c20"]), "attributes": {"access_group": "secret"}}
        ),
        CHUNKS["c22"],
        CHUNKS["c24"],
    ]
    mixed.replace_document(
        "qd", chunks, embedder.encode_passages([c.text for c in chunks])
    )
    readable = ChunkFilter(attributes={"access_group": TextIn(values=("it",))})
    search = FixedSearch(hit("c22", 0.9), hit("c24", 0.8))

    passages = complete_sections(
        search, index=mixed, chunk_filter=readable, max_chars=10_000
    )("q", limit=5)
    assert ids(passages) == ["c19", "c22", "c24"]


def test_a_document_outside_the_filter_is_not_completed(
    index: InMemoryChunkIndex,
) -> None:
    other = ChunkFilter(document_ids=("another",))
    assert (
        complete_sections(HITS, index=index, chunk_filter=other, max_chars=10_000)(
            "q", limit=5
        )
        == HITS.hits
    )


def test_chunks_without_an_address_are_never_grouped(index: InMemoryChunkIndex) -> None:
    search = FixedSearch(hit("cover", 0.9), hit("cover", 0.8))
    assert (
        complete_sections(search, index=index, max_chars=10_000)("q", limit=5)
        == search.hits
    )


def test_a_short_key_that_matches_is_not_mistaken_for_the_section() -> None:
    # a chunk under another chương whose address *ends* like the section's
    embedder = StubEmbedder()
    index = InMemoryChunkIndex(
        embedder_identity=embedder.identity, vector_specs=embedder.vector_specs
    )
    look_alike = make_chunk(
        50, "elsewhere", (("phan", "1"), *KHOAN_2, ("diem", "d")), "it"
    )
    chunks = [CHUNKS["c22"], CHUNKS["c24"], look_alike]
    index.replace_document(
        "qd", chunks, embedder.encode_passages([c.text for c in chunks])
    )
    search = FixedSearch(hit("c22", 0.9), hit("c24", 0.8))
    assert ids(
        complete_sections(search, index=index, max_chars=10_000)("q", limit=5)
    ) == ["c22", "c24"]


def test_bad_settings_are_refused(index: InMemoryChunkIndex) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        complete_sections(HITS, index=index, max_chars=-1)
    with pytest.raises(ValueError, match="min_hits"):
        complete_sections(HITS, index=index, max_chars=100, min_hits=1)