"""to_indexed_chunks and index_document: chunker output into a ChunkIndex."""

from datetime import date
from typing import Any

import pytest

from document_processor.chunking import Chunk, PathStep
from document_processor.retrieval.attributes import TextIn
from document_processor.retrieval.in_memory_index import InMemoryChunkIndex
from document_processor.retrieval.index import ChunkFilter, IndexedChunk
from document_processor.retrieval.ingest import index_document, to_indexed_chunks
from document_processor.retrieval.reference import address_keys
from tests.retrieval.stub_embedder import StubEmbedder

CHUONG = PathStep(
    node_id="n0",
    level="chuong",
    number="I",
    marker="Chương I",
    title="QUY ĐỊNH CHUNG",
    start=1,
)
DIEU = PathStep(
    node_id="n1",
    level="dieu",
    number="2",
    marker="Điều 2.",
    title="Giải thích từ ngữ",
    start=3,
)
KHOAN = PathStep(
    node_id="n2", level="khoan", number="1", marker="1.", title=None, start=4
)


def make_chunk(**overrides: Any) -> Chunk:
    fields: dict[str, Any] = {
        "id": "c0",
        "logical_document_id": "ld1",
        "part": "body",
        "path": (CHUONG, DIEU),
        "start": 3,
        "end": 5,
        "text": "Điều 2. Giải thích từ ngữ\n1. “Dữ liệu” là ...",
        "token_count": 12,
        "block_ids": ("pb3", "pb4"),
    }
    fields.update(overrides)
    return Chunk(**fields)


FRONT = make_chunk(
    id="c0",
    part="front_matter",
    path=(),
    start=0,
    end=1,
    text="BỘ TÀI CHÍNH\nQuyết định số 12/QĐ-BTC",
)
ARTICLE = make_chunk(id="c1")
CLAUSE = make_chunk(
    id="c2", path=(CHUONG, DIEU, KHOAN), start=4, end=5, text="1. “Dữ liệu” là ..."
)
END = make_chunk(
    id="c3", part="end_matter", path=(), start=6, end=7, text="Nơi nhận:\n- Như Điều 3;"
)
CHUNKS = (FRONT, ARTICLE, CLAUSE, END)


# to_indexed_chunks -------------------------------------------------------------------------


def test_each_chunk_becomes_an_indexed_chunk() -> None:
    clause = to_indexed_chunks(
        CHUNKS, document_id="qd-12", attributes={"issued_on": date(2025, 3, 1)}
    )[2]
    assert clause == IndexedChunk(
        document_id="qd-12",
        chunk_id="c2",
        position=2,
        text=CLAUSE.text,
        embedded_text=CLAUSE.contextualize(),
        headings=CLAUSE.headings,
        reference_keys=address_keys(CLAUSE.path),
        attributes={"issued_on": date(2025, 3, 1), "part": "body"},
    )


def test_what_is_embedded_carries_the_headings_not_already_in_the_text() -> None:
    article, clause = to_indexed_chunks((ARTICLE, CLAUSE), document_id="qd-12")
    # the article's own heading is the first line of its text, so only the chapter is added
    assert article.embedded_text == f"{CHUONG.label}\n{ARTICLE.text}"
    assert clause.embedded_text == f"{CHUONG.label}\n{DIEU.label}\n{CLAUSE.text}"
    assert article.text == ARTICLE.text


def test_positions_follow_the_chunk_order() -> None:
    indexed = to_indexed_chunks(CHUNKS, document_id="qd-12")
    assert [(c.chunk_id, c.position) for c in indexed] == [
        ("c0", 0),
        ("c1", 1),
        ("c2", 2),
        ("c3", 3),
    ]


def test_every_chunk_gets_its_part() -> None:
    indexed = to_indexed_chunks(
        CHUNKS, document_id="qd-12", attributes={"kind": "quyet_dinh"}
    )
    assert [dict(c.attributes) for c in indexed] == [
        {"kind": "quyet_dinh", "part": "front_matter"},
        {"kind": "quyet_dinh", "part": "body"},
        {"kind": "quyet_dinh", "part": "body"},
        {"kind": "quyet_dinh", "part": "end_matter"},
    ]


def test_reference_keys_come_from_the_path() -> None:
    front, article, clause, end = to_indexed_chunks(CHUNKS, document_id="qd-12")
    assert front.reference_keys == end.reference_keys == ()
    assert "dieu:2" in article.reference_keys
    assert "dieu:2/khoan:1" not in article.reference_keys
    assert "dieu:2/khoan:1" in clause.reference_keys


def test_a_caller_part_is_refused() -> None:
    with pytest.raises(ValueError, match='"part" is reserved'):
        to_indexed_chunks(CHUNKS, document_id="qd-12", attributes={"part": "body"})


def test_no_chunks_give_no_indexed_chunks() -> None:
    assert to_indexed_chunks((), document_id="qd-12") == []


# index_document ----------------------------------------------------------------------------


def make_index(embedder: StubEmbedder) -> InMemoryChunkIndex:
    return InMemoryChunkIndex(
        embedder_identity=embedder.identity, vector_specs=embedder.vector_specs
    )


def all_chunks(
    index: InMemoryChunkIndex, chunk_filter: ChunkFilter | None = None
) -> list[IndexedChunk]:
    return index.list_chunks(chunk_filter or ChunkFilter(), limit=100)


def test_the_document_is_embedded_as_contextualized_and_stored() -> None:
    embedder = StubEmbedder()
    index = make_index(embedder)
    index_document(CHUNKS, document_id="qd-12", embedder=embedder, index=index)
    assert embedder.passage_batches == [[c.contextualize() for c in CHUNKS]]
    assert all_chunks(index) == to_indexed_chunks(CHUNKS, document_id="qd-12")


def test_indexing_again_replaces_the_document() -> None:
    embedder = StubEmbedder()
    index = make_index(embedder)
    index_document(CHUNKS, document_id="qd-12", embedder=embedder, index=index)
    index_document((ARTICLE,), document_id="qd-12", embedder=embedder, index=index)
    assert [c.chunk_id for c in all_chunks(index)] == ["c1"]


def test_indexing_no_chunks_removes_the_document() -> None:
    embedder = StubEmbedder()
    index = make_index(embedder)
    index_document(CHUNKS, document_id="qd-12", embedder=embedder, index=index)
    index_document((), document_id="qd-12", embedder=embedder, index=index)
    assert all_chunks(index) == []


def test_other_documents_are_left_alone() -> None:
    embedder = StubEmbedder()
    index = make_index(embedder)
    index_document(CHUNKS, document_id="qd-12", embedder=embedder, index=index)
    index_document((ARTICLE,), document_id="nd-43", embedder=embedder, index=index)
    assert sorted({c.document_id for c in all_chunks(index)}) == ["nd-43", "qd-12"]


def test_a_caller_part_is_refused_before_anything_is_embedded() -> None:
    embedder = StubEmbedder()
    with pytest.raises(ValueError, match='"part" is reserved'):
        index_document(
            CHUNKS,
            document_id="qd-12",
            embedder=embedder,
            index=make_index(embedder),
            attributes={"part": "x"},
        )
    assert embedder.passage_batches == []


def test_indexed_chunks_can_be_found_by_citation_and_part() -> None:
    embedder = StubEmbedder()
    index = make_index(embedder)
    index_document(CHUNKS, document_id="qd-12", embedder=embedder, index=index)

    cited = all_chunks(index, ChunkFilter(reference_keys=("dieu:2/khoan:1",)))
    whole_article = all_chunks(index, ChunkFilter(reference_keys=("dieu:2",)))
    body = all_chunks(index, ChunkFilter(attributes={"part": TextIn(values=("body",))}))

    assert [c.chunk_id for c in cited] == ["c2"]
    assert [c.chunk_id for c in whole_article] == ["c1", "c2"]
    assert [c.chunk_id for c in body] == ["c1", "c2"]
