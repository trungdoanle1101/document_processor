"""Tests for retrieval/embedder.py: SparseVector, Embedding and the Embedder protocol.

SparseVector holds token ids and their weights. It checks only what would
otherwise fail silently or far away: equal lengths (a mismatch pairs weights
with the wrong tokens), non-negative indices (Qdrant stores them unsigned) and
unique indices (Qdrant rejects duplicates, but only at upsert, inside a job).
Values may be anything, and an empty vector is valid.

Embedding needs a non-empty dense vector; its length against the model's
dimension is checked where both are known, in the index.

The protocol is checked through StubEmbedder (tests/retrieval/stub_embedder.py),
which later tests (index, search) use in place of a model.
"""

from dataclasses import FrozenInstanceError

import pytest

from document_processor.retrieval.embedder import Embedder, Embedding, SparseVector
from tests.retrieval.stub_embedder import StubEmbedder


def sparse(*pairs: tuple[int, float]) -> SparseVector:
    return SparseVector(indices=tuple(i for i, _ in pairs), values=tuple(v for _, v in pairs))


# ============================================================ SparseVector


def test_sparse_vector_keeps_indices_and_values() -> None:
    vector = sparse((7, 0.5), (3, 0.25))
    assert vector.indices == (7, 3)
    assert vector.values == (0.5, 0.25)


def test_indices_need_not_be_sorted() -> None:
    assert sparse((9, 1.0), (2, 1.0), (5, 1.0)).indices == (9, 2, 5)


def test_empty_sparse_vector_is_valid() -> None:
    vector = SparseVector(indices=(), values=())
    assert (vector.indices, vector.values) == ((), ())


def test_index_zero_is_valid() -> None:
    assert sparse((0, 1.0)).indices == (0,)


def test_large_index_is_valid() -> None:
    # bge-m3's vocabulary is about 250,000 tokens; other encoders hash into 2**31.
    assert sparse((2**31 - 1, 1.0)).indices == (2**31 - 1,)


@pytest.mark.parametrize("value", [0.0, -0.5, 1e-9, 123.0])
def test_any_value_is_accepted(value: float) -> None:
    # Weights are positive for bge-m3 after its ReLU, but that is the model's
    # property, not the type's.
    assert sparse((1, value)).values == (value,)


@pytest.mark.parametrize(
    ("indices", "values", "message"),
    [
        ((1, 2, 3), (0.5, 0.5), "3 indices but 2 values"),
        ((1,), (0.5, 0.5), "1 indices but 2 values"),
        ((), (0.5,), "0 indices but 1 values"),
        ((1,), (), "1 indices but 0 values"),
    ],
)
def test_lengths_must_match(indices: tuple[int, ...], values: tuple[float, ...], message: str) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        SparseVector(indices=indices, values=values)


def test_negative_index_is_refused() -> None:
    with pytest.raises(ValueError, match="^negative index -1$"):
        sparse((4, 1.0), (-1, 1.0))


def test_the_smallest_negative_index_is_named() -> None:
    with pytest.raises(ValueError, match="^negative index -7$"):
        sparse((-2, 1.0), (5, 1.0), (-7, 1.0))


def test_duplicate_index_is_refused() -> None:
    with pytest.raises(ValueError, match="^index 3 appears more than once$"):
        sparse((3, 1.0), (8, 1.0), (3, 2.0))


def test_the_index_repeated_first_is_named() -> None:
    with pytest.raises(ValueError, match="^index 9 appears more than once$"):
        sparse((5, 1.0), (9, 1.0), (9, 1.0), (5, 1.0))


def test_lengths_are_checked_before_indices() -> None:
    with pytest.raises(ValueError, match="indices but"):
        SparseVector(indices=(-1, -1), values=(1.0,))


def test_sparse_vector_is_frozen() -> None:
    vector = sparse((1, 1.0))
    with pytest.raises(FrozenInstanceError):
        vector.indices = (2,)  # type: ignore[misc]


def test_sparse_vector_takes_keywords_only() -> None:
    with pytest.raises(TypeError):
        SparseVector((1,), (1.0,))  # type: ignore[call-arg]


def test_sparse_vectors_compare_by_value() -> None:
    assert sparse((1, 0.5), (2, 0.5)) == sparse((1, 0.5), (2, 0.5))
    assert sparse((1, 0.5), (2, 0.5)) != sparse((2, 0.5), (1, 0.5))


# ============================================================ Embedding


def test_embedding_with_dense_and_sparse() -> None:
    embedding = Embedding(dense=(0.1, 0.2, 0.3), sparse=sparse((4, 0.7)))
    assert embedding.dense == (0.1, 0.2, 0.3)
    assert embedding.sparse == sparse((4, 0.7))


def test_embedding_without_sparse() -> None:
    assert Embedding(dense=(1.0,), sparse=None).sparse is None


def test_empty_sparse_part_is_not_the_same_as_none() -> None:
    assert Embedding(dense=(1.0,), sparse=sparse()).sparse == SparseVector(indices=(), values=())


def test_empty_dense_vector_is_refused() -> None:
    with pytest.raises(ValueError, match="^dense vector must not be empty$"):
        Embedding(dense=(), sparse=None)


def test_a_zero_dense_vector_is_not_refused() -> None:
    # Only emptiness is checked; whether a vector is usable is the model's business.
    assert Embedding(dense=(0.0, 0.0), sparse=None).dense == (0.0, 0.0)


def test_sparse_must_be_given() -> None:
    with pytest.raises(TypeError):
        Embedding(dense=(1.0,))  # type: ignore[call-arg]


def test_embedding_is_frozen() -> None:
    embedding = Embedding(dense=(1.0,), sparse=None)
    with pytest.raises(FrozenInstanceError):
        embedding.sparse = sparse()  # type: ignore[misc]


def test_embeddings_compare_by_value() -> None:
    assert Embedding(dense=(1.0, 2.0), sparse=sparse((1, 1.0))) == Embedding(
        dense=(1.0, 2.0), sparse=sparse((1, 1.0))
    )
    assert Embedding(dense=(1.0, 2.0), sparse=None) != Embedding(dense=(1.0, 2.0), sparse=sparse())


# ============================================================ the protocol, through the stub


def test_stub_is_an_embedder() -> None:
    embedder: Embedder = StubEmbedder()  # mypy checks the protocol here
    assert embedder.dense_dimension == 8
    assert embedder.has_sparse


def test_one_embedding_per_text_in_order() -> None:
    embedder = StubEmbedder()
    texts = ["Điều 1. Phạm vi điều chỉnh", "Điều 2. Đối tượng áp dụng", "Điều 1. Phạm vi điều chỉnh"]
    embeddings = embedder.encode_documents(texts)
    assert len(embeddings) == 3
    assert embeddings[0] == embeddings[2] != embeddings[1]


def test_no_texts_give_no_embeddings() -> None:
    assert StubEmbedder().encode_documents([]) == []


@pytest.mark.parametrize("dimension", [1, 4, 1024])
def test_dense_vectors_have_the_stated_dimension(dimension: int) -> None:
    embedder = StubEmbedder(dimension=dimension)
    for embedding in [*embedder.encode_documents(["Nội dung", ""]), embedder.encode_query("Điều 14")]:
        assert len(embedding.dense) == embedder.dense_dimension


def test_has_sparse_means_every_embedding_has_sparse() -> None:
    embedder = StubEmbedder(sparse=True)
    embeddings = [*embedder.encode_documents(["a b", ""]), embedder.encode_query("c")]
    assert all(embedding.sparse is not None for embedding in embeddings)


def test_no_sparse_means_no_embedding_has_sparse() -> None:
    embedder = StubEmbedder(sparse=False)
    assert not embedder.has_sparse
    embeddings = [*embedder.encode_documents(["a b", ""]), embedder.encode_query("c")]
    assert all(embedding.sparse is None for embedding in embeddings)


def test_stub_records_which_method_was_called() -> None:
    embedder = StubEmbedder()
    embedder.encode_documents(["một", "hai"])
    embedder.encode_query("ba")
    embedder.encode_documents(["bốn"])
    assert embedder.document_batches == [("một", "hai"), ("bốn",)]
    assert embedder.queries == ["ba"]


def test_stub_query_matches_a_document_with_the_same_words() -> None:
    embedder = StubEmbedder()
    document = embedder.encode_documents(["Phạm vi điều chỉnh"])[0]
    assert embedder.encode_query("phạm vi ĐIỀU CHỈNH") == document


def test_stub_sparse_counts_repeated_words() -> None:
    embedding = StubEmbedder().encode_query("khoản khoản điều")
    assert embedding.sparse is not None
    assert sorted(embedding.sparse.values) == [1.0, 2.0]


def test_stub_text_without_words_is_not_a_zero_vector() -> None:
    embedding = StubEmbedder().encode_query("... !!")
    assert any(embedding.dense)
    assert embedding.sparse == SparseVector(indices=(), values=())