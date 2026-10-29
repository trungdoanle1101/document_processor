"""Tests for retrieval/embedding.py: named vectors, their specs, and the Embedder protocol.

An Embedding holds one text's vectors by name, each a DenseVector or a
SparseVector. An embedder declares the vectors it produces (vector_specs),
and check_embedding confirms an embedding matches them: the same names, each
of its kind, dense ones of their dimension. Dense and sparse are equals, so a
dense model alone, BM25 alone, or both together are all just embedders.

SparseVector checks only what would otherwise fail silently or far away:
equal lengths, non-negative indices (Qdrant stores them unsigned) and unique
indices (Qdrant rejects duplicates, but only at upsert, inside a job).

The protocol is checked through StubEmbedder (tests/retrieval/stub_embedder.py),
which later tests (index, search) use in place of a model.
"""

from dataclasses import FrozenInstanceError

import pytest

from document_processor.chunking.tokenizer import WordTokenizer
from document_processor.retrieval.embedding import (
    DenseVector,
    DenseVectorSpec,
    Embedder,
    Embedding,
    SparseVector,
    SparseVectorSpec,
    VectorSpec,
    check_embedding,
)
from tests.retrieval.stub_embedder import StubEmbedder


def sparse(*pairs: tuple[int, float]) -> SparseVector:
    return SparseVector(indices=tuple(i for i, _ in pairs), values=tuple(v for _, v in pairs))


def dense(*values: float) -> DenseVector:
    return DenseVector(values=values)


# ============================================================ DenseVector


def test_dense_vector_keeps_its_values() -> None:
    assert dense(0.1, 0.2).values == (0.1, 0.2)


def test_empty_dense_vector_is_refused() -> None:
    with pytest.raises(ValueError, match="^a dense vector must not be empty$"):
        DenseVector(values=())


def test_a_zero_dense_vector_is_not_refused() -> None:
    # Only emptiness is checked; whether a vector is usable is the model's business.
    assert dense(0.0, 0.0).values == (0.0, 0.0)


def test_dense_vector_is_frozen_and_takes_keywords_only() -> None:
    with pytest.raises(FrozenInstanceError):
        dense(1.0).values = (2.0,)  # type: ignore[misc]
    with pytest.raises(TypeError):
        DenseVector((1.0,))  # type: ignore[call-arg]


# ============================================================ SparseVector


def test_sparse_vector_keeps_indices_and_values() -> None:
    vector = sparse((7, 0.5), (3, 0.25))
    assert (vector.indices, vector.values) == ((7, 3), (0.5, 0.25))


def test_empty_sparse_vector_is_valid() -> None:
    assert SparseVector(indices=(), values=()).indices == ()


def test_index_zero_and_large_indices_are_valid() -> None:
    assert sparse((0, 1.0), (2**31 - 1, 1.0)).indices == (0, 2**31 - 1)


@pytest.mark.parametrize("value", [0.0, -0.5, 1e-9, 123.0])
def test_any_value_is_accepted(value: float) -> None:
    assert sparse((1, value)).values == (value,)


@pytest.mark.parametrize(
    ("indices", "values", "message"),
    [
        ((1, 2, 3), (0.5, 0.5), "3 indices but 2 values"),
        ((), (0.5,), "0 indices but 1 values"),
        ((1,), (), "1 indices but 0 values"),
    ],
)
def test_lengths_must_match(indices: tuple[int, ...], values: tuple[float, ...], message: str) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        SparseVector(indices=indices, values=values)


def test_the_smallest_negative_index_is_named() -> None:
    with pytest.raises(ValueError, match="^negative index -7$"):
        sparse((-2, 1.0), (5, 1.0), (-7, 1.0))


def test_the_index_repeated_first_is_named() -> None:
    with pytest.raises(ValueError, match="^index 9 appears more than once$"):
        sparse((5, 1.0), (9, 1.0), (9, 1.0), (5, 1.0))


def test_sparse_vectors_compare_by_value() -> None:
    assert sparse((1, 0.5), (2, 0.5)) == sparse((1, 0.5), (2, 0.5))
    assert sparse((1, 0.5), (2, 0.5)) != sparse((2, 0.5), (1, 0.5))


# ============================================================ vector specs


def test_dense_spec_needs_a_positive_dimension() -> None:
    assert DenseVectorSpec(dimension=1024).dimension == 1024
    for dimension in (0, -1):
        with pytest.raises(ValueError, match=f"^dimension must be positive, got {dimension}$"):
            DenseVectorSpec(dimension=dimension)


def test_sparse_spec_values_are_final_unless_said_otherwise() -> None:
    assert SparseVectorSpec().needs_idf is False
    assert SparseVectorSpec(needs_idf=True).needs_idf is True


def test_specs_are_frozen_values() -> None:
    assert DenseVectorSpec(dimension=8) == DenseVectorSpec(dimension=8)
    assert SparseVectorSpec() != SparseVectorSpec(needs_idf=True)
    with pytest.raises(FrozenInstanceError):
        DenseVectorSpec(dimension=8).dimension = 16  # type: ignore[misc]


# ============================================================ Embedding


def test_an_embedding_holds_vectors_by_name() -> None:
    embedding = Embedding(vectors={"dense": dense(1.0, 0.0), "lexical": sparse((4, 0.7))})
    assert embedding.dense("dense") == dense(1.0, 0.0)
    assert embedding.sparse("lexical") == sparse((4, 0.7))


def test_a_dense_only_and_a_sparse_only_embedding() -> None:
    assert Embedding(vectors={"dense": dense(1.0)}).dense("dense").values == (1.0,)
    assert Embedding(vectors={"bm25": sparse()}).sparse("bm25") == SparseVector(indices=(), values=())


def test_several_vectors_of_one_kind() -> None:
    embedding = Embedding(vectors={"bge-m3.lexical": sparse((1, 1.0)), "bm25.sparse": sparse((2, 1.0))})
    assert embedding.sparse("bge-m3.lexical") != embedding.sparse("bm25.sparse")


def test_an_embedding_without_vectors_is_refused() -> None:
    with pytest.raises(ValueError, match="^an embedding must hold at least one vector$"):
        Embedding(vectors={})


def test_an_empty_vector_name_is_refused() -> None:
    with pytest.raises(ValueError, match="^vector names must not be empty$"):
        Embedding(vectors={"": dense(1.0)})


def test_asking_for_the_wrong_kind_says_what_it_is() -> None:
    embedding = Embedding(vectors={"dense": dense(1.0), "sparse": sparse()})
    with pytest.raises(TypeError, match="^vector 'dense' is dense, not sparse$"):
        embedding.sparse("dense")
    with pytest.raises(TypeError, match="^vector 'sparse' is sparse, not dense$"):
        embedding.dense("sparse")


def test_asking_for_a_missing_vector_lists_the_names() -> None:
    embedding = Embedding(vectors={"dense": dense(1.0), "sparse": sparse()})
    with pytest.raises(KeyError, match=r"no vector 'lexical'; this embedding has: dense, sparse"):
        embedding.dense("lexical")


def test_the_vectors_cannot_be_changed_afterwards() -> None:
    source = {"dense": dense(1.0)}
    embedding = Embedding(vectors=source)
    source["other"] = dense(2.0)  # changing the dict passed in does not reach the embedding
    assert list(embedding.vectors) == ["dense"]
    with pytest.raises(TypeError):
        embedding.vectors["other"] = dense(2.0)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        embedding.vectors = {}  # type: ignore[misc]


def test_embeddings_compare_by_value() -> None:
    assert Embedding(vectors={"dense": dense(1.0)}) == Embedding(vectors={"dense": dense(1.0)})
    assert Embedding(vectors={"dense": dense(1.0)}) != Embedding(vectors={"other": dense(1.0)})


# ============================================================ check_embedding

SPECS: dict[str, VectorSpec] = {"dense": DenseVectorSpec(dimension=2), "sparse": SparseVectorSpec()}


def test_a_matching_embedding_passes() -> None:
    check_embedding(Embedding(vectors={"dense": dense(1.0, 0.0), "sparse": sparse()}), SPECS)


def test_a_missing_or_extra_vector_is_named() -> None:
    with pytest.raises(ValueError, match="^the embedding holds dense, but the specs are for dense, sparse$"):
        check_embedding(Embedding(vectors={"dense": dense(1.0, 0.0)}), SPECS)
    with pytest.raises(ValueError, match="holds dense, extra, sparse"):
        check_embedding(Embedding(vectors={"dense": dense(1.0, 0.0), "sparse": sparse(), "extra": sparse()}), SPECS)


def test_the_wrong_kind_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="^vector 'dense' should be dense but is sparse$"):
        check_embedding(Embedding(vectors={"dense": sparse(), "sparse": sparse()}), SPECS)
    with pytest.raises(TypeError, match="^vector 'sparse' should be sparse but is dense$"):
        check_embedding(Embedding(vectors={"dense": dense(1.0, 0.0), "sparse": dense(1.0)}), SPECS)


def test_the_wrong_dimension_is_refused() -> None:
    with pytest.raises(ValueError, match="^vector 'dense' has 3 values, but its spec says 2$"):
        check_embedding(Embedding(vectors={"dense": dense(1.0, 0.0, 0.0), "sparse": sparse()}), SPECS)


# ============================================================ the protocol, through the stub


def test_stub_is_an_embedder() -> None:
    embedder: Embedder = StubEmbedder()  # mypy checks the protocol here
    assert embedder.vector_specs == {"dense": DenseVectorSpec(dimension=8), "sparse": SparseVectorSpec()}
    assert embedder.tokenizer is None


@pytest.mark.parametrize(
    "stub",
    [
        StubEmbedder(),
        StubEmbedder(sparse=None),
        StubEmbedder(dense=None, sparse="bm25", needs_idf=True),
        StubEmbedder(dense="e5", dimension=4, sparse="splade"),
    ],
)
def test_every_embedding_matches_the_specs(stub: StubEmbedder) -> None:
    for embedding in [*stub.encode_passages(["Điều 1. Phạm vi điều chỉnh", ""]), stub.encode_query("Điều 14")]:
        check_embedding(embedding, stub.vector_specs)


def test_a_stub_without_any_vector_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a dense or a sparse vector"):
        StubEmbedder(dense=None, sparse=None)


def test_stub_vector_specs_cannot_be_changed() -> None:
    with pytest.raises(TypeError):
        StubEmbedder().vector_specs["extra"] = SparseVectorSpec()  # type: ignore[index]


def test_one_embedding_per_text_in_order() -> None:
    texts = ["Điều 1. Phạm vi điều chỉnh", "Điều 2. Đối tượng áp dụng", "Điều 1. Phạm vi điều chỉnh"]
    embeddings = StubEmbedder().encode_passages(texts)
    assert len(embeddings) == 3
    assert embeddings[0] == embeddings[2] != embeddings[1]


def test_no_texts_give_no_embeddings() -> None:
    assert StubEmbedder().encode_passages([]) == []


def test_stubs_built_alike_share_an_identity_and_their_vectors() -> None:
    first, second = StubEmbedder(dimension=4), StubEmbedder(dimension=4)
    assert first.identity == second.identity
    assert first.encode_query("Điều 14") == second.encode_query("Điều 14")


@pytest.mark.parametrize(
    "other",
    [
        StubEmbedder(dimension=16),
        StubEmbedder(sparse=None),
        StubEmbedder(needs_idf=True),
        StubEmbedder(dense="e5"),
        StubEmbedder(dense=None),
    ],
)
def test_stubs_built_differently_have_different_identities(other: StubEmbedder) -> None:
    assert StubEmbedder().identity != other.identity


def test_stub_records_which_method_was_called() -> None:
    stub = StubEmbedder()
    stub.encode_passages(["một", "hai"])
    stub.encode_query("ba")
    stub.encode_passages(["bốn"])
    assert stub.passage_batches == [("một", "hai"), ("bốn",)]
    assert stub.queries == ["ba"]


def test_stub_query_matches_a_passage_with_the_same_words() -> None:
    stub = StubEmbedder()
    assert stub.encode_query("phạm vi ĐIỀU CHỈNH") == stub.encode_passages(["Phạm vi điều chỉnh"])[0]


def test_stub_sparse_counts_repeated_words() -> None:
    assert sorted(StubEmbedder().encode_query("khoản khoản điều").sparse("sparse").values) == [1.0, 2.0]


def test_stub_text_without_words_is_not_a_zero_vector() -> None:
    embedding = StubEmbedder().encode_query("... !!")
    assert any(embedding.dense("dense").values)
    assert embedding.sparse("sparse") == SparseVector(indices=(), values=())


def test_stub_can_carry_a_tokenizer() -> None:
    assert isinstance(StubEmbedder(tokenizer=WordTokenizer()).tokenizer, WordTokenizer)