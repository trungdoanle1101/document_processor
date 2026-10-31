import pytest

from document_processor.retrieval.embedding import (
    DenseVector,
    DenseVectorSpec,
    Embedder,
    Embedding,
    SparseVector,
    SparseVectorSpec,
    Vector,
    VectorSpec,
    check_embedding,
)
from tests.retrieval.stub_embedder import StubEmbedder

IDENTITY = "stub"
SPECS: dict[str, VectorSpec] = {
    "dense": DenseVectorSpec(dimension=2),
    "sparse": SparseVectorSpec(),
}


def embedding(vectors: dict[str, Vector], identity: str = IDENTITY) -> Embedding:
    return Embedding(embedder_identity=identity, vectors=vectors)


def fitting() -> Embedding:
    """An embedding that fits SPECS exactly."""
    return embedding(
        {
            "dense": DenseVector(values=(1.0, 0.0)),
            "sparse": SparseVector(indices=(3,), values=(1.0,)),
        }
    )


# DenseVector, SparseVector ----------------------------------------------------------------


def test_a_dense_vector_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        DenseVector(values=())


def test_a_sparse_vector_may_be_empty() -> None:
    assert SparseVector(indices=(), values=()).indices == ()


def test_a_sparse_vector_needs_a_value_per_index() -> None:
    with pytest.raises(ValueError, match="2 indices but 1 values"):
        SparseVector(indices=(1, 2), values=(1.0,))


def test_a_sparse_vector_refuses_a_negative_index() -> None:
    with pytest.raises(ValueError, match="negative index -1"):
        SparseVector(indices=(3, -1), values=(1.0, 1.0))


def test_a_sparse_vector_refuses_an_index_twice() -> None:
    with pytest.raises(ValueError, match="index 3 appears more than once"):
        SparseVector(indices=(3, 3), values=(1.0, 2.0))


# Specs ------------------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", [0, -1])
def test_a_dense_spec_needs_a_positive_dimension(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimension must be positive"):
        DenseVectorSpec(dimension=dimension)


def test_a_sparse_spec_takes_final_weights_by_default() -> None:
    assert SparseVectorSpec().needs_idf is False


# Embedding --------------------------------------------------------------------------------


def test_an_embedding_must_name_its_embedder() -> None:
    with pytest.raises(ValueError, match="name the embedder"):
        embedding({"dense": DenseVector(values=(1.0,))}, identity="")


def test_an_embedding_needs_a_vector() -> None:
    with pytest.raises(ValueError, match="at least one vector"):
        embedding({})


def test_vector_names_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="names must not be empty"):
        embedding({"": DenseVector(values=(1.0,))})


def test_an_embedding_keeps_its_own_copy_of_the_vectors() -> None:
    vectors: dict[str, Vector] = {"dense": DenseVector(values=(1.0,))}
    held = embedding(vectors)
    vectors["other"] = DenseVector(values=(2.0,))
    assert set(held.vectors) == {"dense"}
    with pytest.raises(TypeError):
        held.vectors["other"] = DenseVector(values=(2.0,))  # type: ignore[index]


def test_embeddings_are_equal_only_with_the_same_identity() -> None:
    vectors: dict[str, Vector] = {"dense": DenseVector(values=(1.0,))}
    assert embedding(vectors) == embedding(vectors)
    assert embedding(vectors) != embedding(vectors, identity="other")


def test_dense_and_sparse_return_the_named_vector() -> None:
    held = fitting()
    assert held.dense("dense") == DenseVector(values=(1.0, 0.0))
    assert held.sparse("sparse") == SparseVector(indices=(3,), values=(1.0,))


def test_asking_for_the_wrong_kind_raises_type_error() -> None:
    held = fitting()
    with pytest.raises(TypeError, match="'dense' is dense, not sparse"):
        held.sparse("dense")
    with pytest.raises(TypeError, match="'sparse' is sparse, not dense"):
        held.dense("sparse")


def test_asking_for_a_missing_vector_lists_the_names() -> None:
    with pytest.raises(
        KeyError, match="no vector 'colbert'; this embedding has: dense, sparse"
    ):
        fitting().dense("colbert")


# check_embedding --------------------------------------------------------------------------


def test_check_embedding_accepts_an_embedding_that_fits() -> None:
    check_embedding(fitting(), IDENTITY, SPECS)


def test_check_embedding_refuses_another_embedders_vectors() -> None:
    other = Embedding(embedder_identity="other", vectors=fitting().vectors)
    with pytest.raises(ValueError, match="made by 'other', not 'stub'"):
        check_embedding(other, IDENTITY, SPECS)


def test_check_embedding_checks_the_identity_before_the_vectors() -> None:
    with pytest.raises(ValueError, match="made by 'other'"):
        check_embedding(
            embedding({"wrong": DenseVector(values=(1.0,))}, identity="other"),
            IDENTITY,
            SPECS,
        )


@pytest.mark.parametrize(
    "vectors",
    [
        {"dense": DenseVector(values=(1.0, 0.0))},
        {
            "dense": DenseVector(values=(1.0, 0.0)),
            "sparse": SparseVector(indices=(), values=()),
            "extra": DenseVector(values=(1.0,)),
        },
    ],
    ids=["missing", "extra"],
)
def test_check_embedding_needs_exactly_the_specified_names(
    vectors: dict[str, Vector],
) -> None:
    with pytest.raises(ValueError, match="but the specs are for dense, sparse"):
        check_embedding(embedding(vectors), IDENTITY, SPECS)


def test_check_embedding_refuses_a_sparse_vector_where_dense_is_specified() -> None:
    vectors: dict[str, Vector] = {
        "dense": SparseVector(indices=(1,), values=(1.0,)),
        "sparse": SparseVector(indices=(), values=()),
    }
    with pytest.raises(TypeError, match="'dense' should be dense but is sparse"):
        check_embedding(embedding(vectors), IDENTITY, SPECS)


def test_check_embedding_refuses_a_dense_vector_where_sparse_is_specified() -> None:
    vectors: dict[str, Vector] = {
        "dense": DenseVector(values=(1.0, 0.0)),
        "sparse": DenseVector(values=(1.0,)),
    }
    with pytest.raises(TypeError, match="'sparse' should be sparse but is dense"):
        check_embedding(embedding(vectors), IDENTITY, SPECS)


def test_check_embedding_refuses_the_wrong_dimension() -> None:
    vectors: dict[str, Vector] = {
        "dense": DenseVector(values=(1.0, 0.0, 0.0)),
        "sparse": SparseVector(indices=(), values=()),
    }
    with pytest.raises(ValueError, match="'dense' has 3 values, but its spec says 2"):
        check_embedding(embedding(vectors), IDENTITY, SPECS)


# StubEmbedder -----------------------------------------------------------------------------


def test_the_stub_is_an_embedder() -> None:
    embedder: Embedder = StubEmbedder()  # checked by mypy
    assert embedder.tokenizer is None


def test_the_stub_identity_names_its_vectors() -> None:
    assert StubEmbedder().identity == "stub/dense:8/sparse:final"
    assert (
        StubEmbedder(dimension=4, needs_idf=True).identity == "stub/dense:4/sparse:idf"
    )
    assert StubEmbedder(sparse=None).identity == "stub/dense:8/-:final"


def test_the_stub_specs_follow_its_options() -> None:
    assert StubEmbedder(dimension=4).vector_specs == {
        "dense": DenseVectorSpec(dimension=4),
        "sparse": SparseVectorSpec(needs_idf=False),
    }
    assert StubEmbedder(dense="bge", sparse=None).vector_specs == {
        "bge": DenseVectorSpec(dimension=8)
    }
    assert StubEmbedder(dense=None, needs_idf=True).vector_specs == {
        "sparse": SparseVectorSpec(needs_idf=True)
    }


def test_the_stub_needs_a_vector() -> None:
    with pytest.raises(
        ValueError, match="needs a dense vector, a sparse vector, or both"
    ):
        StubEmbedder(dense=None, sparse=None)


@pytest.mark.parametrize(
    "embedder",
    [
        StubEmbedder(),
        StubEmbedder(sparse=None),
        StubEmbedder(dense=None, needs_idf=True),
    ],
)
def test_the_stub_produces_embeddings_that_fit_its_specs(
    embedder: StubEmbedder,
) -> None:
    for produced in [
        embedder.encode_query("Điều 14"),
        *embedder.encode_passages(["một hai", "", "ba ba ba"]),
    ]:
        check_embedding(produced, embedder.identity, embedder.vector_specs)


def test_the_stub_records_what_it_encodes() -> None:
    embedder = StubEmbedder()
    embedder.encode_query("question")
    embedder.encode_passages(["first", "second"])
    embedder.encode_passages(["third"])
    assert embedder.queries == ["question"]
    assert embedder.passage_batches == [["first", "second"], ["third"]]


def test_the_stub_returns_one_embedding_per_passage_in_order() -> None:
    embedder = StubEmbedder()
    first, second = embedder.encode_passages(["đất đai", "chuyển nhượng"])
    assert first == embedder.encode_query("đất đai")
    assert second == embedder.encode_query("chuyển nhượng")


def test_the_stub_is_deterministic_and_ignores_case() -> None:
    assert StubEmbedder().encode_query("Điều 14") == StubEmbedder().encode_query(
        "điều 14"
    )


def test_the_stub_counts_words() -> None:
    produced = StubEmbedder(dimension=4).encode_query("ba ba ba một")
    assert sum(produced.dense("dense").values) == 4.0
    assert sorted(produced.sparse("sparse").values) == [1.0, 3.0]


def test_shared_words_give_shared_sparse_terms() -> None:
    embedder = StubEmbedder()
    query = set(embedder.encode_query("quyền sử dụng đất").sparse("sparse").indices)
    related = set(
        embedder.encode_query("chuyển nhượng quyền sử dụng").sparse("sparse").indices
    )
    unrelated = set(embedder.encode_query("ngân hàng").sparse("sparse").indices)
    assert len(query & related) == 3
    assert not query & unrelated


def test_empty_text_gives_a_zero_dense_vector_and_no_sparse_terms() -> None:
    produced = StubEmbedder(dimension=3).encode_query("")
    assert produced.dense("dense").values == (0.0, 0.0, 0.0)
    assert produced.sparse("sparse").indices == ()


def test_the_stub_passes_its_tokenizer_through() -> None:
    class CountingWords:
        def count_tokens(self, text: str) -> int:
            return len(text.split())

        @property
        def model_max_tokens(self) -> int | None:
            return 512

    tokenizer = CountingWords()
    assert StubEmbedder(tokenizer=tokenizer).tokenizer is tokenizer
