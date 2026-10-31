"""The value types in index.py. ChunkIndex's behaviour is tested by the contract suite."""

from datetime import date, datetime

import pytest

from document_processor.retrieval.attributes import AttributeValue, TextIn
from document_processor.retrieval.embedding import DenseVector, Embedding
from document_processor.retrieval.index import (
    ChunkFilter,
    ChunkKey,
    GeneratedEntry,
    GeneratedEntryKey,
    IndexedChunk,
    SearchQuery,
    UnsupportedConditionError,
)

QUERY = Embedding(
    embedder_identity="stub", vectors={"dense": DenseVector(values=(1.0, 0.0))}
)


def chunk_with(**attributes: AttributeValue) -> IndexedChunk:
    return IndexedChunk(
        document_id="law",
        chunk_id="a",
        position=0,
        text="t",
        embedded_text="t",
        attributes=attributes,
    )


# ChunkFilter ------------------------------------------------------------------------------


@pytest.mark.parametrize("part", ["document_ids", "chunk_keys", "reference_keys"])
def test_a_filter_refuses_an_empty_tuple(part: str) -> None:
    with pytest.raises(ValueError, match=f"{part} must not be empty; use None for any"):
        ChunkFilter(**{part: ()})  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["Title", "issuer.name", "1st", ""])
def test_a_filter_refuses_an_attribute_name_stores_cannot_hold(name: str) -> None:
    with pytest.raises(ValueError, match="attribute name"):
        ChunkFilter(attributes={name: TextIn(values=("x",))})


def test_only_an_empty_filter_selects_everything() -> None:
    assert ChunkFilter().selects_everything
    assert not ChunkFilter(document_ids=("law",)).selects_everything
    assert not ChunkFilter(
        chunk_keys=(ChunkKey(document_id="law", chunk_id="a"),)
    ).selects_everything
    assert not ChunkFilter(reference_keys=("dieu:1",)).selects_everything
    assert not ChunkFilter(
        attributes={"role": TextIn(values=("main",))}
    ).selects_everything


def test_a_filter_keeps_its_own_copy_of_the_conditions() -> None:
    conditions = {"role": TextIn(values=("main",))}
    chunk_filter = ChunkFilter(attributes=conditions)
    conditions["status"] = TextIn(values=("draft",))
    assert set(chunk_filter.attributes) == {"role"}


# Keys -------------------------------------------------------------------------------------


def test_keys_with_the_same_parts_are_equal_and_hash_alike() -> None:
    first = ChunkKey(document_id="law", chunk_id="a")
    second = ChunkKey(document_id="law", chunk_id="a")
    assert first == second
    assert len({first, second}) == 1


def test_a_key_never_equals_a_plain_tuple() -> None:
    plain: object = (
        "law",
        "a",
    )  # typed as object: mypy already refuses the comparison otherwise
    assert ChunkKey(document_id="law", chunk_id="a") != plain


def test_a_chunk_and_an_entry_give_their_keys() -> None:
    entry = GeneratedEntry(
        document_id="law", generator="g", entry_id="q", chunk_ids=("a",), text="Ai?"
    )
    assert chunk_with().key == ChunkKey(document_id="law", chunk_id="a")
    assert entry.key == GeneratedEntryKey(
        document_id="law", generator="g", entry_id="q"
    )


# IndexedChunk -----------------------------------------------------------------------------


def test_a_chunk_refuses_a_datetime_attribute() -> None:
    with pytest.raises(TypeError, match="not datetimes"):
        chunk_with(issued_on=datetime(2024, 1, 18))


def test_a_chunk_refuses_a_number_that_is_not_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        chunk_with(weight=float("inf"))


def test_a_chunk_refuses_an_attribute_name_stores_cannot_hold() -> None:
    with pytest.raises(ValueError, match="attribute name"):
        chunk_with(Title="x")


def test_a_chunk_keeps_its_own_copy_of_the_attributes() -> None:
    attributes: dict[str, AttributeValue] = {"issued_on": date(2024, 1, 18)}
    chunk = IndexedChunk(
        document_id="law",
        chunk_id="a",
        position=0,
        text="t",
        embedded_text="t",
        attributes=attributes,
    )
    attributes["issued_on"] = date(2000, 1, 1)
    assert chunk.attributes["issued_on"] == date(2024, 1, 18)
    with pytest.raises(TypeError):
        chunk.attributes["issued_on"] = date(2000, 1, 1)  # type: ignore[index]


# GeneratedEntry ---------------------------------------------------------------------------


def test_an_entry_must_lead_to_a_chunk() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        GeneratedEntry(
            document_id="law", generator="g", entry_id="q", chunk_ids=(), text="Ai?"
        )


@pytest.mark.parametrize("text", ["", " \n"])
def test_an_entry_refuses_blank_text(text: str) -> None:
    with pytest.raises(ValueError, match="no text"):
        GeneratedEntry(
            document_id="law", generator="g", entry_id="q", chunk_ids=("a",), text=text
        )


# SearchQuery ------------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [0, -1])
def test_a_query_needs_a_positive_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        SearchQuery(embedding=QUERY, limit=limit)


def test_a_query_refuses_no_vector_names() -> None:
    with pytest.raises(ValueError, match="use None for all"):
        SearchQuery(embedding=QUERY, vector_names=())


def test_a_query_refuses_a_vector_name_twice() -> None:
    with pytest.raises(ValueError, match="repeats a name"):
        SearchQuery(embedding=QUERY, vector_names=("dense", "dense"))


def test_a_query_refuses_no_generators() -> None:
    with pytest.raises(ValueError, match='use "none" for no entries'):
        SearchQuery(embedding=QUERY, generated_entries=())


def test_a_query_searches_everywhere_with_everything_by_default() -> None:
    query = SearchQuery(embedding=QUERY)
    assert query.chunk_filter.selects_everything
    assert (query.vector_names, query.generated_entries, query.limit) == (
        None,
        "all",
        10,
    )


# Errors -----------------------------------------------------------------------------------


def test_an_unsupported_condition_is_a_type_error() -> None:
    assert issubclass(UnsupportedConditionError, TypeError)
