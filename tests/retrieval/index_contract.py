"""Tests every ChunkIndex must pass.

An implementation's tests subclass ChunkIndexContract (writes, filters, listing) and
ChunkIndexSearchContract (search), each saying how to make an index with make_index.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pytest

from document_processor.retrieval.attributes import (
    AttributeValue,
    BooleanEquals,
    Condition,
    DateBetween,
    ElementCondition,
    EveryElement,
    IntegerIn,
    NoneOf,
    NumberBetween,
    Scalar,
    TextIn,
)
from document_processor.retrieval.embedding import (
    DenseVector,
    DenseVectorSpec,
    Embedding,
    SparseVector,
    SparseVectorSpec,
    VectorSpec,
)
from document_processor.retrieval.index import (
    ChunkFilter,
    ChunkHit,
    ChunkIndex,
    ChunkKey,
    GeneratedEntry,
    IndexedChunk,
    SearchQuery,
    UnsupportedConditionError,
)

EMBEDDER = "stub"  # the identity make_index must give the index
DENSE: Mapping[str, VectorSpec] = {"dense": DenseVectorSpec(dimension=2)}
HYBRID: Mapping[str, VectorSpec] = {"dense": DenseVectorSpec(dimension=2), "sparse": SparseVectorSpec()}
SPARSE_IDF: Mapping[str, VectorSpec] = {"sparse": SparseVectorSpec(needs_idf=True)}


def dense(x: float, y: float) -> Embedding:
    return Embedding(embedder_identity=EMBEDDER, vectors={"dense": DenseVector(values=(x, y))})


def hybrid(x: float, y: float, terms: Mapping[int, float]) -> Embedding:
    sparse = SparseVector(indices=tuple(terms), values=tuple(terms.values()))
    return Embedding(embedder_identity=EMBEDDER, vectors={"dense": DenseVector(values=(x, y)), "sparse": sparse})


def sparse(terms: Mapping[int, float]) -> Embedding:
    vector = SparseVector(indices=tuple(terms), values=tuple(terms.values()))
    return Embedding(embedder_identity=EMBEDDER, vectors={"sparse": vector})


def chunk(
    chunk_id: str,
    document_id: str = "law",
    position: int = 0,
    reference_keys: tuple[str, ...] = (),
    **attributes: AttributeValue,
) -> IndexedChunk:
    return IndexedChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        position=position,
        text=f"text of {chunk_id}",
        embedded_text=f"embedded {chunk_id}",
        reference_keys=reference_keys,
        attributes=attributes,
    )


def entry(entry_id: str, *chunk_ids: str, document_id: str = "law", generator: str = "questions-v1") -> GeneratedEntry:
    return GeneratedEntry(
        document_id=document_id, generator=generator, entry_id=entry_id, chunk_ids=chunk_ids, text=f"{entry_id}?"
    )


def where(**conditions: Condition) -> ChunkFilter:
    return ChunkFilter(attributes=conditions)


def ids(chunks: list[IndexedChunk]) -> list[str]:
    return [chunk.chunk_id for chunk in chunks]


def hit_ids(hits: list[ChunkHit]) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


@dataclass(frozen=True, kw_only=True)
class TextStartsWith(ElementCondition):
    """A condition no index knows in advance: shows that new conditions work, or are refused."""

    prefix: str

    def match_one(self, value: Scalar) -> bool:
        return isinstance(value, str) and value.startswith(self.prefix)


class _ContractHelpers:
    def make_index(self, vector_specs: Mapping[str, VectorSpec]) -> ChunkIndex:
        raise NotImplementedError

    def index_with(self, *chunks: IndexedChunk, vector_specs: Mapping[str, VectorSpec] = DENSE) -> ChunkIndex:
        """An index holding these chunks, each document's replaced at once, all embedded (1, 0)."""
        index = self.make_index(vector_specs)
        for document_id in dict.fromkeys(c.document_id for c in chunks):
            own = [c for c in chunks if c.document_id == document_id]
            index.replace_document(document_id, own, [dense(1, 0)] * len(own))
        return index

    def index_with_entries(self) -> ChunkIndex:
        """Chunks a and b far from the query (1, 0); entry q, about a, close to it."""
        index = self.make_index(DENSE)
        index.replace_document("law", [chunk("a", position=0), chunk("b", position=1)], [dense(0, 1), dense(0, 1)])
        index.replace_generated("law", "questions-v1", [entry("q", "a")], [dense(1, 0)])
        return index

    def all_chunks(self, index: ChunkIndex, chunk_filter: ChunkFilter | None = None) -> list[str]:
        return ids(index.list_chunks(chunk_filter or ChunkFilter(), limit=1000))


class ChunkIndexContract(_ContractHelpers):
    """Writes, filters and listing: everything but search."""

    # replace_document ---------------------------------------------------------------------

    def test_replacing_a_document_keeps_only_its_new_chunks(self) -> None:
        index = self.index_with(chunk("a", position=0), chunk("b", position=1))
        index.replace_document("law", [chunk("c")], [dense(1, 0)])
        assert self.all_chunks(index) == ["c"]

    def test_replacing_with_no_chunks_removes_the_document(self) -> None:
        index = self.index_with(chunk("a"), chunk("x", document_id="other"))
        index.replace_document("law", [], [])
        assert self.all_chunks(index) == ["x"]

    def test_replacing_a_document_leaves_other_documents(self) -> None:
        index = self.index_with(chunk("a"), chunk("x", document_id="other"))
        index.replace_document("law", [chunk("b")], [dense(1, 0)])
        assert self.all_chunks(index) == ["b", "x"]

    def test_a_refused_replacement_changes_nothing(self) -> None:
        index = self.index_with(chunk("a"))
        with pytest.raises(ValueError, match="belongs to 'other'"):
            index.replace_document("law", [chunk("b"), chunk("c", document_id="other")], [dense(1, 0)] * 2)
        assert self.all_chunks(index) == ["a"]

    def test_replacing_refuses_a_chunk_id_twice(self) -> None:
        with pytest.raises(ValueError, match="appears twice"):
            self.make_index(DENSE).replace_document("law", [chunk("a"), chunk("a")], [dense(1, 0)] * 2)

    def test_replacing_refuses_one_embedding_too_few(self) -> None:
        with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
            self.make_index(DENSE).replace_document("law", [chunk("a"), chunk("b")], [dense(1, 0)])

    def test_replacing_refuses_an_embedding_that_does_not_fit_the_space(self) -> None:
        wrong = Embedding(embedder_identity=EMBEDDER, vectors={"dense": DenseVector(values=(1.0, 0.0, 0.0))})
        with pytest.raises(ValueError, match="chunk 'a'"):
            self.make_index(DENSE).replace_document("law", [chunk("a")], [wrong])

    def test_replacing_refuses_an_embedding_from_another_embedder(self) -> None:
        other = Embedding(embedder_identity="other", vectors={"dense": DenseVector(values=(1.0, 0.0))})
        with pytest.raises(ValueError, match="chunk 'a': the embedding was made by 'other'"):
            self.make_index(DENSE).replace_document("law", [chunk("a")], [other])

    # Round trip ---------------------------------------------------------------------------

    def test_chunks_come_back_as_stored(self) -> None:
        stored = IndexedChunk(
            document_id="law",
            chunk_id="a",
            position=3,
            text="Điều 14. Phạm vi",
            embedded_text="Chương II\nĐiều 14. Phạm vi",
            headings=("Chương II", "Điều 14"),
            reference_keys=("chuong:2/dieu:14", "dieu:14"),
            attributes={
                "title": "Luật Đất đai",
                "year": 2024,
                "weight": 2.0,
                "in_force": True,
                "issued_on": date(2024, 1, 18),
                "knowledge_base_ids": ("kb-2", "kb-1"),
                "pages": (3, 1),
            },
        )
        index = self.make_index(DENSE)
        index.replace_document("law", [stored], [dense(1, 0)])
        (back,) = index.list_chunks(ChunkFilter(), limit=1)
        assert back == stored
        assert type(back.attributes["weight"]) is float
        assert type(back.attributes["issued_on"]) is date
        assert back.attributes["knowledge_base_ids"] == ("kb-2", "kb-1")

    # list_chunks --------------------------------------------------------------------------

    def test_list_chunks_is_in_document_order(self) -> None:
        index = self.index_with(
            chunk("b2", document_id="b", position=2),
            chunk("a1", document_id="a", position=1),
            chunk("b0", document_id="b", position=0),
            chunk("a0", document_id="a", position=0),
        )
        assert self.all_chunks(index) == ["a0", "a1", "b0", "b2"]

    def test_list_chunks_stops_at_the_limit_in_document_order(self) -> None:
        index = self.index_with(*(chunk(f"c{i}", position=i) for i in reversed(range(5))))
        assert ids(index.list_chunks(ChunkFilter(), limit=2)) == ["c0", "c1"]

    def test_list_chunks_refuses_a_limit_of_zero(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            self.make_index(DENSE).list_chunks(ChunkFilter(), limit=0)

    # Filters ------------------------------------------------------------------------------

    def test_document_ids_select_their_documents(self) -> None:
        index = self.index_with(chunk("a"), chunk("x", document_id="other"), chunk("y", document_id="third"))
        assert self.all_chunks(index, ChunkFilter(document_ids=("other", "third"))) == ["x", "y"]

    def test_chunk_keys_select_exactly_those_chunks(self) -> None:
        index = self.index_with(chunk("a", position=0), chunk("b", position=1), chunk("a", document_id="other"))
        keys = (ChunkKey(document_id="law", chunk_id="b"), ChunkKey(document_id="other", chunk_id="a"))
        assert self.all_chunks(index, ChunkFilter(chunk_keys=keys)) == ["b", "a"]

    def test_reference_keys_select_chunks_with_any_of_them(self) -> None:
        index = self.index_with(
            chunk("a", position=0, reference_keys=("dieu:14",)),
            chunk("b", position=1, reference_keys=("dieu:14/khoan:2", "khoan:2")),
            chunk("c", position=2, reference_keys=("dieu:15",)),
        )
        assert self.all_chunks(index, ChunkFilter(reference_keys=("dieu:14", "khoan:2"))) == ["a", "b"]

    def test_all_parts_of_a_filter_must_hold(self) -> None:
        index = self.index_with(
            chunk("a", position=0, reference_keys=("dieu:14",), role="main"),
            chunk("b", position=1, reference_keys=("dieu:14",), role="appendix"),
        )
        chunk_filter = ChunkFilter(reference_keys=("dieu:14",), attributes={"role": TextIn(values=("main",))})
        assert self.all_chunks(index, chunk_filter) == ["a"]

    def test_conditions_compare_kinds_exactly(self) -> None:
        index = self.index_with(
            chunk("int", position=0, value=1),
            chunk("bool", position=1, value=True),
            chunk("text", position=2, value="1"),
            chunk("float", position=3, value=1.0),
        )
        assert self.all_chunks(index, where(value=IntegerIn(values=(1,)))) == ["int"]
        assert self.all_chunks(index, where(value=BooleanEquals(value=True))) == ["bool"]
        assert self.all_chunks(index, where(value=TextIn(values=("1",)))) == ["text"]
        assert self.all_chunks(index, where(value=NumberBetween(at_least=1))) == ["int", "float"]

    def test_text_is_compared_as_stored(self) -> None:
        index = self.index_with(chunk("a", role="Main"))
        assert self.all_chunks(index, where(role=TextIn(values=("main",)))) == []

    def test_a_condition_on_a_list_holds_for_some_element(self) -> None:
        index = self.index_with(
            chunk("a", position=0, knowledge_base_ids=("kb-1", "kb-2")),
            chunk("b", position=1, knowledge_base_ids=("kb-3",)),
            chunk("c", position=2, years=(2020, 2024)),
        )
        assert self.all_chunks(index, where(knowledge_base_ids=TextIn(values=("kb-2", "kb-9")))) == ["a"]
        assert self.all_chunks(index, where(years=NumberBetween(at_least=2023))) == ["c"]

    def test_one_element_must_meet_every_bound(self) -> None:
        index = self.index_with(chunk("split", years=(2020, 2026)))
        assert self.all_chunks(index, where(years=NumberBetween(at_least=2023, below=2025))) == []

    def test_an_empty_list_never_matches(self) -> None:
        index = self.index_with(chunk("a", knowledge_base_ids=()))
        assert self.all_chunks(index, where(knowledge_base_ids=TextIn(values=("kb-1",)))) == []
        assert self.all_chunks(index, where(knowledge_base_ids=EveryElement(condition=TextIn(values=("kb-1",))))) == []

    def test_a_missing_attribute_never_matches(self) -> None:
        index = self.index_with(chunk("a", position=0, year=2024), chunk("b", position=1))
        assert self.all_chunks(index, where(year=NumberBetween(at_least=0))) == ["a"]
        assert self.all_chunks(index, where(typo=IntegerIn(values=(2024,)))) == []

    def test_none_of_selects_chunks_without_the_attribute(self) -> None:
        index = self.index_with(
            chunk("draft", position=0, status="draft"),
            chunk("final", position=1, status="final"),
            chunk("none", position=2),
        )
        assert self.all_chunks(index, where(status=NoneOf(conditions=(TextIn(values=("draft",)),)))) == [
            "final",
            "none",
        ]

    def test_every_element_needs_all_elements(self) -> None:
        index = self.index_with(
            chunk("inside", position=0, knowledge_base_ids=("kb-1", "kb-2")),
            chunk("partly", position=1, knowledge_base_ids=("kb-1", "kb-9")),
        )
        only_ours = EveryElement(condition=TextIn(values=("kb-1", "kb-2")))
        assert self.all_chunks(index, where(knowledge_base_ids=only_ours)) == ["inside"]

    def test_number_between_bounds_are_strict_or_inclusive_as_named(self) -> None:
        index = self.index_with(*(chunk(f"y{year}", position=year, year=year) for year in (2022, 2023, 2024)))
        assert self.all_chunks(index, where(year=NumberBetween(above=2022, at_most=2024))) == ["y2023", "y2024"]
        assert self.all_chunks(index, where(year=NumberBetween(at_least=2022, below=2024))) == ["y2022", "y2023"]

    def test_date_between_selects_a_month(self) -> None:
        index = self.index_with(
            chunk("jan", position=0, issued_on=date(2024, 1, 31)),
            chunk("feb", position=1, issued_on=date(2024, 2, 29)),
            chunk("mar", position=2, issued_on=date(2024, 3, 1)),
        )
        february = DateBetween(at_least=date(2024, 2, 1), below=date(2024, 3, 1))
        assert self.all_chunks(index, where(issued_on=february)) == ["feb"]

    def test_ranges_never_match_another_kind(self) -> None:
        index = self.index_with(
            chunk("date", position=0, value=date(2024, 1, 1)),
            chunk("bool", position=1, value=True),
            chunk("text", position=2, value="5"),
            chunk("number", position=3, value=5),
        )
        assert self.all_chunks(index, where(value=NumberBetween(at_least=0))) == ["number"]
        assert self.all_chunks(index, where(value=DateBetween(at_least=date(2000, 1, 1)))) == ["date"]

    def test_an_unsupported_condition_is_refused_before_anything_changes(self) -> None:
        index = self.index_with(chunk("a", title="Luật Đất đai"))
        hidden = where(title=NoneOf(conditions=(TextStartsWith(prefix="Luật"),)))
        supported = index.supported_conditions
        if supported is None:
            index.delete_chunks(hidden)  # evaluates any condition: the chunk starts with "Luật", so it stays
            assert self.all_chunks(index) == ["a"]
            return
        with pytest.raises(UnsupportedConditionError, match="TextStartsWith"):
            index.delete_chunks(hidden)
        assert self.all_chunks(index) == ["a"]

    # replace_generated (refusals) ---------------------------------------------------------

    def test_an_entry_must_lead_to_chunks_the_document_has(self) -> None:
        index = self.index_with_entries()
        with pytest.raises(ValueError, match="doesn't have: missing"):
            index.replace_generated("law", "questions-v1", [entry("q", "missing")], [dense(1, 0)])

    def test_an_entry_may_lead_to_chunks_with_different_attributes(self) -> None:
        index = self.index_with(chunk("a", position=0, role="main"), chunk("b", position=1, role="appendix"))
        index.replace_generated("law", "questions-v1", [entry("q", "a", "b")], [dense(1, 0)])

    def test_entries_refuse_an_embedding_from_another_embedder(self) -> None:
        other = Embedding(embedder_identity="other", vectors={"dense": DenseVector(values=(1.0, 0.0))})
        with pytest.raises(ValueError, match="entry 'q': the embedding was made by 'other'"):
            self.index_with_entries().replace_generated("law", "questions-v1", [entry("q", "a")], [other])

    def test_an_entry_must_belong_to_the_document_and_generator_given(self) -> None:
        with pytest.raises(ValueError, match="generator 'other'"):
            self.index_with_entries().replace_generated("law", "other", [entry("q", "a")], [dense(1, 0)])

    # delete_chunks ------------------------------------------------------------------------

    def test_deleting_removes_only_the_selected_chunks(self) -> None:
        index = self.index_with(chunk("a", position=0, version_id="v1"), chunk("b", position=1, version_id="v2"))
        index.delete_chunks(where(version_id=TextIn(values=("v1",))))
        assert self.all_chunks(index) == ["b"]

    def test_deleting_one_chunk_by_key(self) -> None:
        index = self.index_with(chunk("a", position=0), chunk("b", position=1))
        index.delete_chunks(ChunkFilter(chunk_keys=(ChunkKey(document_id="law", chunk_id="a"),)))
        assert self.all_chunks(index) == ["b"]

    def test_deleting_everything_is_refused(self) -> None:
        index = self.index_with(chunk("a"))
        with pytest.raises(ValueError, match="every chunk"):
            index.delete_chunks(ChunkFilter())
        assert self.all_chunks(index) == ["a"]

    # update_attributes --------------------------------------------------------------------

    def test_updating_sets_and_removes_only_the_named_attributes(self) -> None:
        index = self.index_with(
            chunk("a", position=0, set_id="s1", title="Old", note="x"), chunk("b", position=1, set_id="s2")
        )
        index.update_attributes(where(set_id=TextIn(values=("s1",))), {"title": "New", "note": None, "year": 2024})
        a, b = index.list_chunks(ChunkFilter(), limit=2)
        assert a == chunk("a", position=0, set_id="s1", title="New", year=2024)
        assert b == chunk("b", position=1, set_id="s2")

    def test_updating_part_of_an_entrys_chunks_is_allowed(self) -> None:
        index = self.make_index(DENSE)
        chunks = [chunk("a", position=0, reference_keys=("dieu:1",)), chunk("b", position=1)]
        index.replace_document("law", chunks, [dense(0, 1), dense(0, 1)])
        index.replace_generated("law", "questions-v1", [entry("q", "a", "b")], [dense(1, 0)])
        index.update_attributes(ChunkFilter(reference_keys=("dieu:1",)), {"role": "main"})
        assert [dict(c.attributes) for c in index.list_chunks(ChunkFilter(), limit=2)] == [{"role": "main"}, {}]

    def test_updating_everything_is_refused(self) -> None:
        with pytest.raises(ValueError, match="every chunk"):
            self.index_with(chunk("a")).update_attributes(ChunkFilter(), {"title": "x"})

    def test_an_update_needs_changes(self) -> None:
        with pytest.raises(ValueError, match="no attribute changes"):
            self.index_with(chunk("a")).update_attributes(ChunkFilter(document_ids=("law",)), {})


class ChunkIndexSearchContract(_ContractHelpers):
    """Search, including how generated entries find chunks."""

    # Ranking ------------------------------------------------------------------------------

    def test_search_ranks_by_cosine_similarity(self) -> None:
        index = self.make_index(DENSE)
        chunks = [chunk("far", position=0), chunk("near", position=1), chunk("same", position=2)]
        index.replace_document("law", chunks, [dense(0, 1), dense(1, 1), dense(2, 0)])
        hits = index.search(SearchQuery(embedding=dense(1, 0)))
        assert hit_ids(hits) == ["same", "near", "far"]
        assert [hit.score for hit in hits] == pytest.approx([1.0, 2**-0.5, 0.0])

    def test_a_zero_vector_scores_zero(self) -> None:
        """An empty text can embed to all zeros; it must rank last, not fail the search."""
        index = self.make_index(DENSE)
        index.replace_document("law", [chunk("blank", position=0), chunk("a", position=1)], [dense(0, 0), dense(1, 0)])
        hits = index.search(SearchQuery(embedding=dense(1, 0)))
        assert hit_ids(hits) == ["a", "blank"]
        assert hits[1].score == pytest.approx(0.0)

    def test_search_stops_at_the_limit(self) -> None:
        index = self.index_with(*(chunk(f"c{i}", position=i) for i in range(5)))
        assert len(index.search(SearchQuery(embedding=dense(1, 0), limit=3))) == 3

    def test_search_only_returns_selected_chunks(self) -> None:
        index = self.index_with(chunk("a", position=0, role="main"), chunk("b", position=1, role="appendix"))
        appendix = where(role=TextIn(values=("appendix",)))
        assert hit_ids(index.search(SearchQuery(embedding=dense(1, 0), chunk_filter=appendix))) == ["b"]

    def test_a_filtered_search_still_fills_the_limit(self) -> None:
        """Filters apply during the search, not to its top results afterwards."""
        index = self.make_index(DENSE)
        near = [chunk(f"near{i}", position=i, role="appendix") for i in range(20)]
        far = [chunk(f"far{i}", position=20 + i, role="main") for i in range(5)]
        index.replace_document("law", near + far, [dense(1, 0)] * 20 + [dense(0, 1)] * 5)
        main = where(role=TextIn(values=("main",)))
        assert sorted(hit_ids(index.search(SearchQuery(embedding=dense(1, 0), chunk_filter=main, limit=5)))) == [
            f"far{i}" for i in range(5)
        ]

    # Refusals -----------------------------------------------------------------------------

    def test_search_refuses_a_query_that_does_not_fit_the_space(self) -> None:
        with pytest.raises(ValueError, match="query"):
            self.make_index(DENSE).search(SearchQuery(embedding=hybrid(1, 0, {1: 1.0})))

    def test_search_refuses_a_query_from_another_embedder(self) -> None:
        other = Embedding(embedder_identity="other", vectors={"dense": DenseVector(values=(1.0, 0.0))})
        with pytest.raises(ValueError, match="query: the embedding was made by 'other'"):
            self.index_with(chunk("a")).search(SearchQuery(embedding=other))

    def test_search_refuses_an_unknown_vector_name(self) -> None:
        with pytest.raises(ValueError, match="no vectors named sparse"):
            self.make_index(DENSE).search(SearchQuery(embedding=dense(1, 0), vector_names=("sparse",)))

    # Generated entries --------------------------------------------------------------------

    def test_an_entry_finds_its_chunk(self) -> None:
        first, _ = self.index_with_entries().search(SearchQuery(embedding=dense(1, 0)))
        assert first.chunk == chunk("a", position=0)
        assert first.score == pytest.approx(1.0)
        assert first.found_through == (entry("q", "a"),)

    def test_a_chunk_found_by_its_own_text_has_no_entries_to_show(self) -> None:
        first, _ = self.index_with_entries().search(SearchQuery(embedding=dense(0, 1)))
        assert first.found_through == ()

    def test_an_entry_about_several_chunks_finds_each_once(self) -> None:
        index = self.make_index(DENSE)
        index.replace_document("law", [chunk("a", position=0), chunk("b", position=1)], [dense(0, 1), dense(0, 1)])
        index.replace_generated(
            "law", "questions-v1", [entry("q1", "a", "b"), entry("q2", "a")], [dense(1, 0), dense(1, 0)]
        )
        assert sorted(hit_ids(index.search(SearchQuery(embedding=dense(1, 0))))) == ["a", "b"]

    def test_entries_can_be_left_out_of_a_search(self) -> None:
        hit_a, hit_b = self.index_with_entries().search(SearchQuery(embedding=dense(1, 0), generated_entries="none"))
        assert hit_a.score == pytest.approx(0.0)
        assert hit_a.found_through == hit_b.found_through == ()

    def test_a_search_can_use_only_some_generators(self) -> None:
        index = self.index_with_entries()
        index.replace_generated("law", "summaries-v1", [entry("s", "b", generator="summaries-v1")], [dense(1, 0)])
        hits = index.search(SearchQuery(embedding=dense(1, 0), generated_entries=("summaries-v1",)))
        assert hits[0].found_through == (entry("s", "b", generator="summaries-v1"),)
        assert hits[1].found_through == ()

    def test_replacing_entries_keeps_other_generators(self) -> None:
        index = self.index_with_entries()
        index.replace_generated("law", "summaries-v1", [entry("s", "b", generator="summaries-v1")], [dense(1, 0)])
        index.replace_generated("law", "questions-v1", [], [])
        hits = index.search(SearchQuery(embedding=dense(1, 0)))
        assert [hit.found_through for hit in hits] == [(entry("s", "b", generator="summaries-v1"),), ()]

    def test_replacing_a_document_removes_its_entries(self) -> None:
        index = self.index_with_entries()
        index.replace_document("law", [chunk("a", position=0), chunk("b", position=1)], [dense(0, 1), dense(0, 1)])
        assert [hit.found_through for hit in index.search(SearchQuery(embedding=dense(1, 0)))] == [(), ()]

    def test_entries_follow_their_chunks_through_filters(self) -> None:
        index = self.make_index(DENSE)
        index.replace_document(
            "law",
            [chunk("a", position=0, reference_keys=("dieu:1",)), chunk("b", position=1, reference_keys=("dieu:2",))],
            [dense(0, 1), dense(0, 1)],
        )
        index.replace_generated("law", "questions-v1", [entry("q", "a")], [dense(1, 0)])
        hits = index.search(SearchQuery(embedding=dense(1, 0), chunk_filter=ChunkFilter(reference_keys=("dieu:2",))))
        assert [(hit.chunk.chunk_id, hit.found_through) for hit in hits] == [("b", ())]

    def test_entries_follow_their_chunks_after_an_update(self) -> None:
        index = self.index_with_entries()
        index.update_attributes(ChunkFilter(document_ids=("law",)), {"knowledge_base_ids": ("kb-1",)})
        in_kb = where(knowledge_base_ids=TextIn(values=("kb-1",)))
        hits = index.search(SearchQuery(embedding=dense(1, 0), chunk_filter=in_kb))
        assert hits[0].found_through == (entry("q", "a"),)

    def test_deleting_a_chunk_leaves_its_entries_leading_to_the_rest(self) -> None:
        index = self.make_index(DENSE)
        index.replace_document("law", [chunk("a", position=0), chunk("b", position=1)], [dense(0, 1), dense(0, 1)])
        index.replace_generated("law", "questions-v1", [entry("q", "a", "b")], [dense(1, 0)])
        index.delete_chunks(ChunkFilter(chunk_keys=(ChunkKey(document_id="law", chunk_id="b"),)))
        (hit,) = index.search(SearchQuery(embedding=dense(1, 0)))
        assert hit.chunk.chunk_id == "a"
        assert hit.found_through == (entry("q", "a"),)

    def test_an_entry_left_with_no_chunks_is_removed(self) -> None:
        index = self.index_with_entries()
        index.delete_chunks(ChunkFilter(chunk_keys=(ChunkKey(document_id="law", chunk_id="a"),)))
        index.replace_document("law", [chunk("a", position=0), chunk("b", position=1)], [dense(0, 1), dense(0, 1)])
        assert [hit.found_through for hit in index.search(SearchQuery(embedding=dense(1, 0)))] == [(), ()]

    # Several vectors ----------------------------------------------------------------------

    def test_a_sparse_vector_only_finds_shared_terms(self) -> None:
        index = self.make_index(HYBRID)
        index.replace_document(
            "law",
            [chunk("shares", position=0), chunk("apart", position=1)],
            [hybrid(1, 0, {7: 1.0}), hybrid(1, 0, {8: 1.0})],
        )
        query = SearchQuery(embedding=hybrid(1, 0, {7: 2.0}), vector_names=("sparse",))
        assert hit_ids(index.search(query)) == ["shares"]

    def test_several_vectors_are_fused(self) -> None:
        index = self.make_index(HYBRID)
        names = ["dense_best", "second_on_both", "filler", "sparse_best"]
        index.replace_document(
            "law",
            [chunk(name, position=i) for i, name in enumerate(names)],
            [hybrid(1, 0, {1: 1.0}), hybrid(0.9, 0.1, {7: 0.9}), hybrid(1, 1, {9: 1.0}), hybrid(0, 1, {7: 1.0})],
        )
        query = hybrid(1, 0, {7: 1.0})
        assert hit_ids(index.search(SearchQuery(embedding=query)))[0] == "second_on_both"
        assert hit_ids(index.search(SearchQuery(embedding=query, vector_names=("dense",))))[0] == "dense_best"
        assert hit_ids(index.search(SearchQuery(embedding=query, vector_names=("sparse",))))[0] == "sparse_best"

    def test_fused_hits_list_entries_from_every_vector(self) -> None:
        index = self.make_index(HYBRID)
        index.replace_document("law", [chunk("a")], [hybrid(0, 1, {1: 0.1})])
        index.replace_generated(
            "law",
            "questions-v1",
            [entry("by_dense", "a"), entry("by_sparse", "a")],
            [hybrid(1, 0, {2: 0.1}), hybrid(0, 1, {1: 5.0})],
        )
        (hit,) = index.search(SearchQuery(embedding=hybrid(1, 0, {1: 1.0})))
        assert sorted(e.entry_id for e in hit.found_through) == ["by_dense", "by_sparse"]

    def test_idf_weighs_rare_terms_more(self) -> None:
        index = self.make_index(SPARSE_IDF)
        index.replace_document(
            "law",
            [chunk("rare", position=0), *(chunk(f"common{i}", position=i + 1) for i in range(4))],
            [sparse({2: 1.0}), *(sparse({1: 1.0}) for _ in range(4))],
        )
        hits = index.search(SearchQuery(embedding=sparse({1: 1.0, 2: 1.0})))
        assert hits[0].chunk.chunk_id == "rare"
        assert hits[0].score > hits[1].score