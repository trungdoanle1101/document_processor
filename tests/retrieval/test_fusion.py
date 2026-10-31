"""fuse is shared by every ChunkIndex, so a bug here would hide from the contract suite: expected values are by hand."""

import pytest

from document_processor.retrieval._fusion import BestMatch, fuse
from document_processor.retrieval.index import ChunkHit, GeneratedEntry, IndexedChunk


def chunk(chunk_id: str) -> IndexedChunk:
    return IndexedChunk(document_id="law", chunk_id=chunk_id, position=0, text=chunk_id, embedded_text=chunk_id)


def entry(entry_id: str, chunk_id: str) -> GeneratedEntry:
    return GeneratedEntry(document_id="law", generator="g", entry_id=entry_id, chunk_ids=(chunk_id,), text="?")


def own(chunk_id: str, score: float = 1.0) -> BestMatch:
    return BestMatch(chunk=chunk(chunk_id), score=score, entry=None)


def through(entry_id: str, chunk_id: str, score: float = 1.0) -> BestMatch:
    return BestMatch(chunk=chunk(chunk_id), score=score, entry=entry(entry_id, chunk_id))


def ids(hits: list[ChunkHit]) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


def rrf(*ranks: int) -> float:
    return sum(1 / (60 + rank) for rank in ranks)


# One ranking ------------------------------------------------------------------------------


def test_one_ranking_keeps_its_scores_best_first() -> None:
    hits = fuse([[own("b", 0.4), own("a", 0.9), own("c", 0.1)]], limit=10)
    assert ids(hits) == ["a", "b", "c"]
    assert [hit.score for hit in hits] == [0.9, 0.4, 0.1]


def test_one_ranking_stops_at_the_limit_after_sorting() -> None:
    assert ids(fuse([[own("low", 0.1), own("high", 0.9), own("mid", 0.5)]], limit=2)) == ["high", "mid"]


def test_one_ranking_names_the_entry_behind_a_match() -> None:
    hit_a, hit_b = fuse([[through("q", "a", 0.9), own("b", 0.5)]], limit=10)
    assert hit_a.found_through == (entry("q", "a"),)
    assert hit_b.found_through == ()


def test_nothing_matched_gives_no_hits() -> None:
    assert fuse([[]], limit=10) == []
    assert fuse([[], []], limit=10) == []


# Several rankings -------------------------------------------------------------------------


def test_several_rankings_score_by_reciprocal_rank() -> None:
    hits = fuse([[own("a"), own("b")], [own("b"), own("c")]], limit=10)
    scores = {hit.chunk.chunk_id: hit.score for hit in hits}
    assert scores == pytest.approx({"a": rrf(1), "b": rrf(2, 1), "c": rrf(2)})


def test_fusion_ignores_the_scores_and_uses_the_ranks() -> None:
    """A sparse score of 50 must not drown a cosine of 0.9: only positions count."""
    hits = fuse([[own("dense_best", 0.9), own("both", 0.8)], [own("both", 50.0)]], limit=10)
    assert ids(hits) == ["both", "dense_best"]


def test_a_chunk_found_by_several_rankings_appears_once() -> None:
    hits = fuse([[own("a"), own("b")], [own("a")], [own("a")]], limit=10)
    assert ids(hits) == ["a", "b"]
    assert hits[0].score == pytest.approx(rrf(1, 1, 1))


def test_fused_hits_stop_at_the_limit_after_sorting() -> None:
    hits = fuse([[own("a"), own("b"), own("c")], [own("c"), own("b"), own("a")], [own("b")]], limit=1)
    assert ids(hits) == ["b"]


# found_through when fused -----------------------------------------------------------------


def test_found_through_lists_entries_from_every_ranking_best_rank_first() -> None:
    first = [own("x"), own("y"), through("by_first", "a")]  # a at rank 3
    second = [through("by_second", "a")]  # a at rank 1
    (hit_a,) = [hit for hit in fuse([first, second], limit=10) if hit.chunk.chunk_id == "a"]
    assert [e.entry_id for e in hit_a.found_through] == ["by_second", "by_first"]


def test_an_entry_behind_several_rankings_is_listed_once() -> None:
    (hit,) = fuse([[through("q", "a")], [through("q", "a")]], limit=10)
    assert hit.found_through == (entry("q", "a"),)


def test_own_text_on_one_ranking_and_an_entry_on_another_lists_only_the_entry() -> None:
    (hit,) = fuse([[own("a")], [through("q", "a")]], limit=10)
    assert hit.found_through == (entry("q", "a"),)