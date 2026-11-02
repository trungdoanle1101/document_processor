"""evaluation: matching answers at three levels, scoring a search, loading questions.

Expected numbers are worked out by hand, so a bug in the scoring can't hide behind itself.
"""

import json
from functools import partial
from pathlib import Path

import pytest

from document_processor.retrieval.evaluation import (
    EvaluationQuestion,
    EvaluationReport,
    ExpectedAnswer,
    MatchLevel,
    QuestionResult,
    evaluate,
    format_summary,
    load_questions,
)
from document_processor.retrieval.in_memory_index import InMemoryChunkIndex
from document_processor.retrieval.index import IndexedChunk
from document_processor.retrieval.search import RetrievedChunk, search_chunks
from tests.retrieval.stub_embedder import StubEmbedder

LAW = "luat-dat-dai"


def chunk(
    chunk_id: str,
    text: str = "",
    *,
    document_id: str = LAW,
    keys: tuple[str, ...] = (),
) -> IndexedChunk:
    return IndexedChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        position=0,
        text=text or f"text of {chunk_id}",
        embedded_text=text or f"text of {chunk_id}",
        reference_keys=keys,
    )


def retrieved(
    c: IndexedChunk, score: float = 1.0, rerank: float | None = None
) -> RetrievedChunk:
    return RetrievedChunk(chunk=c, index_score=score, rerank_score=rerank)


def question(
    id: str,
    *answers: ExpectedAnswer,
    tags: tuple[str, ...] = (),
) -> EvaluationQuestion:
    return EvaluationQuestion(
        id=id, question=f"question {id}?", answers=answers, tags=tags
    )


ARTICLE_9 = ExpectedAnswer(
    document_id=LAW, reference_key="dieu:9", quote="tối đa 07 ngày"
)

STAGING = chunk(
    "staging",
    "Dữ liệu vùng này được lưu trữ ngắn hạn (tối  đa 07 NGÀY).",
    keys=("dieu:9",),
)
RAW = chunk("raw", "Phân vùng dữ liệu thô (RAW).", keys=("dieu:9",))
ARTICLE_10 = chunk("art10", "tối đa 07 ngày", keys=("dieu:10",))
OTHER_LAW = chunk(
    "other", "tối đa 07 ngày", document_id="nghi-dinh-13", keys=("dieu:9",)
)


# ExpectedAnswer -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("c", "document", "section", "passage"),
    [
        (STAGING, True, True, True),
        (RAW, True, True, False),
        (ARTICLE_10, True, False, False),
        (OTHER_LAW, False, False, False),
    ],
    ids=["exact passage", "right article", "right document", "other document"],
)
def test_each_level_is_stricter_than_the_one_before(
    c: IndexedChunk, document: bool, section: bool, passage: bool
) -> None:
    assert [
        ARTICLE_9.match(c, level) for level in ("document", "section", "passage")
    ] == [
        document,
        section,
        passage,
    ]


def test_quotes_match_whatever_the_case_and_spacing() -> None:
    """STAGING writes "(tối  đa 07 NGÀY)": two spaces and capitals."""
    assert ARTICLE_9.match(STAGING, "passage")


def test_quotes_are_looked_for_in_the_chunks_own_text_not_the_headings() -> None:
    c = IndexedChunk(
        document_id=LAW,
        chunk_id="c",
        position=0,
        text="Dữ liệu ...",
        embedded_text="Điều 9. tối đa 07 ngày\nDữ liệu ...",
        reference_keys=("dieu:9",),
    )
    assert not ARTICLE_9.match(c, "passage")


def test_a_quote_without_a_key_matches_anywhere_in_the_document() -> None:
    front_matter = ExpectedAnswer(document_id=LAW, quote="Số: 12/QĐ")
    assert front_matter.match(chunk("c0", "Số: 12/QĐ-BTC"), "passage")
    assert not front_matter.can_match("section")


def test_an_answer_says_which_levels_it_can_be_matched_at() -> None:
    assert [
        ExpectedAnswer(document_id=LAW).can_match(level)
        for level in ("document", "section", "passage")
    ] == [True, False, False]
    assert [
        ARTICLE_9.can_match(level) for level in ("document", "section", "passage")
    ] == [
        True,
        True,
        True,
    ]


def test_matching_at_a_level_the_answer_cannot_be_judged_at_is_refused() -> None:
    with pytest.raises(ValueError, match="section level"):
        ExpectedAnswer(document_id=LAW).match(STAGING, "section")


@pytest.mark.parametrize(
    "fields",
    [
        {"document_id": " "},
        {"document_id": LAW, "reference_key": ""},
        {"document_id": LAW, "quote": " "},
    ],
    ids=["blank document", "blank key", "blank quote"],
)
def test_an_answer_refuses_blank_fields(fields: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="blank"):
        ExpectedAnswer(**fields)


# EvaluationQuestion -------------------------------------------------------------------------


def test_a_question_is_judged_only_at_levels_all_its_answers_allow() -> None:
    keyed = ExpectedAnswer(document_id=LAW, reference_key="dieu:9")
    assert question("q", ARTICLE_9).levels() == ("document", "section", "passage")
    assert question("q", ARTICLE_9, keyed).levels() == ("document", "section")
    assert question("q", ExpectedAnswer(document_id=LAW)).levels() == ("document",)


def test_an_unanswerable_question_is_judged_at_no_level() -> None:
    q = question("q")
    assert not q.answerable
    assert q.levels() == ()


@pytest.mark.parametrize(("id", "text"), [(" ", "Ai?"), ("q", "  ")])
def test_a_question_needs_an_id_and_text(id: str, text: str) -> None:
    with pytest.raises(ValueError):
        EvaluationQuestion(id=id, question=text, answers=())


# evaluate -----------------------------------------------------------------------------------


class FixedSearch:
    """Returns the same hits for each question, from a table; remembers every call."""

    def __init__(self, hits: dict[str, list[RetrievedChunk]]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def __call__(self, question: str, *, limit: int) -> list[RetrievedChunk]:
        self.calls.append((question, limit))
        return self.hits[question]


def test_each_level_gets_the_rank_of_its_first_match() -> None:
    search = FixedSearch(
        {"question q?": [retrieved(ARTICLE_10), retrieved(RAW), retrieved(STAGING)]}
    )
    (result,) = evaluate([question("q", ARTICLE_9)], search).results
    assert dict(result.ranks) == {"document": 1, "section": 2, "passage": 3}


def test_any_answer_counts() -> None:
    other = ExpectedAnswer(
        document_id=LAW, reference_key="dieu:10", quote="tối đa 07 ngày"
    )
    search = FixedSearch({"question q?": [retrieved(ARTICLE_10), retrieved(STAGING)]})
    (result,) = evaluate([question("q", ARTICLE_9, other)], search).results
    assert result.ranks["passage"] == 1


def test_a_match_below_the_depth_is_a_miss() -> None:
    search = FixedSearch({"question q?": [retrieved(RAW), retrieved(STAGING)]})
    (result,) = evaluate([question("q", ARTICLE_9)], search, depth=1).results
    assert dict(result.ranks) == {"document": 1, "section": 1, "passage": None}
    assert len(result.hits) == 1


def test_the_search_is_asked_for_the_depth() -> None:
    search = FixedSearch({"question q?": []})
    evaluate([question("q", ARTICLE_9)], search, depth=7)
    assert search.calls == [("question q?", 7)]


def test_unanswerable_questions_are_searched_but_not_judged() -> None:
    search = FixedSearch({"question q?": [retrieved(STAGING, 0.4)]})
    (result,) = evaluate([question("q")], search).results
    assert dict(result.ranks) == {}
    assert len(result.hits) == 1


def test_evaluate_refuses_repeated_ids_and_a_bad_depth() -> None:
    search = FixedSearch({"question q?": []})
    with pytest.raises(ValueError, match="appears twice"):
        evaluate([question("q", ARTICLE_9), question("q", ARTICLE_9)], search)
    with pytest.raises(ValueError, match="depth must be positive"):
        evaluate([question("q", ARTICLE_9)], search, depth=0)


# EvaluationReport ---------------------------------------------------------------------------


def result(id: str, tags: tuple[str, ...] = (), **ranks: int | None) -> QuestionResult:
    levels: dict[MatchLevel, int | None] = {}
    for level in ("document", "section", "passage"):
        if level in ranks:
            levels[level] = ranks[level]
    return QuestionResult(
        question=question(id, ARTICLE_9, tags=tags), hits=(), ranks=levels
    )


# section ranks 1, 3, 7, missed; passage judged on the first three only
REPORT = EvaluationReport(
    depth=10,
    results=(
        result("a", ("lookup",), document=1, section=1, passage=2),
        result("b", ("lookup",), document=1, section=3, passage=None),
        result("c", ("paraphrase",), document=2, section=7, passage=9),
        result("d", ("paraphrase",), document=None, section=None),
    ),
)


def test_hit_rate_is_the_share_found_in_the_top_k() -> None:
    assert REPORT.hit_rate(1, level="section") == 1 / 4
    assert REPORT.hit_rate(5, level="section") == 2 / 4
    assert REPORT.hit_rate(10, level="section") == 3 / 4


def test_mrr_averages_one_over_the_first_rank() -> None:
    assert REPORT.mrr(level="section") == pytest.approx((1 + 1 / 3 + 1 / 7 + 0) / 4)


def test_each_level_counts_only_the_questions_judged_at_it() -> None:
    assert REPORT.judged_count(level="passage") == 3
    assert REPORT.hit_rate(10, level="passage") == 2 / 3


def test_metrics_can_be_limited_to_a_tag() -> None:
    assert REPORT.hit_rate(1, level="section", tag="lookup") == 1 / 2
    assert REPORT.mrr(level="section", tag="paraphrase") == pytest.approx(
        (1 / 7 + 0) / 2
    )


def test_a_k_beyond_the_depth_is_refused() -> None:
    with pytest.raises(ValueError, match="depth 10"):
        REPORT.hit_rate(20, level="section")


def test_a_level_with_no_judged_questions_is_refused() -> None:
    with pytest.raises(ValueError, match="passage level with tag 'missing'"):
        REPORT.mrr(level="passage", tag="missing")


def test_misses_are_judged_questions_with_no_match() -> None:
    assert [r.question.id for r in REPORT.misses(level="section")] == ["d"]
    assert [r.question.id for r in REPORT.misses(level="passage")] == ["b"]


def test_the_summary_has_every_level_overall_then_by_tag() -> None:
    rows = REPORT.summarize()
    assert [(row.tag, row.level, row.judged) for row in rows] == [
        (None, "document", 4),
        (None, "section", 4),
        (None, "passage", 3),
        ("lookup", "document", 2),
        ("lookup", "section", 2),
        ("lookup", "passage", 2),
        ("paraphrase", "document", 2),
        ("paraphrase", "section", 2),
        ("paraphrase", "passage", 1),
    ]
    assert dict(rows[1].hit_rate) == {1: 1 / 4, 3: 2 / 4, 5: 2 / 4, 10: 3 / 4}


def test_the_summary_leaves_out_ks_beyond_the_depth() -> None:
    shallow = EvaluationReport(depth=3, results=REPORT.results)
    assert list(shallow.summarize()[0].hit_rate) == [1, 3]


def test_top_scores_use_the_rerank_score_when_there_is_one() -> None:
    report = EvaluationReport(
        depth=10,
        results=(
            QuestionResult(
                question=question("a", ARTICLE_9), hits=(retrieved(STAGING, 0.8),)
            ),
            QuestionResult(
                question=question("b", ARTICLE_9), hits=(retrieved(STAGING, 0.8, 5.0),)
            ),
            QuestionResult(question=question("c"), hits=(retrieved(RAW, 0.3),)),
            QuestionResult(question=question("d"), hits=()),
        ),
    )
    assert report.top_scores(answerable=True) == [0.8, 5.0]
    assert report.top_scores(answerable=False) == [0.3]


def test_the_summary_prints_as_a_table() -> None:
    lines = format_summary(REPORT).splitlines()
    assert lines[0].split() == [
        "level",
        "tag",
        "judged",
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "MRR@10",
    ]
    assert lines[2].split() == [
        "section",
        "all",
        "4",
        "0.25",
        "0.50",
        "0.50",
        "0.75",
        "0.37",
    ]


# load_questions -----------------------------------------------------------------------------


def write_lines(tmp_path: Path, *lines: object) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(
            line if isinstance(line, str) else json.dumps(line, ensure_ascii=False)
            for line in lines
        ),
        encoding="utf-8",
    )
    return path


GOOD = {
    "id": "kt-17",
    "question": "Dữ liệu ở vùng Staging được lưu tối đa bao nhiêu ngày?",
    "answers": [
        {"document_id": LAW, "reference_key": "dieu:9", "quote": "tối đa 07 ngày"}
    ],
    "location": "Điều 9 khoản 2",
    "tags": ["lookup"],
}


def test_questions_load_from_json_lines(tmp_path: Path) -> None:
    unanswerable = {
        "id": "kt-35",
        "question": "Log lưu bao lâu?",
        "answers": [],
        "tags": ["unanswerable"],
    }
    (first, second) = load_questions(write_lines(tmp_path, GOOD, "", unanswerable))
    assert first == EvaluationQuestion(
        id="kt-17",
        question=GOOD["question"],  # type: ignore[arg-type]
        answers=(ARTICLE_9,),
        tags=("lookup",),
        location="Điều 9 khoản 2",
    )
    assert not second.answerable


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("{not json", "Expecting property name"),
        ({**GOOD, "anwsers": []}, "unknown question field"),
        (
            {**GOOD, "answers": [{"document_id": LAW, "qoute": "x"}]},
            "unknown answer field",
        ),
        ({k: v for k, v in GOOD.items() if k != "question"}, "question"),
        ({**GOOD, "tags": "lookup"}, "tags must be a list"),
        ({**GOOD, "answers": [{"document_id": 9}]}, "document_id must be a string"),
        ([1, 2], "JSON object"),
    ],
    ids=[
        "bad json",
        "misspelled field",
        "misspelled answer field",
        "missing field",
        "tags",
        "wrong type",
        "not an object",
    ],
)
def test_a_bad_line_is_refused_with_its_line_number(
    tmp_path: Path, line: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message) as error:
        load_questions(write_lines(tmp_path, {**GOOD, "id": "first"}, line))
    assert ":2:" in str(error.value)


def test_repeated_ids_are_refused_when_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'kt-17' appears twice"):
        load_questions(write_lines(tmp_path, GOOD, GOOD))


# Through a real search ----------------------------------------------------------------------


def test_search_chunks_can_be_evaluated_directly() -> None:
    embedder = StubEmbedder()
    index = InMemoryChunkIndex(
        embedder_identity=embedder.identity, vector_specs=embedder.vector_specs
    )
    chunks = [
        chunk("c0", "giao đất nông nghiệp cho hộ gia đình", keys=("dieu:5",)),
        chunk("c1", "thuế thu nhập cá nhân", keys=("dieu:6",)),
    ]
    index.replace_document(
        LAW, chunks, embedder.encode_passages([c.embedded_text for c in chunks])
    )

    q = EvaluationQuestion(
        id="q",
        question="giao đất nông nghiệp",
        answers=(
            ExpectedAnswer(
                document_id=LAW, reference_key="dieu:5", quote="hộ gia đình"
            ),
        ),
    )
    report = evaluate(
        [q], partial(search_chunks, embedder=embedder, index=index), depth=2
    )
    assert dict(report.results[0].ranks) == {"document": 1, "section": 1, "passage": 1}
