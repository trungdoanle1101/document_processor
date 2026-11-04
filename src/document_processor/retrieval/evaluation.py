"""Measure retrieval against questions whose answers we know the location of.

A retrieved chunk counts as finding an answer at one of three levels, each stricter
than the one before:
- document: it comes from the answer's document
- section: it also carries the answer's reference key ("dieu:9")
- passage: its text also contains the answer's quote
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from document_processor.retrieval.index import IndexedChunk
from document_processor.retrieval.search import RetrievedChunk, Search

type MatchLevel = Literal["document", "section", "passage"]
MATCH_LEVELS: tuple[MatchLevel, ...] = ("document", "section", "passage")

_DEFAULT_KS = (1, 3, 5, 10)
_QUESTION_FIELDS = frozenset({"id", "question", "answers", "tags", "location"})
_ANSWER_FIELDS = frozenset({"document_id", "reference_key", "quote"})


@dataclass(frozen=True, kw_only=True)
class ExpectedAnswer:
    document_id: str
    reference_key: str | None = None  # where: "dieu:9"; needed for the section level
    quote: str | None = (
        None  # what: a short verbatim phrase; needed for the passage level
    )

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if self.reference_key is not None and not self.reference_key.strip():
            raise ValueError("reference_key must not be blank; use None for none")
        if self.quote is not None and not self.quote.strip():
            raise ValueError("quote must not be blank; use None for none")

    def can_match(self, level: MatchLevel) -> bool:
        match level:
            case "document":
                return True
            case "section":
                return self.reference_key is not None
            case "passage":
                return self.quote is not None

    def match(self, chunk: IndexedChunk, level: MatchLevel) -> bool:
        """Whether this chunk finds the answer at this level."""
        if not self.can_match(level):
            raise ValueError(f"this answer can't be matched at the {level} level")
        if chunk.document_id != self.document_id:
            return False
        # the key also narrows the passage level, when there is one
        if (
            level != "document"
            and self.reference_key is not None
            and self.reference_key not in chunk.reference_keys
        ):
            return False
        if level == "passage":
            assert self.quote is not None
            return _normalize(self.quote) in _normalize(chunk.text)
        return True


@dataclass(frozen=True, kw_only=True)
class EvaluationQuestion:
    id: str
    question: str
    answers: tuple[
        ExpectedAnswer, ...
    ]  # any one counts; () means the documents can't answer it
    tags: tuple[str, ...] = ()
    location: str | None = None  # for people reading misses, e.g. "Điều 9 khoản 2"

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("question id must not be blank")
        if not self.question.strip():
            raise ValueError(f"question {self.id!r} has no text")

    @property
    def answerable(self) -> bool:
        return bool(self.answers)

    def levels(self) -> tuple[MatchLevel, ...]:
        """The levels every one of its answers can be matched at; () if unanswerable."""
        if not self.answers:
            return ()
        return tuple(
            level
            for level in MATCH_LEVELS
            if all(answer.can_match(level) for answer in self.answers)
        )


@dataclass(frozen=True, kw_only=True)
class QuestionResult:
    question: EvaluationQuestion
    hits: tuple[RetrievedChunk, ...]
    # first matching rank (from 1) for each level the question is judged at;
    # None: not in the top depth
    ranks: Mapping[MatchLevel, int | None] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", MappingProxyType(dict(self.ranks)))

    def found(self, *, level: MatchLevel, k: int) -> bool:
        rank = self.ranks[level]
        return rank is not None and rank <= k


@dataclass(frozen=True, kw_only=True)
class MetricRow:
    level: MatchLevel
    tag: str | None  # None: all questions
    judged: int  # how many questions these numbers are over
    hit_rate: Mapping[int, float]  # k -> share found in the top k
    mrr: float  # MRR@depth


@dataclass(frozen=True, kw_only=True)
class EvaluationReport:
    depth: int  # how many hits each question saw; the "@10" in MRR@10
    results: tuple[QuestionResult, ...]

    def hit_rate(self, k: int, *, level: MatchLevel, tag: str | None = None) -> float:
        """Share of the judged questions with a match in the top k."""
        if not 1 <= k <= self.depth:
            raise ValueError(f"k must be between 1 and depth {self.depth}, got {k}")
        judged = self._judged(level, tag)
        return sum(r.found(level=level, k=k) for r in judged) / len(judged)

    def mrr(self, *, level: MatchLevel, tag: str | None = None) -> float:
        """Mean of 1/rank of the first match; 0 when not in the top depth."""
        judged = self._judged(level, tag)
        return sum(_reciprocal(r.ranks[level]) for r in judged) / len(judged)

    def judged_count(self, *, level: MatchLevel, tag: str | None = None) -> int:
        return len(self._judged(level, tag, allow_none=True))

    def misses(self, *, level: MatchLevel) -> list[QuestionResult]:
        """Judged at this level, but nothing matched in the top depth."""
        return [r for r in self.results if level in r.ranks and r.ranks[level] is None]

    def summarize(self, ks: Sequence[int] | None = None) -> list[MetricRow]:
        """Every level, for all questions and then for each tag; skips empty rows."""
        ks = (
            tuple(k for k in _DEFAULT_KS if k <= self.depth)
            if ks is None
            else tuple(ks)
        )
        tags = sorted({tag for r in self.results for tag in r.question.tags})
        rows: list[MetricRow] = []
        for tag in (None, *tags):
            for level in MATCH_LEVELS:
                judged = self.judged_count(level=level, tag=tag)
                if judged == 0:
                    continue
                rows.append(
                    MetricRow(
                        level=level,
                        tag=tag,
                        judged=judged,
                        hit_rate=MappingProxyType(
                            {k: self.hit_rate(k, level=level, tag=tag) for k in ks}
                        ),
                        mrr=self.mrr(level=level, tag=tag),
                    )
                )
        return rows

    def top_scores(self, *, answerable: bool) -> list[float]:
        """Each question's best score, reranked if it was; to compare answerable with not."""
        return [
            _final_score(r.hits[0])
            for r in self.results
            if r.question.answerable == answerable and r.hits
        ]

    def _judged(
        self, level: MatchLevel, tag: str | None, *, allow_none: bool = False
    ) -> list[QuestionResult]:
        judged = [
            r
            for r in self.results
            if level in r.ranks and (tag is None or tag in r.question.tags)
        ]
        if not judged and not allow_none:
            raise ValueError(
                f"no question is judged at the {level} level"
                + (f" with tag {tag!r}" if tag is not None else "")
            )
        return judged


def evaluate(
    questions: Sequence[EvaluationQuestion], search: Search, *, depth: int = 10
) -> EvaluationReport:
    """Run search on every question and find where its answers first came."""
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    _check_unique_ids(questions)

    results: list[QuestionResult] = []
    for question in questions:
        hits = tuple(search(question.question, limit=depth)[:depth])
        ranks = {
            level: _first_rank(question.answers, hits, level)
            for level in question.levels()
        }
        results.append(QuestionResult(question=question, hits=hits, ranks=ranks))
    return EvaluationReport(depth=depth, results=tuple(results))


def load_questions(path: Path | str) -> list[EvaluationQuestion]:
    """One JSON object per line; blank lines are skipped."""
    path = Path(path)
    questions: list[EvaluationQuestion] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            questions.append(_parse_question(json.loads(line)))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    _check_unique_ids(questions)
    return questions


def format_summary(report: EvaluationReport, ks: Sequence[int] | None = None) -> str:
    """The summary as a plain-text table."""
    rows = report.summarize(ks)
    if not rows:
        return "no questions were judged"
    shown_ks = list(rows[0].hit_rate)
    header = [
        "level",
        "tag",
        "judged",
        *(f"hit@{k}" for k in shown_ks),
        f"MRR@{report.depth}",
    ]
    lines = [
        [
            row.level,
            row.tag or "all",
            str(row.judged),
            *(f"{row.hit_rate[k]:.2f}" for k in shown_ks),
            f"{row.mrr:.2f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(cells[i]) for cells in (header, *lines)) for i in range(len(header))
    ]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()
        for cells in (header, *lines)
    )


def _first_rank(
    answers: Sequence[ExpectedAnswer],
    hits: Sequence[RetrievedChunk],
    level: MatchLevel,
) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if any(answer.match(hit.chunk, level) for answer in answers):
            return rank
    return None


def _reciprocal(rank: int | None) -> float:
    return 0.0 if rank is None else 1 / rank


def _final_score(hit: RetrievedChunk) -> float:
    return hit.rerank_score if hit.rerank_score is not None else hit.index_score


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _check_unique_ids(questions: Sequence[EvaluationQuestion]) -> None:
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise ValueError(f"question id {question.id!r} appears twice")
        seen.add(question.id)


def _parse_question(data: Any) -> EvaluationQuestion:
    if not isinstance(data, dict):
        raise TypeError("each line must be a JSON object")
    _check_fields("question", data, _QUESTION_FIELDS)
    answers = data.get("answers", [])
    tags = data.get("tags", [])
    if not isinstance(answers, list):
        raise TypeError("answers must be a list")
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise TypeError("tags must be a list of strings")
    return EvaluationQuestion(
        id=_require_str(data, "id"),
        question=_require_str(data, "question"),
        answers=tuple(_parse_answer(answer) for answer in answers),
        tags=tuple(tags),
        location=_optional_str(data, "location"),
    )


def _parse_answer(data: Any) -> ExpectedAnswer:
    if not isinstance(data, dict):
        raise TypeError("each answer must be a JSON object")
    _check_fields("answer", data, _ANSWER_FIELDS)
    return ExpectedAnswer(
        document_id=_require_str(data, "document_id"),
        reference_key=_optional_str(data, "reference_key"),
        quote=_optional_str(data, "quote"),
    )


def _check_fields(kind: str, data: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {kind} field(s): {', '.join(unknown)}")


def _require_str(data: dict[str, Any], name: str) -> str:
    value = data[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _optional_str(data: dict[str, Any], name: str) -> str | None:
    value = data.get(name)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value
