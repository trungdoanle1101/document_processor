"""Answer a question from retrieved passages, citing them by number.

The framework part of answering: retrieve, decide whether anything relevant was
found, number the passages, ask a chat model, and map its [n] markers back to
chunks. The words around it (instructions, how a passage is shown, what to say
when nothing is found) come from an AnswerTemplate the caller supplies, so a
domain or language is configuration, not code.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from document_processor.retrieval._templates import check_fields
from document_processor.retrieval.chat_model import ChatModel, Message, StopReason
from document_processor.retrieval.index import IndexedChunk
from document_processor.retrieval.search import RetrievedChunk, Search

_CITATION = re.compile(r"\[(\d+)\]")
_PASSAGE_FIELDS = frozenset({"number", "source", "text"})
_QUESTION_FIELDS = frozenset({"passages", "question"})

type SourceOf = Callable[[IndexedChunk], str]


@dataclass(frozen=True, kw_only=True)
class AnswerTemplate:
    """The words around the passages: what a domain or language decides."""

    instructions: str  # the system message: how to answer, and to cite passages as [n]
    not_found: str  # the reply when nothing relevant was found; no model is called
    passage: str = "[{number}] {source}\n{text}"  # fields: number, source, text
    question: str = "{passages}\n\n{question}"  # fields: passages, question
    passage_separator: str = "\n\n"

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise ValueError("instructions must not be blank")
        if not self.not_found.strip():
            raise ValueError("not_found must not be blank")
        check_fields("passage", self.passage, _PASSAGE_FIELDS)
        check_fields("question", self.question, _QUESTION_FIELDS)


@dataclass(frozen=True, kw_only=True)
class Citation:
    number: int  # as the answer wrote it: [number]
    hit: RetrievedChunk


@dataclass(frozen=True, kw_only=True)
class Answer:
    question: str
    text: str
    found: bool  # False: nothing passed the threshold, and text is the template's not_found
    passages: tuple[RetrievedChunk, ...]  # what the model was shown, numbered from 1
    citations: tuple[Citation, ...]  # passages the text cites, in the order first cited
    stop: StopReason | None  # None when no model was called
    chat_model: str | None  # identity of the model that wrote the text


def answer_question(
    question: str,
    *,
    search: Search,
    chat_model: ChatModel,
    template: AnswerTemplate,
    passages: int = 5,
    min_score: float | None = None,
    min_passage_score: float | None = None,
    max_tokens: int = 1024,
    source_of: SourceOf | None = None,
    history: Sequence[Message] = (),
) -> Answer:
    """Answer from the best passages, or say nothing was found.

    min_score: answer only if the best passage scores at least this.
    min_passage_score: leave out passages scoring below this.
    Scores are rerank scores when the search reranks, index scores otherwise.
    history: earlier turns of the conversation, shown to the model before the passages
    so it can stay consistent with them; the search uses the question alone.
    """
    if not question.strip():
        raise ValueError("question has no text")
    if passages <= 0:
        raise ValueError(f"passages must be positive, got {passages}")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    hits = search(question, limit=passages)
    if min_passage_score is not None:
        hits = [hit for hit in hits if score_of(hit) >= min_passage_score]
    if not hits or (min_score is not None and score_of(hits[0]) < min_score):
        return Answer(
            question=question,
            text=template.not_found,
            found=False,
            passages=tuple(hits),
            citations=(),
            stop=None,
            chat_model=None,
        )

    messages = build_messages(
        question, hits, template, source_of=source_of, history=history
    )
    completion = chat_model.complete(messages, max_tokens=max_tokens)
    return Answer(
        question=question,
        text=completion.text,
        found=True,
        passages=tuple(hits),
        citations=find_citations(completion.text, hits),
        stop=completion.stop,
        chat_model=chat_model.identity,
    )


def build_messages(
    question: str,
    passages: Sequence[RetrievedChunk],
    template: AnswerTemplate,
    *,
    source_of: SourceOf | None = None,
    history: Sequence[Message] = (),
) -> list[Message]:
    """The instructions, the earlier turns, then the numbered passages and the question."""
    describe = source_of or describe_source
    shown = template.passage_separator.join(
        template.passage.format(
            number=number, source=describe(hit.chunk), text=hit.chunk.text
        )
        for number, hit in enumerate(passages, start=1)
    )
    return [
        Message(role="system", content=template.instructions),
        *(turn for turn in history if turn.role != "system"),
        Message(
            role="user",
            content=template.question.format(passages=shown, question=question),
        ),
    ]


def find_citations(
    text: str, passages: Sequence[RetrievedChunk]
) -> tuple[Citation, ...]:
    """The passages cited as [n], each once, in the order first cited; other numbers are ignored."""
    citations: dict[int, Citation] = {}
    for match in _CITATION.finditer(text):
        number = int(match.group(1))
        if 1 <= number <= len(passages) and number not in citations:
            citations[number] = Citation(number=number, hit=passages[number - 1])
    return tuple(citations.values())


def describe_source(chunk: IndexedChunk) -> str:
    """The document and where in it: "law › Chương II › Điều 5"."""
    return " › ".join((chunk.document_id, *chunk.headings))


def score_of(hit: RetrievedChunk) -> float:
    """The score that ranked the hit: reranked if it was."""
    return hit.rerank_score if hit.rerank_score is not None else hit.index_score