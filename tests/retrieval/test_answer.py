"""answer_question: retrieve, decide whether anything was found, ask the model, map its citations back."""

from functools import partial

import pytest

from document_processor.retrieval.answer import (
    AnswerTemplate,
    answer_question,
    build_messages,
    describe_source,
    find_citations,
    score_of,
)
from document_processor.retrieval.chat_model import ChatModel, Message
from document_processor.retrieval.in_memory_index import InMemoryChunkIndex
from document_processor.retrieval.index import IndexedChunk
from document_processor.retrieval.search import RetrievedChunk, search_chunks
from tests.retrieval.fake_chat_model import FakeChatModel
from tests.retrieval.stub_embedder import StubEmbedder

TEMPLATE = AnswerTemplate(
    instructions="Answer only from the passages; cite them as [n].",
    not_found="Not found.",
)


def chunk(chunk_id: str, text: str, *headings: str) -> IndexedChunk:
    return IndexedChunk(
        document_id="law",
        chunk_id=chunk_id,
        position=0,
        text=text,
        embedded_text=text,
        headings=headings,
    )


def hit(
    chunk_id: str, score: float, rerank: float | None = None, *headings: str
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk(chunk_id, f"text of {chunk_id}", *headings),
        index_score=score,
        rerank_score=rerank,
    )


class FixedSearch:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits
        self.limits: list[int] = []
        self.questions: list[str] = []

    def __call__(self, question: str, *, limit: int) -> list[RetrievedChunk]:
        self.limits.append(limit)
        self.questions.append(question)
        return self.hits[:limit]


THREE = [hit("a", 0.9, 0.95, "Điều 5"), hit("b", 0.8, 0.40), hit("c", 0.7, 0.02)]


# AnswerTemplate ---------------------------------------------------------------------------


def test_the_fake_is_a_chat_model() -> None:
    model: ChatModel = FakeChatModel()
    assert model.identity == "fake/chat"


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"passage": "[{number}] {text}"}, "passage template must use source"),
        ({"passage": "{number} {source} {text} {page}"}, "unknown field"),
        ({"question": "{question}"}, "question template must use passages"),
        ({"instructions": " "}, "instructions must not be blank"),
        ({"not_found": ""}, "not_found must not be blank"),
    ],
)
def test_a_template_with_wrong_fields_is_refused(
    fields: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AnswerTemplate(**{"instructions": "x", "not_found": "y", **fields})


# Building the prompt ----------------------------------------------------------------------


def test_the_prompt_is_the_instructions_then_numbered_passages_and_the_question() -> (
    None
):
    messages = build_messages("Ai?", THREE[:2], TEMPLATE)
    assert messages[0] == Message(role="system", content=TEMPLATE.instructions)
    assert messages[1] == Message(
        role="user",
        content="[1] law › Điều 5\ntext of a\n\n[2] law\ntext of b\n\nAi?",
    )


def test_how_a_source_is_named_can_be_chosen() -> None:
    messages = build_messages(
        "Ai?", THREE[:1], TEMPLATE, source_of=lambda c: f"DOC-{c.chunk_id}"
    )
    assert messages[1].content.startswith("[1] DOC-a\n")


def test_a_template_decides_the_layout() -> None:
    template = AnswerTemplate(
        instructions="x",
        not_found="y",
        passage="<{number}|{source}> {text}",
        question="Q: {question}\n{passages}",
        passage_separator="\n",
    )
    assert (
        build_messages("Ai?", THREE[:2], template)[1].content
        == "Q: Ai?\n<1|law › Điều 5> text of a\n<2|law> text of b"
    )


def test_describe_source_joins_the_document_and_headings() -> None:
    assert (
        describe_source(chunk("a", "t", "Chương II", "Điều 5"))
        == "law › Chương II › Điều 5"
    )


# Citations --------------------------------------------------------------------------------


def test_citations_map_numbers_to_passages_in_first_cited_order_each_once() -> None:
    citations = find_citations("Theo [2] và [1], cũng như [2].", THREE)
    assert [(c.number, c.hit.chunk.chunk_id) for c in citations] == [(2, "b"), (1, "a")]


def test_numbers_beyond_the_passages_are_ignored() -> None:
    assert find_citations("[0] [4] [10] [3]", THREE)[0].number == 3
    assert len(find_citations("[0] [4] [10] [3]", THREE)) == 1


# answer_question --------------------------------------------------------------------------


def test_an_answer_carries_the_text_passages_citations_and_model() -> None:
    model = FakeChatModel("Được quy định tại [1].")
    answer = answer_question(
        "Ai?",
        search=FixedSearch(THREE),
        chat_model=model,
        template=TEMPLATE,
        passages=2,
    )
    assert answer.found
    assert answer.text == "Được quy định tại [1]."
    assert [h.chunk.chunk_id for h in answer.passages] == ["a", "b"]
    assert [c.hit.chunk.chunk_id for c in answer.citations] == ["a"]
    assert (answer.stop, answer.chat_model) == ("done", "fake/chat")


def test_the_search_is_asked_for_the_passages_and_the_model_gets_max_tokens() -> None:
    search, model = FixedSearch(THREE), FakeChatModel()
    answer_question(
        "Ai?",
        search=search,
        chat_model=model,
        template=TEMPLATE,
        passages=3,
        max_tokens=77,
    )
    assert search.limits == [3]
    assert model.calls[0][1] == 77


def test_nothing_found_returns_the_template_reply_without_calling_the_model() -> None:
    model = FakeChatModel()
    answer = answer_question(
        "Ai?", search=FixedSearch([]), chat_model=model, template=TEMPLATE
    )
    assert (answer.found, answer.text, answer.stop, answer.chat_model) == (
        False,
        "Not found.",
        None,
        None,
    )
    assert model.calls == []


def test_a_best_passage_below_min_score_means_not_found() -> None:
    model = FakeChatModel()
    answer = answer_question(
        "Ai?",
        search=FixedSearch([hit("x", 0.9, 0.2)]),
        chat_model=model,
        template=TEMPLATE,
        min_score=0.5,
    )
    assert not answer.found
    assert [h.chunk.chunk_id for h in answer.passages] == [
        "x"
    ]  # kept, to show what was considered
    assert model.calls == []


def test_min_score_uses_the_rerank_score_when_there_is_one() -> None:
    assert score_of(hit("a", 0.9, 0.1)) == 0.1
    assert score_of(hit("a", 0.9)) == 0.9


def test_passages_below_min_passage_score_are_left_out() -> None:
    model = FakeChatModel()
    answer = answer_question(
        "Ai?",
        search=FixedSearch(THREE),
        chat_model=model,
        template=TEMPLATE,
        min_passage_score=0.3,
    )
    assert [h.chunk.chunk_id for h in answer.passages] == ["a", "b"]
    assert "[3]" not in model.calls[0][0][1].content


def test_a_cut_off_or_refused_reply_is_reported() -> None:
    for stop in ("length", "refused"):
        answer = answer_question(
            "Ai?",
            search=FixedSearch(THREE),
            chat_model=FakeChatModel("…", stop=stop),
            template=TEMPLATE,
        )
        assert answer.found and answer.stop == stop


def test_bad_limits_are_refused() -> None:
    search, model = FixedSearch(THREE), FakeChatModel()
    with pytest.raises(ValueError, match="passages must be positive"):
        answer_question(
            "Ai?", search=search, chat_model=model, template=TEMPLATE, passages=0
        )
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        answer_question(
            "Ai?", search=search, chat_model=model, template=TEMPLATE, max_tokens=0
        )


def test_a_blank_question_is_refused_before_searching() -> None:
    search = FixedSearch(THREE)
    with pytest.raises(ValueError, match="no text"):
        answer_question(
            "  ", search=search, chat_model=FakeChatModel(), template=TEMPLATE
        )
    assert search.limits == []


def test_search_chunks_plugs_in_directly() -> None:
    embedder = StubEmbedder()
    index = InMemoryChunkIndex(
        embedder_identity=embedder.identity, vector_specs=embedder.vector_specs
    )
    chunks = [
        chunk("land", "giao đất nông nghiệp"),
        chunk("tax", "thuế thu nhập cá nhân"),
    ]
    index.replace_document(
        "law", chunks, embedder.encode_passages([c.embedded_text for c in chunks])
    )

    answer = answer_question(
        "giao đất",
        search=partial(search_chunks, embedder=embedder, index=index),
        chat_model=FakeChatModel("Xem [1]."),
        template=TEMPLATE,
        passages=1,
    )
    assert [c.hit.chunk.chunk_id for c in answer.citations] == ["land"]


# Conversation history -----------------------------------------------------------------------


def test_earlier_turns_come_between_the_instructions_and_the_passages() -> None:
    history = [
        Message(role="system", content="from the front end"),
        Message(role="user", content="Có những lớp nào?"),
        Message(role="assistant", content="Ba lớp [1]."),
    ]
    model = FakeChatModel()
    answer_question(
        "Lớp ngoài gồm gì?",
        search=FixedSearch(THREE),
        chat_model=model,
        template=TEMPLATE,
        passages=1,
        history=history,
    )
    ((messages, _),) = model.calls
    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert (
        messages[0].content == TEMPLATE.instructions
    )  # the front end's system message is not used
    assert messages[1:3] == history[1:]
    assert messages[-1].content.endswith("Lớp ngoài gồm gì?")


def test_the_search_sees_the_question_only() -> None:
    search = FixedSearch(THREE)
    history = [Message(role="user", content="Có những lớp nào?")]
    answer_question(
        "Lớp ngoài gồm gì?",
        search=search,
        chat_model=FakeChatModel(),
        template=TEMPLATE,
        history=history,
    )
    assert search.questions == ["Lớp ngoài gồm gì?"]