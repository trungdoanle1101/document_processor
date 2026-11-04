"""Follow-up questions: rewritten from the conversation, then searched and answered."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from document_processor.retrieval.chat_model import Message
from knowledge_base.api import create_api
from knowledge_base.settings import parse_settings
from tests.knowledge_base.helpers import DATA_REGULATION, Fakes, make_fakes, write

FIRST = "Vùng Staging là gì?"
FOLLOW_UP = "Nó được lưu bao lâu?"
STANDALONE = "Dữ liệu vùng Staging được lưu trữ bao lâu?"
REPLY = "Là vùng tập kết dữ liệu [1].\n\nNguồn:\n[1] Quy định quản lý dữ liệu › Điều 2"
HISTORY = [
    Message(role="user", content=FIRST),
    Message(role="assistant", content=REPLY),
]


@pytest.fixture
def fakes(tmp_path: Path) -> Fakes:
    fakes = make_fakes(history_turns=2)
    fakes.indexer.add_document(
        write(tmp_path, "du-lieu.txt", DATA_REGULATION), access_group="it"
    )
    fakes.chat_model.replies = [STANDALONE, "Tối đa 07 ngày [1]."]
    return fakes


def test_a_follow_up_is_rewritten_from_the_conversation_then_answered(
    fakes: Fakes,
) -> None:
    answer = fakes.knowledge_base.ask(FOLLOW_UP, groups=["it"], history=HISTORY)

    assert (answer.question, answer.text, answer.found) == (
        STANDALONE,
        "Tối đa 07 ngày [1].",
        True,
    )
    (condense, _), (answering, _) = fakes.chat_model.calls
    assert condense[0].content == fakes.knowledge_base.style.condense.instructions
    request = condense[1].content
    assert (
        f"Người hỏi: {FIRST}" in request
        and "Trợ lý: Là vùng tập kết dữ liệu [1]." in request
    )
    assert request.rstrip().endswith("Câu hỏi đầy đủ:") and FOLLOW_UP in request
    # the answer prompt asks the standalone question, after the earlier turns
    assert answering[-1].content.endswith(f"Câu hỏi: {STANDALONE}")
    assert answering[1:3] == [
        Message(role="user", content=FIRST),
        Message(role="assistant", content="Là vùng tập kết dữ liệu [1]."),
    ]


def test_old_sources_lists_never_reach_the_model(fakes: Fakes) -> None:
    fakes.knowledge_base.ask(FOLLOW_UP, groups=["it"], history=HISTORY)
    for messages, _ in fakes.chat_model.calls:
        assert all("Nguồn:" not in m.content for m in messages)


def test_a_first_question_costs_no_extra_call(fakes: Fakes) -> None:
    fakes.chat_model.replies = []
    fakes.knowledge_base.ask(STANDALONE, groups=["it"])
    assert len(fakes.chat_model.calls) == 1


def test_only_the_last_turns_are_used(tmp_path: Path) -> None:
    fakes = make_fakes(history_turns=1)
    fakes.indexer.add_document(
        write(tmp_path, "du-lieu.txt", DATA_REGULATION), access_group="it"
    )
    fakes.chat_model.replies = [STANDALONE, "Tối đa 07 ngày [1]."]
    older = [
        Message(role="user", content="Câu hỏi cũ"),
        Message(role="assistant", content="Trả lời cũ"),
    ]

    fakes.knowledge_base.ask(FOLLOW_UP, groups=["it"], history=[*older, *HISTORY])
    request = fakes.chat_model.calls[0][0][1].content
    assert "Câu hỏi cũ" not in request and FIRST in request


def test_with_history_off_a_follow_up_is_searched_as_it_is(tmp_path: Path) -> None:
    fakes = make_fakes(history_turns=0)
    fakes.indexer.add_document(
        write(tmp_path, "du-lieu.txt", DATA_REGULATION), access_group="it"
    )
    answer = fakes.knowledge_base.ask(FOLLOW_UP, groups=["it"], history=HISTORY)
    assert answer.question == FOLLOW_UP
    ((messages, _),) = fakes.chat_model.calls
    assert [m.role for m in messages] == ["system", "user"]


def test_a_failed_rewrite_searches_the_message_itself(fakes: Fakes) -> None:
    fakes.chat_model.replies = ["   ", "Không rõ."]
    assert (
        fakes.knowledge_base.ask(FOLLOW_UP, groups=["it"], history=HISTORY).question
        == FOLLOW_UP
    )


# Through the API ------------------------------------------------------------------------------


@pytest.fixture
def client(fakes: Fakes) -> TestClient:
    settings = parse_settings({"users": {"an@bank.vn": ["it"]}})
    return TestClient(create_api(fakes.services, settings))


def test_a_chat_request_brings_its_own_conversation(
    client: TestClient, fakes: Fakes
) -> None:
    messages = [
        {"role": "system", "content": "Open WebUI's own system prompt"},
        {"role": "user", "content": FIRST},
        {"role": "assistant", "content": REPLY},
        {"role": "user", "content": [{"type": "text", "text": FOLLOW_UP}]},
    ]
    body = client.post(
        "/v1/chat/completions",
        json={"messages": messages},
        headers={"X-Reader": "an@bank.vn"},
    ).json()

    assert body["choices"][0]["message"]["content"].startswith("Tối đa 07 ngày [1].")
    condense, answering = (messages for messages, _ in fakes.chat_model.calls)
    assert FIRST in condense[1].content
    assert all(
        "Open WebUI's own system prompt" not in m.content
        for m in [*condense, *answering]
    )


def test_the_answer_endpoint_shows_what_was_searched(client: TestClient) -> None:
    history = [
        {"role": "user", "content": FIRST},
        {"role": "assistant", "content": REPLY},
    ]
    body = client.post(
        "/v1/answer",
        json={"question": FOLLOW_UP, "history": history},
        headers={"X-Reader": "an@bank.vn"},
    ).json()
    assert (body["question"], body["searched_for"]) == (FOLLOW_UP, STANDALONE)


def test_the_history_turns_setting_is_read() -> None:
    assert parse_settings({"answer": {"history_turns": 4}}).answer.history_turns == 4
    assert parse_settings({}).answer.history_turns == 2
    with pytest.raises(ValueError, match="history_turns"):
        parse_settings({"answer": {"history_turns": -1}})