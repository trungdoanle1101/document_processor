"""A sources line the model writes itself ("Nguồn: [2][3]") is dropped: the reply adds the real list."""

from pathlib import Path

import pytest

from document_processor.retrieval.chat_model import Message
from knowledge_base.assistant import format_reply
from tests.knowledge_base.helpers import DATA_REGULATION, Fakes, make_fakes, write

QUESTION = "Dữ liệu vùng Staging được lưu trữ bao lâu?"


@pytest.fixture
def fakes(tmp_path: Path) -> Fakes:
    fakes = make_fakes(history_turns=2)
    fakes.indexer.add_document(
        write(tmp_path, "du-lieu.txt", DATA_REGULATION), access_group="it"
    )
    return fakes


def ask(fakes: Fakes, reply: str) -> str:
    fakes.chat_model.reply = reply
    answer = fakes.knowledge_base.ask(QUESTION, groups=["it"])
    assert len(answer.passages) >= 2  # so a stray [2] could have been cited
    return answer.text


@pytest.mark.parametrize(
    "line",
    [
        "Nguồn: [1][2]",
        "Nguồn: [1], [2].",
        "**Nguồn:** [1] [2]",
        "  nguồn : [2]",
        "_Nguồn_: [1];[2]",
    ],
)
def test_a_line_only_listing_sources_is_dropped(fakes: Fakes, line: str) -> None:
    assert ask(fakes, f"Tối đa 07 ngày [1].\n\n{line}") == "Tối đa 07 ngày [1]."


def test_citations_come_from_the_answer_not_the_dropped_line(fakes: Fakes) -> None:
    fakes.chat_model.reply = "Tối đa 07 ngày [1].\n\nNguồn: [1][2]"
    answer = fakes.knowledge_base.ask(QUESTION, groups=["it"])
    assert [citation.number for citation in answer.citations] == [1]


@pytest.mark.parametrize(
    "line",
    [
        "Nguồn dữ liệu: hệ thống ERP [2].",
        "Nguồn: Điều 2, khoản 1.",
        "Theo nguồn: [2] và [1].",
    ],
)
def test_lines_that_say_more_than_numbers_are_kept(fakes: Fakes, line: str) -> None:
    assert ask(fakes, f"Tối đa 07 ngày [1].\n{line}") == f"Tối đa 07 ngày [1].\n{line}"


def test_the_reply_has_one_sources_list(fakes: Fakes) -> None:
    fakes.chat_model.reply = "Tối đa 07 ngày [1].\n\nNguồn: [1][2]"
    kb = fakes.knowledge_base
    reply = format_reply(kb.ask(QUESTION, groups=["it"]), kb)
    assert reply.count("Nguồn:") == 1
    assert reply.startswith("Tối đa 07 ngày [1].\n\nNguồn:\n[1] ")


def test_earlier_replies_lose_both_kinds_of_sources_list(fakes: Fakes) -> None:
    earlier = (
        "Kiến trúc tích hợp gồm ba lớp [2][3].\n\n"
        "Nguồn: [2][3][4]\n\n"
        "Nguồn:\n[2] Quy định › Điều 7 › a)\n[3] Quy định › Điều 7 › b)"
    )
    history = [
        Message(role="user", content="Kiến trúc tích hợp gồm những lớp nào?"),
        Message(role="assistant", content=earlier),
    ]
    fakes.chat_model.replies = [QUESTION, "Tối đa 07 ngày [1]."]
    fakes.knowledge_base.ask(
        "Giải thích chi tiết từng lớp.", groups=["it"], history=history
    )

    for messages, _ in fakes.chat_model.calls:
        for message in messages:
            assert "Nguồn:" not in message.content
    answering = fakes.chat_model.calls[1][0]
    assert answering[2] == Message(
        role="assistant", content="Kiến trúc tích hợp gồm ba lớp [2][3]."
    )


def test_a_not_found_reply_is_left_alone(tmp_path: Path) -> None:
    fakes = make_fakes()
    answer = fakes.knowledge_base.ask(QUESTION, groups=["it"])  # nothing indexed
    assert (answer.found, answer.text) == (
        False,
        fakes.knowledge_base.style.template.not_found,
    )