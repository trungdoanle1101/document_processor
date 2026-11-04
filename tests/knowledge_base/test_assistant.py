"""KnowledgeBase: readers only ever search, see and get answers from what their groups may read."""

from pathlib import Path

import pytest

from document_processor.retrieval.chat_model import Message
from knowledge_base.access import readable_by
from knowledge_base.assistant import format_reply
from tests.knowledge_base.helpers import (
    DATA_REGULATION,
    LEAVE_REGULATION,
    Fakes,
    make_fakes,
    write,
)

QUESTION = "Dữ liệu vùng Staging được lưu trữ bao lâu?"


def two_documents(tmp_path: Path, *, min_score: float | None = None) -> Fakes:
    fakes = make_fakes(min_score=min_score)
    fakes.indexer.add_document(
        write(tmp_path, "du-lieu.txt", DATA_REGULATION),
        access_group="it",
        title="Quy định dữ liệu",
    )
    fakes.indexer.add_document(
        write(tmp_path, "nghi-phep.txt", LEAVE_REGULATION),
        access_group="hr",
        title="Quy định nghỉ phép",
    )
    return fakes


def test_a_reader_needs_a_group() -> None:
    with pytest.raises(ValueError, match="at least one access group"):
        readable_by([])


def test_search_sees_only_the_readers_groups(tmp_path: Path) -> None:
    kb = two_documents(tmp_path).knowledge_base
    assert {
        h.chunk.document_id for h in kb.search(QUESTION, groups=["it"], limit=10)
    } == {"du-lieu"}
    assert {
        h.chunk.document_id for h in kb.search(QUESTION, groups=["hr"], limit=10)
    } == {"nghi-phep"}
    assert kb.search(QUESTION, groups=["public"], limit=10) == []


def test_search_leaves_out_front_and_end_matter(tmp_path: Path) -> None:
    kb = two_documents(tmp_path).knowledge_base
    hits = kb.search(QUESTION, groups=["it", "hr"], limit=50)
    assert hits and {h.chunk.attributes["part"] for h in hits} == {"body"}


def test_documents_lists_only_the_readers_groups(tmp_path: Path) -> None:
    kb = two_documents(tmp_path).knowledge_base
    assert [r.document_id for r in kb.documents(groups=["it"])] == ["du-lieu"]
    assert [r.document_id for r in kb.documents(groups=["it", "hr"])] == [
        "du-lieu",
        "nghi-phep",
    ]


def test_asking_shows_the_model_only_readable_passages_under_their_titles(
    tmp_path: Path,
) -> None:
    fakes = two_documents(tmp_path)
    answer = fakes.knowledge_base.ask(QUESTION, groups=["it"])

    assert answer.found
    assert {h.chunk.document_id for h in answer.passages} == {"du-lieu"}
    ((messages, _),) = fakes.chat_model.calls
    prompt = messages[-1].content
    assert "Quy định dữ liệu › " in prompt
    assert "nghỉ phép" not in prompt
    assert prompt.endswith(f"Câu hỏi: {QUESTION}")


def test_a_reader_with_nothing_to_read_is_told_so_without_calling_the_model(
    tmp_path: Path,
) -> None:
    fakes = two_documents(tmp_path)
    answer = fakes.knowledge_base.ask(QUESTION, groups=["public"])
    assert not answer.found
    assert answer.text == fakes.knowledge_base.style.template.not_found
    assert fakes.chat_model.calls == []


def test_the_reply_lists_the_cited_sources(tmp_path: Path) -> None:
    fakes = two_documents(tmp_path)
    kb = fakes.knowledge_base
    reply = format_reply(kb.ask(QUESTION, groups=["it"]), kb)

    text, heading, source = (
        reply.split("\n")[0],
        reply.split("\n")[2],
        reply.split("\n")[3],
    )
    assert text == "Dữ liệu được lưu tối đa 07 ngày [1]."
    assert heading == "Nguồn:"
    assert source.startswith("[1] Quy định dữ liệu › ")


def test_a_reply_without_citations_is_just_the_text(tmp_path: Path) -> None:
    fakes = two_documents(tmp_path)
    fakes.chat_model.reply = "Không rõ."
    kb = fakes.knowledge_base
    assert format_reply(kb.ask(QUESTION, groups=["it"]), kb) == "Không rõ."


def test_the_model_gets_the_style_instructions(tmp_path: Path) -> None:
    fakes = two_documents(tmp_path)
    fakes.knowledge_base.ask(QUESTION, groups=["it"])
    system = fakes.chat_model.calls[0][0][0]
    assert system == Message(
        role="system", content=fakes.knowledge_base.style.template.instructions
    )
