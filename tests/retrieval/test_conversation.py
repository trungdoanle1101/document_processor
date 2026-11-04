"""condense_question and recent_turns: follow-ups rewritten to stand on their own."""

import pytest

from document_processor.retrieval.chat_model import Message
from document_processor.retrieval.conversation import (
    CondenseTemplate,
    condense_question,
    recent_turns,
)
from tests.retrieval.fake_chat_model import FakeChatModel

TEMPLATE = CondenseTemplate(
    instructions="Rewrite the last message as a standalone question.",
    request="Conversation:\n{history}\n\nLast message: {message}",
    user="Q",
    assistant="A",
)
HISTORY = [
    Message(role="user", content="Which integration layers are there?"),
    Message(role="assistant", content="External, internal and application [1]."),
]
STANDALONE = "Which gateways does the external integration layer have?"


def test_without_history_the_message_is_kept_and_no_model_is_called() -> None:
    model = FakeChatModel(STANDALONE)
    assert (
        condense_question(
            "  What is Staging?  ", [], chat_model=model, template=TEMPLATE
        )
        == "What is Staging?"
    )
    assert model.calls == []


def test_system_messages_alone_count_as_no_history() -> None:
    model = FakeChatModel(STANDALONE)
    system = [Message(role="system", content="You are helpful.")]
    assert (
        condense_question(
            "What is Staging?", system, chat_model=model, template=TEMPLATE
        )
        == "What is Staging?"
    )
    assert model.calls == []


def test_the_model_sees_the_instructions_the_turns_and_the_message() -> None:
    model = FakeChatModel(STANDALONE)
    result = condense_question(
        "And the external one?",
        HISTORY,
        chat_model=model,
        template=TEMPLATE,
        max_tokens=50,
    )

    assert result == STANDALONE
    ((messages, max_tokens),) = model.calls
    assert max_tokens == 50
    assert messages == [
        Message(role="system", content=TEMPLATE.instructions),
        Message(
            role="user",
            content=(
                "Conversation:\n"
                "Q: Which integration layers are there?\n"
                "A: External, internal and application [1].\n\n"
                "Last message: And the external one?"
            ),
        ),
    ]


@pytest.mark.parametrize(
    "reply",
    [
        f'"{STANDALONE}"',
        f"\n\n  {STANDALONE}  \n",
        f"“{STANDALONE}”\nI rewrote it to name the layer.",
    ],
)
def test_quotes_blank_lines_and_extra_lines_are_dropped(reply: str) -> None:
    assert (
        condense_question(
            "And the external one?",
            HISTORY,
            chat_model=FakeChatModel(reply),
            template=TEMPLATE,
        )
        == STANDALONE
    )


@pytest.mark.parametrize(
    ("reply", "stop"),
    [('   \n "" ', "done"), (STANDALONE, "length"), (STANDALONE, "refused")],
)
def test_an_unusable_rewrite_falls_back_to_the_message(reply: str, stop: str) -> None:
    model = FakeChatModel(reply, stop=stop)  # type: ignore[arg-type]
    assert (
        condense_question(
            "And the external one?", HISTORY, chat_model=model, template=TEMPLATE
        )
        == "And the external one?"
    )


def test_a_blank_message_and_bad_max_tokens_are_refused() -> None:
    with pytest.raises(ValueError, match="no text"):
        condense_question("  ", HISTORY, chat_model=FakeChatModel(), template=TEMPLATE)
    with pytest.raises(ValueError, match="max_tokens"):
        condense_question(
            "Why?", HISTORY, chat_model=FakeChatModel(), template=TEMPLATE, max_tokens=0
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"instructions": " "}, "instructions"),
        ({"request": "{history}"}, "must use message"),
        ({"request": "{history} {message} {extra}"}, "unknown field"),
        ({"turn": "{content}"}, "must use speaker"),
    ],
)
def test_bad_templates_are_refused(changes: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CondenseTemplate(**{"instructions": "Rewrite.", **changes})


# recent_turns -------------------------------------------------------------------------------

CONVERSATION = [
    Message(role="system", content="Be brief."),
    Message(role="user", content="q1"),
    Message(role="assistant", content="a1"),
    Message(role="user", content="q2"),
    Message(role="assistant", content="a2"),
    Message(role="user", content="q3"),
    Message(role="assistant", content="a3"),
]


def contents(messages: list[Message]) -> list[str]:
    return [message.content for message in messages]


def test_the_last_questions_are_kept_with_their_replies() -> None:
    assert contents(recent_turns(CONVERSATION, 2)) == ["q2", "a2", "q3", "a3"]
    assert contents(recent_turns(CONVERSATION, 1)) == ["q3", "a3"]


def test_asking_for_more_turns_than_there_are_keeps_them_all() -> None:
    assert contents(recent_turns(CONVERSATION, 10)) == [
        "q1",
        "a1",
        "q2",
        "a2",
        "q3",
        "a3",
    ]


def test_no_turns_or_no_questions_give_nothing() -> None:
    assert recent_turns(CONVERSATION, 0) == []
    assert recent_turns([Message(role="assistant", content="Hello!")], 2) == []
    with pytest.raises(ValueError, match="negative"):
        recent_turns(CONVERSATION, -1)