"""Follow-up questions: rewrite them to stand on their own before searching.

"Còn lớp ngoài thì sao?" can't be searched by itself: what it refers to is in the
turns before it. condense_question asks the chat model to rewrite such a message as
a complete question, so the search finds what a first question would. Keeping the
conversation is the caller's job; chat front ends send it with every message.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from document_processor.retrieval._templates import check_fields
from document_processor.retrieval.chat_model import ChatModel, Message

_REQUEST_FIELDS = frozenset({"history", "message"})
_TURN_FIELDS = frozenset({"speaker", "content"})
_QUOTES = "\"'“”‘’«»"


@dataclass(frozen=True, kw_only=True)
class CondenseTemplate:
    """The words around a rewrite: what a domain or language decides."""

    # the system message: rewrite the last message as one standalone question
    instructions: str
    request: str = "{history}\n\n{message}"  # fields: history, message
    turn: str = "{speaker}: {content}"  # one earlier turn; fields: speaker, content
    user: str = "User"  # who's speaking, as the shown history names them
    assistant: str = "Assistant"
    turn_separator: str = "\n"

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise ValueError("instructions must not be blank")
        check_fields("request", self.request, _REQUEST_FIELDS)
        check_fields("turn", self.turn, _TURN_FIELDS)


def recent_turns(history: Sequence[Message], turns: int) -> list[Message]:
    """The last `turns` questions with the replies after them; system messages left out."""
    if turns < 0:
        raise ValueError(f"turns must not be negative, got {turns}")
    kept = [message for message in history if message.role != "system"]
    questions = [i for i, message in enumerate(kept) if message.role == "user"]
    if turns == 0 or not questions:
        return []
    return kept[questions[-min(turns, len(questions))] :]


def condense_question(
    message: str,
    history: Sequence[Message],
    *,
    chat_model: ChatModel,
    template: CondenseTemplate,
    max_tokens: int = 200,
) -> str:
    """The message as a question that stands on its own, given the turns before it.

    Without history the message is returned as it is, and no model is called. A
    reply that is empty or cut off also gives back the message: never worse than
    not condensing.
    """
    message = message.strip()
    if not message:
        raise ValueError("message has no text")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    turns = [turn for turn in history if turn.role != "system"]
    if not turns:
        return message

    shown = template.turn_separator.join(
        template.turn.format(
            speaker=template.user if turn.role == "user" else template.assistant,
            content=turn.content.strip(),
        )
        for turn in turns
    )
    completion = chat_model.complete(
        [
            Message(role="system", content=template.instructions),
            Message(
                role="user",
                content=template.request.format(history=shown, message=message),
            ),
        ],
        max_tokens=max_tokens,
    )
    rewritten = _first_line(completion.text)
    if completion.stop != "done" or not rewritten:
        return message
    return rewritten


def _first_line(text: str) -> str:
    """The first line with words, without quotes around it: small models add both."""
    for line in text.splitlines():
        cleaned = line.strip().strip(_QUOTES).strip()
        if cleaned:
            return cleaned
    return ""