"""A ChatModel for tests: replies with fixed text (or fails as told) and remembers every conversation."""

from collections.abc import Sequence

from document_processor.retrieval.chat_model import Completion, Message, StopReason


class FakeChatModel:
    def __init__(self, reply: str = "Theo [1].", stop: StopReason = "done") -> None:
        self.reply = reply
        self.stop: StopReason = stop
        self.error: Exception | None = None  # raised instead of replying, when set
        self.replies: list[
            str
        ] = []  # given first, one per call, before falling back to reply
        self.calls: list[tuple[list[Message], int]] = []

    @property
    def identity(self) -> str:
        return "fake/chat"

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> Completion:
        self.calls.append((list(messages), max_tokens))
        if self.error is not None:
            raise self.error
        text = self.replies.pop(0) if self.replies else self.reply
        return Completion(text=text, stop=self.stop)