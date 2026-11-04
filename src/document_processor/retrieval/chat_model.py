"""What the answer layer needs from a language model: a reply to a conversation.

Each backend (Ollama, a hosted API, a fake for tests) is an adapter under
integrations/ that translates this into its own API: where instructions go,
what its stop reasons are called, its options.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

type Role = Literal["system", "user", "assistant"]

# done: finished on its own; length: cut off at max_tokens or the context window;
# refused: the model declined to answer
type StopReason = Literal["done", "length", "refused"]


@dataclass(frozen=True, kw_only=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, kw_only=True)
class Completion:
    text: str  # the answer only: adapters leave out any reasoning the model shows
    stop: StopReason


class ChatModelError(Exception):
    """The model could not reply: unreachable, missing or failing. The message says which."""


class ChatModelTimeout(ChatModelError):
    """The model took longer to reply than the adapter waits."""


class ChatModel(Protocol):
    """Writes a reply to a conversation. Equal identities mean the same model and settings.

    Settings that change replies (temperature, context size, model options) are
    fixed when the adapter is built and named in its identity, so callers can't
    vary them per call and runs stay comparable.
    """

    @property
    def identity(self) -> str: ...

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> Completion:
        """Raises ChatModelError (or ChatModelTimeout) when no reply can be had."""
        ...