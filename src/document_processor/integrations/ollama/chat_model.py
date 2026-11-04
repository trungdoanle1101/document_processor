"""A ChatModel served by Ollama, through its native /api/chat endpoint.

The native endpoint is used because it takes the context size per request:
Ollama's default context is small, and a prompt longer than it loses its
beginning (usually the instructions) without any error.

Thinking-only models (e.g. Qwen3 Thinking-2507) ignore the thinking switch and put
their reasoning in the reply, ended by </think>; only what follows it is the answer.
"""

from collections.abc import Sequence
from typing import Any

import httpx

from document_processor.retrieval.chat_model import (
    ChatModelError,
    ChatModelTimeout,
    Completion,
    Message,
    StopReason,
)

_END_OF_THINKING = "</think>"


class OllamaChatModel:
    """Replies come from one Ollama model, with its settings fixed here.

    The identity names the model tag, context size, temperature and thinking switch.
    A tag can be re-pulled to different weights; pin a specific tag for comparable runs.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        context_size: int = 8192,
        # None: the model's default; False turns reasoning off where supported
        think: bool | None = None,
        timeout: float = 120.0,  # seconds to wait for a whole reply
        client: httpx.Client | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if temperature < 0:
            raise ValueError(f"temperature must not be negative, got {temperature}")
        if context_size <= 0:
            raise ValueError(f"context_size must be positive, got {context_size}")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        self._model = model
        self._temperature = temperature
        self._context_size = context_size
        self._think = think
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

        thinking = "" if think is None else ("/think" if think else "/nothink")
        self._identity = f"ollama/{model}/ctx{context_size}/t{temperature:g}{thinking}"

    @property
    def identity(self) -> str:
        return self._identity

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> Completion:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_ctx": self._context_size,
                "num_predict": max_tokens,
            },
        }
        if self._think is not None:
            body["think"] = self._think

        where = f"Ollama at {self._client.base_url}"
        try:
            response = self._client.post("/api/chat", json=body)
        except httpx.TimeoutException as error:
            waited = self._client.timeout.read
            limit = f"within {waited:g} s" if waited is not None else "in time"
            raise ChatModelTimeout(
                f"{where} did not reply with {self._model!r} {limit}"
            ) from error
        except httpx.TransportError as error:
            raise ChatModelError(f"cannot reach {where}: {error}") from error
        if response.status_code == 404:
            raise ChatModelError(
                f"{where} has no model {self._model!r}: run `ollama pull {self._model}`"
            )
        if response.is_error:
            raise ChatModelError(
                f"{where} answered HTTP {response.status_code}: {_error_of(response)}"
            )

        data = response.json()
        return Completion(
            text=_answer_of(data["message"]["content"]), stop=_stop_reason(data)
        )

    def close(self) -> None:
        self._client.close()


def _answer_of(content: str) -> str:
    """The reply without the reasoning a thinking model puts before </think>."""
    _, ended, answer = content.rpartition(_END_OF_THINKING)
    return answer.strip() if ended else content


def _error_of(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", response.text))
    except ValueError:
        return response.text


def _stop_reason(data: dict[str, Any]) -> StopReason:
    # Ollama reports "stop" or "length"; anything else is treated as finished
    return "length" if data.get("done_reason") == "length" else "done"