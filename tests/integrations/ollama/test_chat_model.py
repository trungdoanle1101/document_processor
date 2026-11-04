"""OllamaChatModel: what it sends to /api/chat and how it reads the reply. No Ollama server needed."""

import json
from typing import Any

import httpx
import pytest

from document_processor.integrations.ollama import OllamaChatModel
from document_processor.retrieval.chat_model import (
    ChatModel,
    ChatModelError,
    ChatModelTimeout,
    Completion,
    Message,
)

MESSAGES = [
    Message(role="system", content="Trả lời ngắn."),
    Message(role="user", content="Ai?"),
]


class Recorder:
    """Answers every request with a fixed reply; remembers the request bodies."""

    def __init__(self, reply: dict[str, Any], status: int = 200) -> None:
        self.reply = reply
        self.status = status
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(self.status, json=self.reply)


def model_with(recorder: Recorder, **settings: Any) -> OllamaChatModel:
    client = httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(recorder)
    )
    return OllamaChatModel("qwen3:4b", client=client, **settings)


def reply(content: str = "Theo [1].", done_reason: str = "stop") -> dict[str, Any]:
    return {
        "model": "qwen3:4b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
    }


def test_is_a_chat_model() -> None:
    model: ChatModel = model_with(Recorder(reply()))
    assert model.identity.startswith("ollama/qwen3:4b/")


def test_it_sends_the_messages_and_fixed_settings_to_api_chat() -> None:
    recorder = Recorder(reply())
    model_with(recorder, temperature=0.2, context_size=4096).complete(
        MESSAGES, max_tokens=300
    )
    ((path, body),) = recorder.requests
    assert path == "/api/chat"
    assert body == {
        "model": "qwen3:4b",
        "messages": [
            {"role": "system", "content": "Trả lời ngắn."},
            {"role": "user", "content": "Ai?"},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 300},
    }


def test_the_thinking_switch_is_sent_only_when_chosen() -> None:
    off, default = Recorder(reply()), Recorder(reply())
    model_with(off, think=False).complete(MESSAGES, max_tokens=10)
    model_with(default).complete(MESSAGES, max_tokens=10)
    assert off.requests[0][1]["think"] is False
    assert "think" not in default.requests[0][1]


def test_a_finished_reply_is_done() -> None:
    assert model_with(Recorder(reply("Xin chào"))).complete(
        MESSAGES, max_tokens=10
    ) == Completion(text="Xin chào", stop="done")


def test_a_reply_cut_at_the_token_limit_is_length() -> None:
    completion = model_with(
        Recorder(reply("Theo khoản", done_reason="length"))
    ).complete(MESSAGES, max_tokens=3)
    assert completion.stop == "length"


def test_the_identity_names_every_setting_that_changes_replies() -> None:
    assert model_with(Recorder(reply())).identity == "ollama/qwen3:4b/ctx8192/t0"
    assert model_with(
        Recorder(reply()), temperature=0.7, context_size=4096, think=False
    ).identity == ("ollama/qwen3:4b/ctx4096/t0.7/nothink")


@pytest.mark.parametrize(
    "content",
    [
        "Okay, let me look at [3] and [4].\n</think>\n\nTối đa 07 ngày [1].",
        "<think>\nOkay, let me look at [3] and [4].\n</think>\n\nTối đa 07 ngày [1].",
    ],
)
def test_reasoning_before_the_end_of_thinking_is_left_out(content: str) -> None:
    completion = model_with(Recorder(reply(content))).complete(MESSAGES, max_tokens=10)
    assert completion.text == "Tối đa 07 ngày [1]."


def failing(error: type[httpx.TransportError]) -> OllamaChatModel:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("boom", request=request)

    client = httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
        timeout=30,
    )
    return OllamaChatModel("qwen3:4b", client=client)


def test_a_slow_model_is_a_timeout() -> None:
    with pytest.raises(
        ChatModelTimeout, match="did not reply with 'qwen3:4b' within 30 s"
    ):
        failing(httpx.ReadTimeout).complete(MESSAGES, max_tokens=10)


def test_an_unreachable_server_says_where() -> None:
    with pytest.raises(
        ChatModelError, match="cannot reach Ollama at http://ollama.test"
    ):
        failing(httpx.ConnectError).complete(MESSAGES, max_tokens=10)


def test_a_missing_model_says_how_to_get_it() -> None:
    recorder = Recorder({"error": "model 'qwen3:4b' not found"}, status=404)
    with pytest.raises(ChatModelError, match=r"run `ollama pull qwen3:4b`"):
        model_with(recorder).complete(MESSAGES, max_tokens=10)


def test_a_server_error_carries_ollamas_message() -> None:
    recorder = Recorder({"error": "out of memory"}, status=500)
    with pytest.raises(ChatModelError, match="HTTP 500: out of memory") as raised:
        model_with(recorder).complete(MESSAGES, max_tokens=10)
    assert not isinstance(raised.value, ChatModelTimeout)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"temperature": -1.0}, "temperature must not be negative"),
        ({"context_size": 0}, "context_size must be positive"),
        ({"timeout": 0}, "timeout must be positive"),
    ],
)
def test_bad_settings_are_refused(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        model_with(Recorder(reply()), **settings)


def test_a_blank_model_and_a_bad_max_tokens_are_refused() -> None:
    with pytest.raises(ValueError, match="model must not be blank"):
        OllamaChatModel(" ")
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        model_with(Recorder(reply())).complete(MESSAGES, max_tokens=0)