"""The HTTP API: an OpenAI-compatible chat for Open WebUI, plus search, answer and documents."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from knowledge_base.api import MODEL_ID, create_api
from knowledge_base.settings import parse_settings
from tests.knowledge_base.helpers import (
    DATA_REGULATION,
    LEAVE_REGULATION,
    Fakes,
    make_fakes,
    write,
)

QUESTION = "Dữ liệu vùng Staging được lưu trữ bao lâu?"
SETTINGS = parse_settings(
    {
        "users": {"an@bank.vn": ["it"], "binh@bank.vn": ["hr"]},
        "default_groups": ["public"],
    }
)


@pytest.fixture
def fakes(tmp_path: Path) -> Fakes:
    fakes = make_fakes()
    fakes.indexer.add_document(
        write(tmp_path, "du-lieu.txt", DATA_REGULATION),
        access_group="it",
        title="Dữ liệu",
    )
    fakes.indexer.add_document(
        write(tmp_path, "nghi-phep.txt", LEAVE_REGULATION), access_group="hr"
    )
    return fakes


@pytest.fixture
def client(fakes: Fakes) -> TestClient:
    return TestClient(create_api(fakes.services, SETTINGS))


def chat(question: str, **extra: object) -> dict[str, object]:
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": question}],
        **extra,
    }


def test_health_and_the_one_model(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == [MODEL_ID]


def test_a_chat_reply_carries_the_answer_and_its_sources(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions", json=chat(QUESTION), headers={"X-Reader": "an@bank.vn"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    (choice,) = body["choices"]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    content = choice["message"]["content"]
    assert content.startswith("Dữ liệu được lưu tối đa 07 ngày [1].")
    assert "\n\nNguồn:\n[1] Dữ liệu › " in content


def test_open_webui_users_are_known_by_their_forwarded_email(
    client: TestClient, fakes: Fakes
) -> None:
    client.post(
        "/v1/chat/completions",
        json=chat(QUESTION),
        headers={"X-OpenWebUI-User-Email": "an@bank.vn"},
    )
    assert len(fakes.chat_model.calls) == 1


def test_an_unknown_reader_sees_only_the_default_groups(
    client: TestClient, fakes: Fakes
) -> None:
    body = client.post("/v1/chat/completions", json=chat(QUESTION)).json()
    assert body["choices"][0]["message"]["content"].startswith("Không tìm thấy")
    assert fakes.chat_model.calls == []


def test_the_question_is_the_last_user_message_even_in_parts(
    client: TestClient, fakes: Fakes
) -> None:
    messages = [
        {"role": "user", "content": "Nghỉ phép bao nhiêu ngày?"},
        {"role": "assistant", "content": "12 ngày."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUESTION},
                {"type": "image_url", "image_url": {}},
            ],
        },
    ]
    client.post(
        "/v1/chat/completions",
        json={"messages": messages},
        headers={"X-Reader": "an@bank.vn"},
    )
    ((sent, _),) = fakes.chat_model.calls
    assert sent[-1].content.endswith(QUESTION)


@pytest.mark.parametrize(
    "messages",
    [[{"role": "user", "content": "  "}], [{"role": "system", "content": "Hi"}], []],
)
def test_a_chat_without_a_question_is_refused(
    client: TestClient, messages: list[dict[str, str]]
) -> None:
    assert (
        client.post("/v1/chat/completions", json={"messages": messages}).status_code
        == 400
    )


def test_streaming_sends_the_reply_then_the_finish_then_done(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=chat(QUESTION, stream=True),
        headers={"X-Reader": "an@bank.vn"},
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        line.removeprefix("data: ") for line in response.text.split("\n\n") if line
    ]
    assert events[-1] == "[DONE]"
    first, last = json.loads(events[0]), json.loads(events[1])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["content"].startswith("Dữ liệu được lưu")
    assert (last["choices"][0]["delta"], last["choices"][0]["finish_reason"]) == (
        {},
        "stop",
    )
    assert first["id"] == last["id"]


def test_a_cut_off_answer_finishes_with_length(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.chat_model.stop = "length"
    body = client.post(
        "/v1/chat/completions", json=chat(QUESTION), headers={"X-Reader": "an@bank.vn"}
    ).json()
    assert body["choices"][0]["finish_reason"] == "length"


def test_the_answer_endpoint_keeps_the_structure(client: TestClient) -> None:
    body = client.post(
        "/v1/answer", json={"question": QUESTION}, headers={"X-Reader": "an@bank.vn"}
    ).json()
    assert (body["found"], body["stop"], body["chat_model"]) == (
        True,
        "done",
        "fake/chat",
    )
    assert body["citations"][0]["number"] == 1
    assert {p["document_id"] for p in body["passages"]} == {"du-lieu"}


def test_search_is_limited_to_the_readers_groups(client: TestClient) -> None:
    hits = client.post(
        "/v1/search",
        json={"question": QUESTION, "limit": 10},
        headers={"X-Reader": "binh@bank.vn"},
    )
    assert {h["document_id"] for h in hits.json()["hits"]} == {"nghi-phep"}


@pytest.mark.parametrize(
    ("question", "limit"), [(" ", 5), (QUESTION, 0), (QUESTION, 51)]
)
def test_bad_searches_are_refused(
    client: TestClient, question: str, limit: int
) -> None:
    assert (
        client.post(
            "/v1/search", json={"question": question, "limit": limit}
        ).status_code
        == 400
    )


def test_documents_are_listed_by_group(client: TestClient) -> None:
    listing = client.get("/v1/documents", headers={"X-Reader": "an@bank.vn"}).json()[
        "documents"
    ]
    assert [(d["document_id"], d["title"], d["version"]) for d in listing] == [
        ("du-lieu", "Dữ liệu", 1)
    ]
    assert client.get("/v1/documents").json() == {"documents": []}
