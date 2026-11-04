"""When the chat model fails, readers get a clear 503 or 504, and the CLI a clear message."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from document_processor.retrieval.chat_model import ChatModelError, ChatModelTimeout
from knowledge_base.api import create_api
from knowledge_base.cli import main
from knowledge_base.settings import AppSettings, parse_settings
from knowledge_base.wiring import Services
from tests.knowledge_base.helpers import DATA_REGULATION, Fakes, make_fakes, write

QUESTION = "Dữ liệu vùng Staging được lưu trữ bao lâu?"
TIMEOUT = ChatModelTimeout(
    "Ollama at http://localhost:11434 did not reply within 120 s"
)
MISSING = ChatModelError("Ollama has no model 'qwen3:4b': run `ollama pull qwen3:4b`")


@pytest.fixture
def fakes(tmp_path: Path) -> Fakes:
    fakes = make_fakes()
    fakes.indexer.add_document(
        write(tmp_path, "a.txt", DATA_REGULATION), access_group="public"
    )
    return fakes


@pytest.fixture
def client(fakes: Fakes) -> TestClient:
    return TestClient(create_api(fakes.services, parse_settings({})))


def chat(question: str = QUESTION, **extra: object) -> dict[str, object]:
    return {"messages": [{"role": "user", "content": question}], **extra}


@pytest.mark.parametrize(
    ("error", "status", "kind"),
    [(TIMEOUT, 504, "chat_model_timeout"), (MISSING, 503, "chat_model_unavailable")],
)
@pytest.mark.parametrize("stream", [False, True])
def test_chat_reports_the_failure(
    client: TestClient,
    fakes: Fakes,
    error: Exception,
    status: int,
    kind: str,
    stream: bool,
) -> None:
    fakes.chat_model.error = error
    response = client.post("/v1/chat/completions", json=chat(stream=stream))
    assert response.status_code == status
    body = response.json()
    assert body["detail"] == f"the chat model could not answer: {error}"
    assert body["error"] == {"message": body["detail"], "type": kind}


def test_the_answer_endpoint_reports_it_too(client: TestClient, fakes: Fakes) -> None:
    fakes.chat_model.error = MISSING
    response = client.post("/v1/answer", json={"question": QUESTION})
    assert response.status_code == 503
    assert "ollama pull qwen3:4b" in response.json()["detail"]


def test_search_does_not_need_the_chat_model(client: TestClient, fakes: Fakes) -> None:
    fakes.chat_model.error = MISSING
    assert client.post("/v1/search", json={"question": QUESTION}).status_code == 200


def test_ask_prints_the_failure_and_fails(
    fakes: Fakes, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes.chat_model.error = TIMEOUT
    config = tmp_path / "kb.toml"
    config.write_text("", encoding="utf-8")

    def build(_: AppSettings) -> Services:
        return fakes.services

    assert main(["--config", str(config), "ask", QUESTION], build=build) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"the chat model could not answer: {TIMEOUT}\n"


def test_the_timeout_setting_is_read() -> None:
    assert parse_settings({"chat": {"timeout": 300}}).chat.timeout == 300
    assert parse_settings({}).chat.timeout == 120.0